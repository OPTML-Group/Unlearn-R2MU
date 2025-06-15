import os
import datetime
import numpy as np
import torch
from torch.optim import AdamW
import tqdm as tqdm
import wandb

from rmu.utils import load_model, get_params, forward_with_cache, get_data
from rmu.metrics import eval_few_shots

def lm_evaluation(model_name):
    eval_few_shots(model_name=model_name, task_list=["mmlu"], output_path=f"{model_name}/mmlu.json")
    torch.cuda.empty_cache()
    eval_few_shots(model_name=model_name, task_list=["wmdp_bio"], output_path=f"{model_name}/wmdp.json")

def run_rmu(updated_model, frozen_model, tokenizer, forget_data_list, retain_data_list, args):
    wandb.init(
        project="rmu-unlearning",
        name=args.output_dir.split("/")[-1],
        group=args.wandb_group,
        config=vars(args)
    )
    rmu_config = vars(args)
    print("====rmu Config====")
    print("\n".join(f"{k}={v}" for k, v in rmu_config.items()))
    print("=====")

    updated_model = updated_model.train()
    params = get_params(updated_model, args.layer_ids, args.param_ids)
    optimizer = AdamW(params, lr=args.lr)
    frozen_module = eval(args.module_str.format(model_name="frozen_model", layer_id=args.layer_id))
    updated_module = eval(args.module_str.format(model_name="updated_model", layer_id=args.layer_id))

    control_vectors_list = []
    for i in range(len(forget_data_list)):
        random_vector = torch.rand(1, 1, updated_model.config.hidden_size, dtype=updated_model.dtype, device=updated_model.device)
        control_vec = random_vector / torch.norm(random_vector) * args.steering_coeff_list[i]
        control_vectors_list.append(control_vec)

    num_batches = min(
        args.max_num_batches,
        min([len(f) for f in forget_data_list]),
        min([len(r) for r in retain_data_list]),
    )

    print("num_batches", num_batches)

    truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side = "right"

    for epoch in range(1):
        print(f"======= Epoch {epoch} =======")
        with tqdm.tqdm(total=num_batches) as pbar:
            for idx in range(num_batches):
                topic_idx = idx % len(forget_data_list)
                batch_idx = idx // len(forget_data_list)
                control_vec = control_vectors_list[topic_idx]
                unlearn_batch = forget_data_list[topic_idx][batch_idx]
                retain_batch = retain_data_list[topic_idx][batch_idx]

                max_length = 512 if topic_idx == 0 else 768
                unlearn_inputs = tokenizer(
                    unlearn_batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length
                ).to(updated_model.device)

                updated_forget_activations = forward_with_cache(
                    updated_model, unlearn_inputs, module=updated_module, no_grad=False
                ).to(updated_model.device)

                unlearn_loss = torch.nn.functional.mse_loss(updated_forget_activations, control_vec)

                retain_inputs = tokenizer(
                    retain_batch, return_tensors="pt", padding=True, truncation=True, max_length=512
                ).to(updated_model.device)
                updated_retain_activations = forward_with_cache(
                    updated_model, retain_inputs, module=updated_module, no_grad=False
                ).to(updated_model.device)
                frozen_retain_activations = forward_with_cache(
                    frozen_model, retain_inputs, module=frozen_module, no_grad=True
                ).to(updated_model.device)

                retain_loss = torch.nn.functional.mse_loss(
                    updated_retain_activations, frozen_retain_activations
                ) * args.alpha[topic_idx]

                # === Unthinking Loss (softmax probability suppression) ===
                reflection_tokens = ["<think>", "Wait", "wait", "but", "Okay", "Hmm",  "Albeit", "However", "But", "Yet", "Still", "Nevertheless", "Though", "Meanwhile", "Whereas"]
                
                reflection_token_ids = [
                    tid for tok in reflection_tokens
                    if (tid := tokenizer.convert_tokens_to_ids(tok)) is not None
                ]

                logits = updated_model(
                    input_ids=unlearn_inputs.input_ids,
                    attention_mask=unlearn_inputs.attention_mask
                ).logits  # (B, L, V)
                probs = torch.softmax(logits, dim=-1)  # Convert to probabilities

                # Gather average probability of reflection tokens
                reflection_probs = torch.stack(
                    [probs[..., tid] for tid in reflection_token_ids], dim=-1  # (B, L, R)
                )
                unthink_prob = reflection_probs.mean(dim=-1)  # (B, L)
                unthink_loss_raw = unthink_prob.mean()  # scalar
                unthink_loss = args.lambda_unthink * unthink_loss_raw if args.lambda_unthink > 0 else torch.tensor(0.0, device=logits.device)

                loss = unlearn_loss + retain_loss + unthink_loss
                # import ipdb; ipdb.set_trace()
                
                # loss = unthink_loss
                
                
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                print(f"loss: {loss.item():.4g} | unlearn: {unlearn_loss.item():.4g} | retain: {retain_loss.item():.4g} | unthink (raw): {unthink_loss_raw.item():.4g} | scaled: {unthink_loss.item():.4g} | param_change: {params[0].grad.abs().mean().item():.4g}")

                # print(f"loss: {loss.item():.4g} | unlearn: {unlearn_loss.item():.4g} | retain: {retain_loss.item():.4g} | unthink (raw): {unthink_loss_raw.item():.4g} | scaled: {unthink_loss.item():.4g} | param_change: {0:.4g}")

                
                wandb.log({
                    "loss": loss.item(),
                    "unlearn_loss": unlearn_loss.item(),
                    "retain_loss": retain_loss.item(),
                    "unthink_loss_raw": unthink_loss_raw.item(),
                    "unthink_loss_scaled": unthink_loss.item(),
                    "param_change": params[0].grad.abs().mean().item(),
                    "step": idx
                })
                
                # wandb.log({
                #     "loss": loss.item(),
                #     "unlearn_loss": unlearn_loss.item(),
                #     "retain_loss": retain_loss.item(),
                #     "unthink_loss_raw": unthink_loss_raw.item(),
                #     "unthink_loss_scaled": unthink_loss.item(),
                #     "param_change": 0,
                #     "step": idx
                # })

                if args.verbose:
                    frozen_forget_activations = forward_with_cache(frozen_model, unlearn_inputs, module=frozen_module, no_grad=True).to(updated_model.device)
                    unlearn_cosine = torch.nn.functional.cosine_similarity(updated_forget_activations, frozen_forget_activations, dim=-1).mean()
                    retain_cosine = torch.nn.functional.cosine_similarity(updated_retain_activations, frozen_retain_activations, dim=-1).mean()
                    print(f"unlearn_cosine_sim={unlearn_cosine.item():.4g}")
                    print(f"retain_cosine_sim={retain_cosine.item():.4g}")

                pbar.update(1)

                if idx != 0 and ((idx + 1) % 200 == 0 or (idx + 1) == num_batches):
                    tokenizer.truncation_side = truncation_side
                    path = os.path.join(args.output_dir, f"checkpoint_{idx}") if args.output_dir else f"models/{args.model_name_or_path}_step-{idx}_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
                    updated_model.save_pretrained(path)
                    tokenizer.save_pretrained(path)
                    print(f"Saved model to {path}")

    return path

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_group", type=str, default="default", help="Group name for wandb runs")
    parser.add_argument("--model_name_or_path", type=str, default="HuggingFaceH4/zephyr-7b-beta")
    parser.add_argument("--module_str", type=str, default="{model_name}.model.layers[{layer_id}]")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--retain_corpora", type=str, default="wikitext,wikitext")
    parser.add_argument("--forget_corpora", type=str, default="bio-forget-corpus,cyber-forget-corpus")
    parser.add_argument("--alpha", type=str, default="100,100")
    parser.add_argument("--steering_coeffs", type=str, default="20,20")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min_len", type=int, default=0)
    parser.add_argument("--max_len", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_num_batches", type=int, default=80)
    parser.add_argument("--layer_id", type=int, default=7)
    parser.add_argument("--layer_ids", type=str, default="5,6,7")
    parser.add_argument("--param_ids", type=str, default="6")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--lambda_unthink", type=float, default=0.0, help="Weight for unthinking logits suppression loss")
    args = parser.parse_args()
    args.retain_corpora = args.retain_corpora.split(",")
    args.forget_corpora = args.forget_corpora.split(",")
    args.steering_coeff_list = [float(c) for c in args.steering_coeffs.split(",")]
    args.alpha = [float(c) for c in args.alpha.split(",")]
    args.layer_ids = [int(x) for x in args.layer_ids.split(",")]
    args.param_ids = [int(x) for x in args.param_ids.split(",")]
    return args

if __name__ == "__main__":
    args = get_args()
    SEED = args.seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)

    frozen_model, tokenizer = load_model(args.model_name_or_path)
    updated_model, tokenizer = load_model(args.model_name_or_path)
    forget_data_list, retain_data_list = get_data(
        args.forget_corpora, args.retain_corpora, args.min_len, args.max_len, args.batch_size
    )
    path = run_rmu(updated_model, frozen_model, tokenizer, forget_data_list, retain_data_list, args)
    lm_evaluation(path)


# CUDA_VISIBLE_DEVICES=0,1 python3 -m rmu.unlearn_think --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Llama-8B --max_num_batches 2 --batch_size 4 --retain_corpora wikitext --forget_corpora original --steering_coeffs 6.5,6.5 --alpha 3,3 --lr 1e-3 --seed 42 --output_dir models/R1_llama_8B_test --lambda_unthink 0.0 --verbose

# 