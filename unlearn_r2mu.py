import os
import datetime
import json
import torch
import numpy as np
import tqdm as tqdm
import random
import csv
from glob import glob
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb

from rmu.utils import load_model, get_params, forward_with_cache, get_data
from rmu.metrics import eval_few_shots

# os.environ["WANDB_MODE"] = "offline"
random.seed(0)

def get_dataset_with_generated(raw_path, generated_path, tokenizer, max_gen_tokens=100, min_len=50, batch_size=4):
    generated_data = []
    with open(generated_path, "r") as f:
        for line in f:
            gen_text = json.loads(line)["512"]
            token_ids = tokenizer.encode(gen_text, add_special_tokens=False)[:max_gen_tokens]
            truncated = tokenizer.decode(token_ids, skip_special_tokens=True)
            generated_data.append(truncated)

    paired_data = ["Let's reason this step by step.\n<think>" + gen for gen in generated_data]

    paired_batches = [
        paired_data[i:i + batch_size] for i in range(0, len(paired_data), batch_size)
    ]

    return [paired_batches]  


def get_s1k_dataset(tokenizer, batch_size=4, min_len=50, max_len=2000):
    from datasets import load_dataset

    dataset = load_dataset("simplescaling/s1K", split="train")

    def map_func(examples):
        messages = []
        questions = examples["question"]
        answers = examples["deepseek_attempt"]
        thoughts = examples["deepseek_thinking_trajectory"]
        for q, a, t in zip(questions, answers, thoughts):
            content = "<think>{}</think>{}".format(t, a)
            messages.append([
                {"role": "user", "content": q},
                {"role": "assistant", "content": content}
            ])
        examples["messages"] = messages
        return examples

    dataset = dataset.map(map_func, batched=True)
    dataset = dataset.remove_columns(["question", "deepseek_attempt", "deepseek_thinking_trajectory"])

    user_list = []
    assistant_list = []
    reasoning_list = []

    for ex in dataset:
        user_msg = [m["content"] for m in ex["messages"] if m["role"] == "user"][0]
        assistant_msg = [m["content"] for m in ex["messages"] if m["role"] == "assistant"][0]

        user_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=False,
        )
        assistant_response = tokenizer.apply_chat_template(
            [{"role": "assistant", "content": assistant_msg}],
            tokenize=False,
            add_generation_prompt=False,
        )


        user_list.append(user_prompt)
        assistant_list.append(assistant_response)
        reasoning_list.append("Let's reason this step by step.\n<think>" + assistant_response)

    def batchify(lst):
        return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]

    return batchify(user_list), batchify(assistant_list), batchify(reasoning_list)


def get_limo_dataset(tokenizer, batch_size=4, min_len=50, max_len=2000):
    from datasets import load_dataset

    dataset = load_dataset("GAIR/LIMO",split="train")

    def map_func(examples):
        messages = []
        questions = examples["question"]
        solutions = examples["solution"]
        # COT_TEMPLATE = "<think>{}</think>\n\n**Final Answer**{}"
        COT_TEMPLATE = "{}</think>\n\n"
        for q, s in zip(questions, solutions):
            # thought = s
            # answer = s.split("**Final Answer**")[-1]
            
            content = COT_TEMPLATE.format(s)
            messages.append([
                {"role": "user", "content": q},
                {"role": "assistant", "content": content}
            ])
        examples["messages"] = messages
        return examples

    dataset = dataset.map(map_func, batched=True)
    dataset = dataset.remove_columns(["question", "solution"])

    user_list = []
    assistant_list = []
    reasoning_list = []

    for ex in dataset:
        user_msg = [m["content"] for m in ex["messages"] if m["role"] == "user"][0]
        assistant_msg = [m["content"] for m in ex["messages"] if m["role"] == "assistant"][0]

        user_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=False,
        )
        # assistant_response = tokenizer.apply_chat_template(
        #     [{"role": "assistant", "content": assistant_msg}],
        #     tokenize=False,
        #     add_generation_prompt=False,
        # )

        user_list.append(user_prompt)
        assistant_list.append(assistant_msg)
        reasoning_list.append("Let's reason this step by step.\n<think>" + assistant_msg)
    
    user_list = user_list * 3
    assistant_list = assistant_list * 3
    reasoning_list = reasoning_list * 3

    def batchify(lst):
        return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]

    return [batchify(user_list)], [batchify(assistant_list)], [batchify(reasoning_list)]

# ============ LM Evaluation ============
def lm_evaluation(model_name):
    torch.cuda.empty_cache()
    eval_few_shots(model_name=model_name, task_list=["wmdp_bio"], output_path=f"{model_name}/wmdp.json")
    torch.cuda.empty_cache()
    eval_few_shots(model_name=model_name, task_list=["mmlu"], output_path=f"{model_name}/mmlu.json")

# ============ Save Results ============
def save_experiment_metrics(args, model_path, output_csv_path="/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/unlearn_reasoning_trace_qw14b.csv"):
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    if not os.path.exists(output_csv_path):
        with open(output_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "name", "model_name", "seed", "data_number", "batch_size", "steering_coeffs",
                "alpha", "lr", "reflection_num" ,"mmlu_acc", "wmdp_acc"
            ])

    name = os.path.join(*model_path.strip("/").split("/")[-2:])
    mmlu_file = glob(os.path.join(model_path, "mmlu.json", "*", "*.json"))
    mmlu_acc = None
    if mmlu_file:
        with open(mmlu_file[0], "r") as f:
            mmlu_data = json.load(f)
            mmlu_acc = mmlu_data["results"]["mmlu"]["acc,none"]

    wmdp_file = glob(os.path.join(model_path, "wmdp.json", "*", "*.json"))
    wmdp_acc = None
    if wmdp_file:
        with open(wmdp_file[0], "r") as f:
            wmdp_data = json.load(f)
            wmdp_acc = wmdp_data["results"]["wmdp_bio"]["acc,none"]

    with open(output_csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            name,
            args.model_name_or_path,
            args.seed,
            args.max_num_batches,  
            args.batch_size,
            args.steering_coeffs,
            args.alpha,
            f"{args.lr:.1e}",
            args.reflection_num,
            f"{mmlu_acc:.6f}" if mmlu_acc is not None else "NA",
            f"{wmdp_acc:.6f}" if wmdp_acc is not None else "NA"
        ])

# ============ Main RMU Training Loop ============
def run_rmu(updated_model, frozen_model, tokenizer, forget_data_list, retain_data_list, extent_data_list, user_data_list, assistant_data_list, reasoning_data_list,args):
    wandb.init(
        project="unlearn_v11",
        name=args.output_dir.split("/")[-1],
        group=args.wandb_group,
        config=vars(args)
    )

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
        min([len(e) for e in extent_data_list]),
    )

    tokenizer.truncation_side = "right"

    for epoch in range(1):
        with tqdm.tqdm(total=num_batches) as pbar:
            for idx in range(num_batches):
                topic_idx = idx % len(forget_data_list)
                batch_idx = idx // len(forget_data_list)
                control_vec = control_vectors_list[topic_idx]
                unlearn_batch = forget_data_list[topic_idx][batch_idx]
                retain_batch = retain_data_list[topic_idx][batch_idx]
                extent_batch = extent_data_list[topic_idx][batch_idx]

                
                user_batch = user_data_list[topic_idx][batch_idx]
                assistant_batch = assistant_data_list[topic_idx][batch_idx]
                reasoning_batch = reasoning_data_list[topic_idx][batch_idx]
                
                # import ipdb; ipdb.set_trace()
                
                # print(unlearn_batch)
                # print(extent_batch)
                # import sys
                # sys.exit()
                
                unlearn_inputs = tokenizer(unlearn_batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(updated_model.device)
                retain_inputs = tokenizer(retain_batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(updated_model.device)
                extent_inputs = tokenizer(extent_batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(updated_model.device)

                updated_forget_activations = forward_with_cache(updated_model, unlearn_inputs, module=updated_module, no_grad=False).to(updated_model.device)
                updated_retain_activations = forward_with_cache(updated_model, retain_inputs, module=updated_module, no_grad=False).to(updated_model.device)
                frozen_retain_activations = forward_with_cache(frozen_model, retain_inputs, module=frozen_module, no_grad=True).to(updated_model.device)
                extent_activations = forward_with_cache(updated_model, extent_inputs, module=updated_module, no_grad=False).to(updated_model.device)

                unlearn_loss = torch.nn.functional.mse_loss(updated_forget_activations, control_vec)
                retain_loss = torch.nn.functional.mse_loss(updated_retain_activations, frozen_retain_activations) * args.alpha[topic_idx]
                extent_loss = torch.nn.functional.mse_loss(extent_activations, control_vec)
                # loss = unlearn_loss + retain_loss + extent_loss
                
                user_inputs = tokenizer(user_batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(updated_model.device)
                assistant_inputs = tokenizer(assistant_batch, return_tensors="pt", padding=True, truncation=True, max_length=150).to(updated_model.device)
                reasoning_inputs = tokenizer(reasoning_batch, return_tensors="pt", padding=True, truncation=True, max_length=150).to(updated_model.device)

                updated_user_activations = forward_with_cache(updated_model, user_inputs, module=updated_module, no_grad=False).to(updated_model.device)
                updated_assistant_activations = forward_with_cache(updated_model, assistant_inputs, module=updated_module, no_grad=False).to(updated_model.device)
                updated_reasoning_activations = forward_with_cache(updated_model, reasoning_inputs, module=updated_module, no_grad=False).to(updated_model.device)

                frozen_user_activations = forward_with_cache(frozen_model, user_inputs, module=frozen_module, no_grad=True).to(updated_model.device)
                frozen_assistant_activations = forward_with_cache(frozen_model, assistant_inputs, module=frozen_module, no_grad=True).to(updated_model.device)
                frozen_reasoning_activations = forward_with_cache(frozen_model, reasoning_inputs, module=frozen_module, no_grad=True).to(updated_model.device)

                user_loss = torch.nn.functional.mse_loss(updated_user_activations, frozen_user_activations)
                assistant_loss = torch.nn.functional.mse_loss(updated_assistant_activations, frozen_assistant_activations)
                reasoning_loss = torch.nn.functional.mse_loss(updated_reasoning_activations, frozen_reasoning_activations)
                
                frozen_forget_activations = forward_with_cache(frozen_model, unlearn_inputs, module=frozen_module, no_grad=True).to(updated_model.device)
                unlearn_cosine = torch.nn.functional.cosine_similarity(updated_forget_activations, frozen_forget_activations, dim=-1).mean()
                retain_cosine = torch.nn.functional.cosine_similarity(updated_retain_activations, frozen_retain_activations, dim=-1).mean()
                # extent_cosine = torch.nn.functional.cosine_similarity(extent_activations, control_vec.expand_as(extent_activations), dim=-1).mean()

                frozen_extent_activations = forward_with_cache(frozen_model, extent_inputs, module=frozen_module, no_grad=True).to(updated_model.device)

                extent_cosine = torch.nn.functional.cosine_similarity(extent_activations,frozen_extent_activations,dim=-1).mean()
                
                user_cosine = torch.nn.functional.cosine_similarity(updated_user_activations, frozen_user_activations, dim=-1).mean()
                assistant_cosine = torch.nn.functional.cosine_similarity(updated_assistant_activations, frozen_assistant_activations, dim=-1).mean()
                reasoning_cosine = torch.nn.functional.cosine_similarity(updated_reasoning_activations, frozen_reasoning_activations, dim=-1).mean()
                

                # loss = retain_loss + extent_loss             
                
                
                
                # loss = unlearn_loss + retain_loss + extent_loss + args.assist_loss * assistant_loss
                
                # loss = assistant_loss
                
                loss = unlearn_loss + retain_loss + extent_loss + args.assist_loss * assistant_loss
                
                # loss = unlearn_loss + retain_loss + extent_loss + 0.5 * (reasoning_loss +assistant_loss)
                
                # + reasoning_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # wandb.log({
                #     "loss": loss.item(),
                #     "unlearn_loss": unlearn_loss.item(),
                #     "retain_loss": retain_loss.item(),
                #     "extent_loss": extent_loss.item(),
                #     "step": idx
                # })
                
                wandb.log({
                    "loss": loss.item(),
                    "unlearn_loss": unlearn_loss.item(),
                    "retain_loss": retain_loss.item(),
                    "extent_loss": extent_loss.item(),
                    "user_loss": user_loss.item(),
                    "assistant_loss": assistant_loss.item(),
                    "reasoning_loss": reasoning_loss.item(),

                    "unlearn_cosine": unlearn_cosine.item(),
                    "retain_cosine": retain_cosine.item(),
                    "unlearn_reasoning_trace_cosine": extent_cosine.item(),
                    "reatain_question_cosine": user_cosine.item(),
                    "reatain_trace_cosine": assistant_cosine.item(),
                    "reatain_think_trace_cosine": reasoning_cosine.item(),

                    "step": idx
                })

                pbar.update(1)

                if idx != 0 and ((idx + 1) % 200 == 0 or (idx + 1) == num_batches):
                    path = os.path.join(args.output_dir, f"checkpoint_{idx}")
                    updated_model.save_pretrained(path)
                    tokenizer.save_pretrained(path)
                    print(f"Saved model to {path}")

    return path

# ============ Argument Parsing ============
def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb_group", type=str, default="default")
    parser.add_argument("--model_name_or_path", type=str)
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
    parser.add_argument("--assist_loss", type=float, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--reflection_num", type=int, default=0)
    parser.add_argument("--generated_path", type=str, default="generated_all.jsonl")
    parser.add_argument("--raw_path", type=str, default="/egr/research-optml/wangc168/watermark/IBM_Run_test/dataset/bio_remove_dataset.jsonl")
    parser.add_argument("--max_gen_tokens", type=int, default=100)
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
    updated_model, _ = load_model(args.model_name_or_path)
    forget_data_list, retain_data_list = get_data(
        args.forget_corpora, args.retain_corpora, args.min_len, args.max_len, args.batch_size
    )
    extent_data_list = get_dataset_with_generated(
        raw_path=args.raw_path,
        generated_path=args.generated_path,
        tokenizer=tokenizer,
        max_gen_tokens=args.max_gen_tokens,
        batch_size=args.batch_size
    )
    

    # user_data_list, assistant_data_list, reasoning_data_list = get_s1k_dataset(tokenizer, batch_size=args.batch_size, min_len=args.min_len, max_len=args.max_len)
    
    user_data_list, assistant_data_list, reasoning_data_list = get_limo_dataset(
            tokenizer, batch_size=args.batch_size, min_len=args.min_len, max_len=args.max_len
        )
    
    
    path = run_rmu(updated_model, frozen_model, tokenizer, forget_data_list, retain_data_list, extent_data_list, user_data_list, assistant_data_list, reasoning_data_list, args)

    del updated_model, frozen_model, tokenizer, forget_data_list, retain_data_list, extent_data_list
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    lm_evaluation(path)
    save_experiment_metrics(args, path)
