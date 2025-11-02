import os
import json
import torch.nn as nn
import torch
import torch.nn.functional as F
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from fractions import Fraction

# =========== 配置部分 ==============
model_data = {
    "llama8b": {
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "tokenizer_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "special_token_id": 128014,
    },
    "qwen1.5b": {
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "tokenizer_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "special_token_id": 151649,
    },
    "qwen7b": {
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "tokenizer_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "special_token_id": 151649,
    },
    "qwen14b": {
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "tokenizer_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "special_token_id": 151649,
    },
    "qwen32b": {
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "tokenizer_name": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "special_token_id": 151649,
        "Wait_token_id": 14190,
    },
    "qwen1.5b-org": {
        "model_name": "Qwen/Qwen2.5-Math-1.5B",
        "tokenizer_name": "Qwen/Qwen2.5-Math-1.5B",
        "special_token_id": None,
    },
    "qwen7b-org": {
        "model_name": "Qwen/Qwen2.5-Math-7B",
        "tokenizer_name": "Qwen/Qwen2.5-Math-7B",
        "special_token_id": None,
    },
    "qwen14b-org": {
        "model_name": "Qwen/Qwen2.5-14B",
        "tokenizer_name": "Qwen/Qwen2.5-14B",
        "special_token_id": None,
    },
    "qwen32b-org": {
        "model_name": "Qwen/Qwen2.5-32B",
        "tokenizer_name": "Qwen/Qwen2.5-32B",
        "special_token_id": None,
    },
    "qihu32b": {
        "model_name": "qihoo360/TinyR1-32B-Preview",
        "tokenizer_name": "qihoo360/TinyR1-32B-Preview",
        "special_token_id": 151649,
    },
    "jhft7b":{
        "model_name": "flyingbugs/OpenR1-Qwen-7B-SFT",
        "tokenizer_name": "flyingbugs/OpenR1-Qwen-7B-SFT",
    },
    "deepseek_wmdp": {
        "model_name": "/egr/research-optml/chongyu/wmdp/deepseek_wmdp_model/deepseek_7.5e-05_200_5_42/checkpoint-200",
        "tokenizer_name": "/egr/research-optml/chongyu/wmdp/deepseek_wmdp_model/deepseek_7.5e-05_200_5_42/checkpoint-200",
        "special_token_id": 128014,
    },
    "deepseek_gsm8k": {
        "model_name": "/egr/research-optml/chongyu/wmdp/deepseek_gsm8k_model/deepseek_5e-06_200_100_42/checkpoint-200",
        "tokenizer_name": "/egr/research-optml/chongyu/wmdp/deepseek_gsm8k_model/deepseek_5e-06_200_100_42/checkpoint-200",
        "special_token_id": 128014,
    },

    "meta_wmdp": {
        "model_name": "/egr/research-optml/chongyu/wmdp/deepseek_wmdp_model/meta_6e-05_200_50_42/checkpoint-200",
        "tokenizer_name": "/egr/research-optml/chongyu/wmdp/deepseek_wmdp_model/meta_6e-05_200_50_42/checkpoint-200",
        "special_token_id": 128014,
    },
    
    "reasoning_unlearn": {
        "model_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/Final_ckpt/unlearned_reason/checkpoint-200",
        "tokenizer_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/Final_ckpt/unlearned_reason/checkpoint-200",
        "special_token_id": 128014,
    },

    # "meta_wmdp": {
    #     "model_name": "/egr/research-optml/chongyu/wmdp/deepseek_wmdp_model/meta_6e-05_200_100_42/checkpoint-200",
    #     "tokenizer_name": "/egr/research-optml/chongyu/wmdp/deepseek_wmdp_model/meta_6e-05_200_100_42/checkpoint-200",
    #     "special_token_id": 128014,
    # },
    # "meta_gsm8k": {
    #     "model_name": "/egr/research-optml/chongyu/wmdp/deepseek_gsm8k_model/meta_5e-05_200_750_42/checkpoint-200",
    #     "tokenizer_name": "/egr/research-optml/chongyu/wmdp/deepseek_gsm8k_model/meta_5e-05_200_750_42/checkpoint-200",
    #     "special_token_id": 128014,
    # },

    "meta": {
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "tokenizer_name": "meta-llama/Llama-3.1-8B-Instruct",
        "special_token_id": 128014,
    },
    "deepseek": {
        "model_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "tokenizer_name": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "special_token_id": 128014,
    },
    "non_reasoning_llama":{
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "tokenizer_name": "meta-llama/Llama-3.1-8B-Instruct",
        "special_token_id": 128014,
    },
    "non_reasoning_unlearned_llama":{
        "model_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/Final_ckpt/unlearned_non_reason/checkpoint-200",
        "tokenizer_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/Final_ckpt/unlearned_non_reason/checkpoint-200",
        "special_token_id": 128014,
    },
    "reasoning_unlearned_llama_unthink":{
        "model_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/R1_llama_8B_unthink_prob_1_unlearn_loss/checkpoint_199",
        "tokenizer_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/R1_llama_8B_unthink_prob_1_unlearn_loss/checkpoint_199",
        "special_token_id": 128014,
    },
    "llama_start_end_think_middle_bio":{
        "model_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/unlearn_v3_3/checkpoint_199",
        "tokenizer_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/unlearn_v3_3/checkpoint_199",
        "special_token_id": 128014,
    },
    "llama_start_end_think_middle_bio_insert_reflection":{
        "model_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/unlearn_v3_7/checkpoint_199",
        "tokenizer_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/unlearn_v3_7/checkpoint_199",
        "special_token_id": 128014,
    },
     "llama_start_end_think_middle_bio_insert_reflection_refle_loss":{
        "model_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/unlearn_v3_refle_loss_3/checkpoint_199",
        "tokenizer_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/unlearn_v3_refle_loss_3/checkpoint_199",
        "special_token_id": 128014,
    },
    
    "reasoning_none": {
        "model_name": "/egr/research-optml/wangc168/watermark/LLM_unlearn/files/results/reasoning_random/2025-05-04-23-18-54-123188/checkpoints",
        "tokenizer_name": "/egr/research-optml/wangc168/watermark/LLM_unlearn/files/results/reasoning_random/2025-05-04-23-18-54-123188/checkpoints",
        "special_token_id": 128014
    },
    "reasoning_random": {
        "model_name": "/egr/research-optml/wangc168/watermark/LLM_unlearn/files/results/reasoning_none/2025-05-04-22-51-42-859781/checkpoints",
        "tokenizer_name": "/egr/research-optml/wangc168/watermark/LLM_unlearn/files/results/reasoning_none/2025-05-04-22-51-42-859781/checkpoints",
        "special_token_id": 128014
    },
    "reasoning_random_v1": {
        "model_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/Unlearn-Smooth/WMDP/files/results/reasoning_random/2025-05-05-16-01-52-820603/checkpoints",
        "tokenizer_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/Unlearn-Smooth/WMDP/files/results/reasoning_random/2025-05-05-16-01-52-820603/checkpoints",
        "special_token_id": 128014
    },
    "reasoning_random_rmu": {
        "model_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/unlearn_v4_alpha3x3_lr7.5e-5/checkpoint_199",
        "tokenizer_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/unlearn_v4_alpha3x3_lr7.5e-5/checkpoint_199",
        "special_token_id": 128014
    },
    "reasoning_random_rmu_v1": {
        "model_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/unlearn_v4_alpha5x5_lr7.5e-5_data500/checkpoint_499",
        "tokenizer_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/unlearn_v4_alpha5x5_lr7.5e-5_data500/checkpoint_499",
        "special_token_id": 128014
    },
    "reasoning_random_rmu_v5": {
        "model_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/unlearn_v5_alpha3x3_lr7.5e-5_data500/checkpoint_499",
        "tokenizer_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/unlearn_v5_alpha3x3_lr7.5e-5_data500/checkpoint_499",
        "special_token_id": 128014
    },
    "NPO_reasoning": {
        "model_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/Unlearn-Smooth/WMDP/files/results/unlearn_llama3_r1_beta001_lr_3--05_gamma15.0_dataset_seed1000/2025-05-05-14-10-03-612361/checkpoints",
        "tokenizer_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/Unlearn-Smooth/WMDP/files/results/unlearn_llama3_r1_beta001_lr_3--05_gamma15.0_dataset_seed1000/2025-05-05-14-10-03-612361/checkpoints",
        "special_token_id": 128014
    },
    "RMU_unthink": {
        "model_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/unthink_alpha3x3_lr5e-5_data500/checkpoint_499",
        "tokenizer_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/unthink_alpha3x3_lr5e-5_data500/checkpoint_499",
        "special_token_id": 128014
    },
    "RMU_unlearn_v7": {
        "model_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/unlearn_v7_alpha3x3_lr7.5e-5_data500/checkpoint_499",
        "tokenizer_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/unlearn_v7_alpha3x3_lr7.5e-5_data500/checkpoint_499",
        "special_token_id": 128014
    },
    "RMU_unlearn_v9": {
        "model_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/unlearn_v9_alpha3x3_lr2e-5_data500/checkpoint_499",
        "tokenizer_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/unlearn_v9_alpha3x3_lr2e-5_data500/checkpoint_499",
        "special_token_id": 128014
    },
    "RMU_unlearn_v10": {
        "model_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/unlearn_v10_alpha3x3_lr3e-5_data500/checkpoint_499",
        "tokenizer_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/unlearn_v10_alpha3x3_lr3e-5_data500/checkpoint_499",
        "special_token_id": 128014
    },
    "RMU_unlearn_utility": {
        "model_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/unthink_alpha3x3_lr7.5e-5_data500_utility/checkpoint_499",
        "tokenizer_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/unthink_alpha3x3_lr7.5e-5_data500_utility/checkpoint_499",
        "special_token_id": 128014
    },
    "RMU_unlearn_assistant_loss": {
        "model_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/v11_alpha3x3_lr7.5e-5_data500_assistant_loss/checkpoint_499",
        "tokenizer_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/v11_alpha3x3_lr7.5e-5_data500_assistant_loss/checkpoint_499",
        "special_token_id": 128014
    },
    "RMU_unlearn_reasoning_assistant": {
        "model_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/v11_alpha3x3_lr7.5e-5_data500_reasoning_assistant/checkpoint_499",
        "tokenizer_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/v11_alpha3x3_lr7.5e-5_data500_reasoning_assistant/checkpoint_499",
        "special_token_id": 128014
    },
    "RMU_unlearn_reasoning_loss": {
        "model_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/v11_alpha3x3_lr7.5e-5_data500_reasoning_loss/checkpoint_499",
        "tokenizer_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/v11_alpha3x3_lr7.5e-5_data500_reasoning_loss/checkpoint_499",
        "special_token_id": 128014
    },
    "RMU_unlearn_best": {
        "model_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/v11_alpha1.4x1.4_lr7.5e-5_data500_reasoning_assistant_assist_loss_1/checkpoint_499",
        "tokenizer_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/v11_alpha1.4x1.4_lr7.5e-5_data500_reasoning_assistant_assist_loss_1/checkpoint_499",
        "special_token_id": 128014
    },
    "RMU_unlearn_qwen14b": {
        "model_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/qwen14b_r1_alpha50x50_steer460x460_lr5.000000e05/checkpoint_499",
        "tokenizer_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/qwen14b_r1_alpha50x50_steer460x460_lr5.000000e05/checkpoint_499",
        "special_token_id": 128014
    },
    "RMU_unlearn_qwen14b_unthink_preserve": {
        "model_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/qwen14b_r1_unthink_alpha150x150_steer460x460_lr5.000000e05/checkpoint_499",
        "tokenizer_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/qwen14b_r1_unthink_alpha150x150_steer460x460_lr5.000000e05/checkpoint_499",
        "special_token_id": 128014
    },
    "RMU_unlearn_test_11_2_2025": {
        "model_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/v11_alpha1.4x1.4_lr7.5e-5_data500_reasoning_assistant_assist_loss_1/checkpoint_499",
        "tokenizer_name": "/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/models/v11_alpha1.4x1.4_lr7.5e-5_data500_reasoning_assistant_assist_loss_1/checkpoint_499",
        "special_token_id": 128014
    },
}


TARGET_WORDS = [
    "Wait",
    "Albeit", "However", "But", "Yet", "Still",
    "Nevertheless", "Though", "Meanwhile", "Whereas"
]

def load_model_and_tokenizer(model_choice: str, device: str = "cuda"):
    """
    Loads a causal language model and tokenizer from HuggingFace transformers.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_data[model_choice]["tokenizer_name"], cache_dir="../cache")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_data[model_choice]["model_name"], cache_dir="../cache", torch_dtype=torch.float16,).to(device)

    model.eval()

    # Ensure the tokenizer will handle <pad> if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


# 在词表里找到 "Wait" 的 token id（假设该 token 存在）
def get_wait_token_id(tokenizer):
    wait_ids = tokenizer("Wait", add_special_tokens=False).input_ids
    if len(wait_ids) == 1:
        return wait_ids[0]
    else:
        raise ValueError("Cannot identify a unique 'Wait' token in the vocabulary.")

import string

def filter_vocabulary(tokenizer, token_embedding_table):
    """
    返回:
        filtered_indices: 一个 1-D 张量，包含筛选后在 vocab 中的索引
        filtered_embs: shape = (filtered_size, hidden_dim)，对应筛选后 token 的嵌入
    """
    allowed_chars = set(string.ascii_letters + string.digits + string.punctuation + ' \t\n\r')
    vocab_size = token_embedding_table.size(0)

    valid_list = []
    for i in range(vocab_size):
        token_str = tokenizer.decode([i])  # 解码单个 token id -> 文本
        # 如果 token_str 为空，或者含有不在 allowed_chars 集合内的字符，就跳过
        if not token_str:
            continue

        # 检查每个字符
        is_valid = True
        for ch in token_str:
            if ch not in allowed_chars:
                is_valid = False
                break

        if is_valid:
            valid_list.append(i)

    # 构建返回值
    filtered_indices = torch.tensor(valid_list, dtype=torch.long, device=token_embedding_table.device)
    filtered_embs = token_embedding_table[filtered_indices]  # shape (filtered_size, hidden_dim)
    return filtered_indices, filtered_embs

# =========== 句子切分示例函数 ==============
def split_into_sentences(text):
    """
    根据 '.' 和 '?' 等简单切分句子。
    注意这只是简易示例，实际需要更健壮的分句。
    """
    import re
    # 以句号或问号切分，并保留分隔符
    # 例如 "你好.我是谁?哈哈." -> ["你好.", "我是谁?", "哈哈."]
    sentences = re.split(r'([.?])', text)
    # 把分隔符合并回前一个文本
    merged = []
    for i in range(0, len(sentences) - 1, 2):
        merged.append(sentences[i].strip() + sentences[i + 1])
    # 如果总数是奇数，说明最后一个不带标点，手动添加
    if len(sentences) % 2 == 1:
        if sentences[-1].strip():
            merged.append(sentences[-1].strip())
    return [s.strip() for s in merged if s.strip()]


def test_sp_sampling(sp, model, tokenizer, question_text, special_token_id, device, wait_id, num_samples=10, use_prefix=True):
    # question_ids = tokenizer.encode(question_text, add_special_tokens=False)

    if sp is not None:
        with torch.no_grad():
            logits = sp.proj(sp.phi)
            discrete_tokens = logits.argmax(dim=-1).cpu().tolist()
            # 把 discrete tokens 变成 text
            discrete_tokens_text = tokenizer.decode(discrete_tokens)
        if use_prefix:
            full_prompt = discrete_tokens_text + question_text + "\n<think>"
        else:
            full_prompt = question_text + discrete_tokens_text + "\n<think>"

    else:
        discrete_tokens = None
        full_prompt = question_text

    answers = []
    think_answers = []
    lengths = []
    wait_counts = []

    print("\n=== Testing soft prompt sampling ===")
    for _ in tqdm(range(num_samples), desc="Sampling answers to evaluate the current prompt:"):
        torch.manual_seed(random.randint(0, 1000000))

        # input_ids_tensor = torch.tensor([full_prompt_ids], dtype=torch.long, device=device)
        inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
        output = model.generate(
            **inputs,
            do_sample=True,
            temperature=0.65,
            top_p=0.95,
            max_new_tokens=10000,
        )
        gen_ids = output[0]
        generated_text = tokenizer.decode(gen_ids, skip_special_tokens=False)
        answers.append(generated_text)

        # 找到 </think> 的位置
        gen_ids_list = gen_ids.detach().cpu().tolist()
        if special_token_id in gen_ids:
            idx = gen_ids_list.index(special_token_id)
            length_before_think = idx
        else:
            length_before_think = len(gen_ids_list)

        # think_answers.append(gen_ids[:length_before_think + 1])
        generated_think_text = tokenizer.decode(gen_ids[:length_before_think + 1], skip_special_tokens=False)
        lengths.append(length_before_think)
        think_answers.append(generated_think_text)

        # 统计在</think> token 之前 “Wait” token 的数量
        if wait_id in gen_ids_list:
            wait_count = gen_ids_list.count(wait_id)
        else:
            wait_count = 0
        wait_counts.append(wait_count)

    average_output_token_length = sum(lengths) / len(lengths)
    average_wait_count = sum(wait_counts) / len(wait_counts)

    print(f"Average length before </think>: {average_output_token_length:.2f}")
    print(f"Wait token count (avg): {average_wait_count:.2f}")

    results = {
        "answers": answers,
        "think_answers": think_answers,
        "lengths": lengths,
        "wait_counts": wait_counts,
        "average_output_token_length": average_output_token_length,
        "average_wait_count": average_wait_count,
    }
    if discrete_tokens is not None:
        results["discrete_tokens"] = discrete_tokens

    return results

def load_answers(data_dir="./datasets/qwen7b/question_1"):
    """
    假设第一阶段已经采集好了回答，
    每个回答保存为 answer_{i}.json,
    其中含有 "answer" 字段或 "tokens" 字段。
    """
    answer_files = [os.path.join(data_dir, f)
                    for f in os.listdir(data_dir)
                    if f.endswith(".json")]
    answers = []
    for file_path in answer_files:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            answers.append(data["answer"])  # or data["tokens"]
    return answers

def build_prefix_ids(question_text, context_text, tokenizer):
    # prefix = [original answer] + [context]
    merged_prompt = question_text + "\n" + context_text
    token_ids = tokenizer.encode(merged_prompt, add_special_tokens=False)
    return token_ids


class SoftPrompt(nn.Module):
    def __init__(self, vocab_size, embedding_dim, prompt_length=5, temperature=1.0):
        super().__init__()
        self.prompt_length = prompt_length
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.temperature = temperature

        # 不要直接 dtype=torch.float16，这里用fp32参数
        self.phi = nn.Parameter(torch.zeros(prompt_length, embedding_dim, dtype=torch.float32))
        self.proj = nn.Linear(embedding_dim, vocab_size, bias=False, dtype=torch.float32)

        nn.init.normal_(self.phi, mean=0.0, std=0.02)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)

    def forward(self, token_embedding, gumbel_tau=1.0, straight_through=False):
        # 让它在 fp32 下完成
        logits = self.proj(self.phi)  # (L, vocab_size), float32

        gumbels = -torch.empty_like(logits).exponential_().log()  # 也在 float32
        gumbel_logits = (logits + gumbels) / gumbel_tau
        y_soft = F.softmax(gumbel_logits, dim=-1)

        if straight_through:
            index = y_soft.argmax(dim=-1)
            y_hard = F.one_hot(index, self.vocab_size).float()
            y = y_hard.detach() - y_soft.detach() + y_soft
        else:
            y = y_soft

        # 这里的 token_embedding 可能是 fp16，也可能是 fp32
        # 如果是 fp16，需要先把 y 转成同样的 dtype
        y = y.to(token_embedding.dtype)

        prompt_emb = torch.mm(y, token_embedding)  # (L, d), 跟 token_embedding 保持同 dtype
        return prompt_emb


# ------------------------------------------------------------------------------------
# Evaluation Utility functions
# ------------------------------------------------------------------------------------

# --- 新增处理：数字位数写法的逗号移除 ---
# 定义一个函数，将 candidate 中类似 "22,222" 或 "2,324,151.23"（逗号后无空格）的数字内部的逗号移除，
# 但不会移除类似 "200, 300" 这种逗号后有空格的情况。
def remove_thousands_commas(text: str) -> str:
    pattern = r'\b\d{1,3}(,\d{3})+\b'
    return re.sub(pattern, lambda m: m.group(0).replace(",", ""), text)


def normalize_expr(expr: str):
    """
    Normalize expressions by:
    - Removing common LaTeX wrappers like \boxed{}, \text{}, $$, $, \( \), and \[ \].
    - Removing \left and \right commands.
    - Removing redundant curly braces.
    - Removing Markdown bold markers (**...** and __...__).
    - Standardizing assignment forms (e.g. x=5 becomes 5).
    - Converting Unicode square root symbol (√) to LaTeX form (\sqrt).
    - Converting angle symbols like "^\circ" to "degrees".
    - Converting \dfrac to \frac.
    - Converting recognized currency strings to a canonical numeric string.
    - Converting recognized date strings to standard format (YYYY-MM-DD).
    - Removing all whitespace.
    """
    import re
    from datetime import datetime

    # Remove thousands commas
    expr = remove_thousands_commas(expr)

    # Convert \dfrac to \frac for consistency
    expr = expr.replace("\\dfrac", "\\frac")
    expr = expr.replace("\\ ", "")
    expr = expr.replace("\\,", "")
    expr = expr.replace("\n ", "")
    expr = expr.replace("\\%", "%")
    expr = expr.replace("\\times", "*")
    expr = expr.replace("\\div", "/")
    expr = expr.replace("\\cdot", "*")
    expr = expr.replace("\\r ", "")
    expr = expr.replace("**", "")
    expr = expr.replace(",\\!", "")

    # Remove markdown bold markers (e.g. **...** and __...__)
    expr = re.sub(r"\*\*(.*?)\*\*", r"\1", expr)
    expr = re.sub(r"__(.*?)__", r"\1", expr)

    # Remove \text{...} wrapper
    expr = re.sub(r"\\text\{(.*?)\}", r"\1", expr)

    # Remove math mode markers: $$ or $
    expr = expr.replace("$$", "").replace("$", "")

    # Remove inline math wrappers: \( ... \) or \\( ... \\)
    expr = re.sub(r"\\\((.*?)\\\)", r"\1", expr)

    # Remove display math wrappers: \[ ... \]
    expr = re.sub(r"\\\[(.*?)\\\]", r"\1", expr)

    # Remove \left and \right commands
    expr = expr.replace("\\left", "").replace("\\right", "")

    # Remove redundant outer curly braces, e.g. { ... }
    expr = re.sub(r"^\{(.*)\}$", r"\1", expr)

    # Remove all remaining curly braces that wrap subexpressions
    # expr = re.sub(r"\{([^\{\}]+)\}", r"\1", expr)

    # Standardize assignment: if expression contains "=", take only the right-hand side.
    if "=" in expr:
        expr = expr.split("=", 1)[1]

    # Convert Unicode square root symbol to LaTeX \sqrt form
    expr = expr.replace("√", r"\sqrt")

    # Convert angle unit: replace "^\circ" with "degrees"
    expr = expr.replace("^\\circ", "degrees")

    # Remove all whitespace (先不去掉空白，方便后续对日期或货币的判断)
    expr = re.sub(r"\s+", "", expr).strip()

    # --------- 货币转换处理 ---------
    # 如果表达式中包含货币符号或货币代码，则提取数字部分并转换为 float 的字符串
    if re.search(r"[\$\¥]|(?:USD|CNY|RMB|EUR)", expr, re.IGNORECASE):
        # 去除除数字、小数点和负号之外的所有字符
        value = re.sub(r"[^0-9\.-]", "", expr)
        try:
            expr = str(float(value))
        except Exception:
            pass

    # --------- 日期格式统一处理 ---------
    # 定义若干常见的日期格式
    date_formats = [
        "%m/%d/%Y", "%m-%d-%Y", "%m/%d/%y", "%m-%d-%y",  # 数字形式，如 1/2/2020 或 01-02-2020
        "%Y/%m/%d", "%Y-%m-%d",  # 年月日顺序，如 2020/1/2 或 2020-01-02
        "%B%d,%Y", "%b%d,%Y", "%B%d,%y", "%b%d,%y",  # 如 January1,2020 或 Jan1,2020（不带空格）
        "%B%d%Y", "%b%d%Y",  # 如 January12020（不常见）
        "%B%d,%Y", "%b%d,%Y",  # 如果有空格可调整
        "%B %d, %Y", "%b %d, %Y",  # 如 January 1, 2020 或 Jan 1, 2020
        "%d%B%Y", "%d%b%Y",  # 如 1January2020 或 1Jan2020
        "%d %B %Y", "%d %b %Y"  # 如 1 January 2020 或 1 Jan 2020
    ]
    # 尝试用以上格式解析整个表达式，如果成功则统一为 YYYY-MM-DD
    for fmt in date_formats:
        try:
            dt = datetime.strptime(expr, fmt)
            expr = dt.strftime("%Y-%m-%d")
            break
        except ValueError:
            continue

    # 最后再次移除所有空白字符（以防前面日期转换后带有空格）
    expr = re.sub(r"\s+", "", expr).strip()

    return expr


def classify_answer(gold_ans: str):
    """
    Classify the answer into a type among (date, fraction, decimal, expression, text, etc.)
    Also create a canonical representation if possible.

    新增处理：
    (1) 如果答案是日期，则类型为 "date"。支持多种日期格式，如数字格式（1/2/2020、2020-1-2等）和包含月份名称的格式（January 1, 2020、Jan 1, 2020）。
    (2) 如果答案含有货币符号（例如 $, ¥, USD, CNY, RMB 等），则去掉货币符号后转换为数字，类型归为 "decimal"。
    (3) 纯数字（整数或小数）统一归为 "decimal" 类型。
    (4) 如果答案写作 \fracxy（其中 x,y 为数字），先转换为标准形式 \frac{x}{y}。
    (5) 如果答案完全不含数字，则当作纯文本，类型为 "text"；其他情况归为 "expression".

    Returns a tuple: (type, canonical_value)
    """
    # 去除前后空白
    gold_ans_stripped = gold_ans.strip()

    gold_ans_stripped = remove_thousands_commas(gold_ans_stripped)

    # 将形如 \frac43 的写法转换为标准形式 \frac{4}{3}
    gold_ans_stripped = re.sub(r"(\\frac)(?!\s*\{)(\d+)(?!\s*\{)(\d+)", r"\1{\2}{\3}", gold_ans_stripped)


    # ----------------- 1. 日期判断 -----------------
    # 日期常见格式：
    #   a. 数字形式：1/2/2020, 01-02-2020, 2020-1-2, 2020/01/02 等
    #   b. 包含月份名称：January 1, 2020, Jan 1 2020, 1 Jan 2020, etc.
    date_numeric_pattern1 = r"^\d{1,2}[/-]\d{1,2}[/-]\d{2,4}$"
    date_numeric_pattern2 = r"^\d{4}[/-]\d{1,2}[/-]\d{1,2}$"
    # 包含月份名称（全写或简写），忽略大小写
    date_text_pattern = r"(?i)^(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[ ,.-]*)+\d{1,2}[, \.-]+\d{2,4}$"
    if (re.match(date_numeric_pattern1, gold_ans_stripped) or
            re.match(date_numeric_pattern2, gold_ans_stripped) or
            re.match(date_text_pattern, gold_ans_stripped)):
        # 这里直接返回原字符串，或可进一步转换为统一的格式，如 YYYY-MM-DD（此处暂直接返回归一化后的字符串）
        return ("date", gold_ans_stripped)

    # ----------------- 2. 货币判断 -----------------
    # 如果包含美元符号、人民币符号或常见货币代码（USD, CNY, RMB, EUR等），则视为货币
    if re.search(r"[\¥\€\￡]|(?:USD|CNY|RMB|EUR|JPY|GBP)", gold_ans_stripped, re.IGNORECASE):
        # 去除所有非数字、非小数点、非负号的字符
        cleaned = re.sub(r"[^0-9\.-]", "", gold_ans_stripped)
        try:
            value = float(cleaned)
            return ("decimal", value)
        except:
            pass  # 如果转换失败，则继续下面的判断

    # ----------------- 3. 分数判断 -----------------
    frac_pattern = r"frac\s*\{(-?\d+)\}\s*\{(-?\d+)\}"
    match_frac = re.search(frac_pattern, gold_ans_stripped)
    if match_frac:
        numerator = int(match_frac.group(1))
        denominator = int(match_frac.group(2))
        return ("fraction", Fraction(numerator, denominator))

    # 4. 检查是否为整数（可能带负号）
    int_pattern = r"^-?\d+$"
    if re.match(int_pattern, gold_ans_stripped):
        return ("integer", int(gold_ans_stripped))

    # 5. 检查是否为小数
    dec_pattern = r"^-?\d+\.\d+$"
    if re.match(dec_pattern, gold_ans_stripped):
        return ("decimal", float(gold_ans_stripped))

    # ----------------- 6. 如果答案完全不含数字，则当作纯文本 -----------------
    if re.search(r"[0-9]", gold_ans_stripped) is None:
        return ("text", gold_ans_stripped)

    # ----------------- 7. 其他情况归为 expression -----------------
    return ("expression", gold_ans_stripped)


def find_all_occurrences(text, pattern):
    return [match.start() for match in re.finditer(re.escape(pattern), text)]


def extract_model_answer(response: str, gold_type: str) -> str:
    """
    Extract the model's final answer from the text response, guided by the known gold_type.

    1) If gold_type is 'fraction', first try to find a LaTeX fraction.
    2) If gold_type is 'integer', first try an integer parse.
    3) If gold_type is 'decimal', try decimal parse.
    4) etc.

    If we fail in the specialized approach, we fall back to a more generic approach
    that tries standard triggers or the last mention of a numeric/fraction pattern, etc.

    Returns a string that (hopefully) represents the model’s best guess.
    """

    # --- 1) Specialized approach by gold_type ---
    tail = response[-300:]  # We just look near the end to reduce noise

    # 2a) Look for typical concluding phrases:
    lower_resp = tail.lower()
    triggers = ["the answer is", "final answer", "the result is", "hence the answer is", "hence the result is",
                "therefore", "conclusion", "final choice", "final solution", "final decision", "final result", "thus",
                "</think>"]
    found_index = -1
    chosen_trigger = None

    # Find the last trigger in the response
    for t in triggers:
        # 从后往前找，找到最后一个触发词
        # idx = lower_resp.rfind(t)
        positions = find_all_occurrences(lower_resp, t)
        if len(positions) > 0:
            new_found_index = positions[-1]
            if new_found_index > found_index:
                found_index = new_found_index
                chosen_trigger = t

    if found_index != -1:
        snippet = tail[found_index + len(chosen_trigger):]
        # remove all the "/n" in the snippet
        snippet = snippet.replace("\n", "")
        snippet = snippet.replace("\r", "")
        # snippet_split = re.split(r'[.\n]', snippet, maxsplit=1)
        candidate = snippet.strip()
    else:
        candidate = tail

    # If there is a structure like \boxed{}, then we should extract the content inside the box and ignore the rest
    idx_box = candidate.find(r'\boxed{')
    if idx_box != -1:
        # 找到 \boxed{ 后，确定内容开始的位置
        start = candidate.find('{', idx_box)
        if start != -1:
            brace_count = 0
            end = start
            # 从 start 位置开始遍历，使用计数法匹配花括号
            for i in range(start, len(candidate)):
                if candidate[i] == '{':
                    brace_count += 1
                elif candidate[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i
                        break
            # 如果成功匹配，则提取 \boxed{...} 内的内容
            if brace_count == 0:
                candidate = candidate[start + 1:end].strip()
    candidate = normalize_expr(candidate)
    if gold_type == "fraction":
        # Look specifically for \frac ... patterns near the end
        # fraction_like = re.search(r"\\frac\s*\{?-?\d+\}?\s*\{?-?\d+\}?", candidate)
        # look for that last string "frac" in the candidate and return anything from that point
        # fraction_like = re.search("frac", candidate)
        positions = find_all_occurrences(normalize_expr(candidate), "frac")
        if len(positions) > 0:
            return candidate[positions[-1]:]


    # elif gold_type == "integer":
    #     # Try to find the last integer in the tail
    #     # This pattern matches strictly integer (no decimal)
    #     int_pattern = re.compile(r"-?\b\d+\b")
    #     all_ints = list(re.finditer(int_pattern, normalize_expr(candidate)))
    #     if all_ints:
    #         # pick the last one
    #         return all_ints[-1].group(0).strip()

    # elif gold_type == "decimal" or gold_type == "integer":
    #     # Look for decimal patterns, e.g. 3.21, -0.15
    #     dec_pattern = re.compile(r"-?\d+\.\d+")
    #     all_decs = list(re.finditer(dec_pattern, normalize_expr(candidate)))
    #     if all_decs:
    #         return all_decs[-1].group(0).strip()

    elif gold_type in ["integer", "decimal"]:
        # 匹配整数或小数：小数部分 (?:\.\d+)? 可有可无
        number_pattern = re.compile(r"-?\d+(?:\.\d+)?")
        all_numbers = list(re.finditer(number_pattern, normalize_expr(candidate)))
        if all_numbers:
            # 选择最后一个匹配的数字
            return all_numbers[-1].group(0).strip()

    # For text or expression, we can skip specialized checks because the model might output anything.
    # We will rely on the fallback approach below.

    # --- 2) Fallback approach (more generic) ---

    # 2b) Look for the last inline LaTeX expression like $...$ or $$...$$
    latex_pattern = re.compile(r'(\${1,2})(.+?)\1', re.DOTALL)
    matches = latex_pattern.findall(candidate)
    if matches:
        # Pick the last one
        last = matches[-1]
        return last[1].strip()

    # # 2c) fraction-like or numeric pattern near the end
    # fraction_like = re.search(r"\\frac\s*\{?-?\d+\}?\s*\{?-?\d+\}?", candidate)
    # if fraction_like:
    #     return fraction_like.group(0).strip()

    # num_pattern = re.compile(r"-?\d+(\.\d+)?")
    # num_iter = list(re.finditer(num_pattern, candidate))
    # if num_iter:
    #     # pick the last one
    #     return num_iter[-1].group(0).strip()

    # 2d) Otherwise, fallback: last sentence
    sentences = [s for s in re.split(r'[.\n]', candidate.strip()) if s != ""]
    if sentences:
        return sentences[-1].strip()

    # final fallback
    return candidate


def classify_model_answer(model_ans_str: str):
    """
    Classify the extracted answer from the model using
    the same logic as classify_answer.
    """
    return classify_answer(model_ans_str)


def extract_number_substrings(s: str):
    """
    从字符串 s 中简单地提取所有连续的数字和小数点组成的子串，
    遍历字符串，当遇到 digit 或者 '.' 就开始记录，直到遇到其他字符停止，
    返回所有提取到的子串列表。
    """
    result = []
    i = 0
    n = len(s)
    while i < n:
        if s[i].isdigit():
            start = i
            while i < n and (s[i].isdigit() or s[i] == '.'):
                i += 1
            result.append(s[start:i])
        else:
            i += 1
    return result


def compare_answers(gold_type, gold_value, pred_type, pred_value, decimal_tolerance=1e-7):
    """
    Compare the gold answer and predicted answer.
    1. 先对两者进行归一化处理。
    2. 如果归一化结果完全一致，或者其中一个归一化结果包含另一个，则视为匹配正确。
    3. 否则，根据类型和数值误差进行比较。
    """

    # 0. 快速判断：提取归一化结果中的所有数字（整数或小数），如果完全一致，则认为匹配
    nums_gold = extract_number_substrings(str(gold_value))
    nums_pred = extract_number_substrings(str(pred_value))
    if nums_gold == nums_pred:
        return True

    norm_gold = normalize_expr(str(gold_value))
    norm_pred = normalize_expr(str(pred_value))

    # 1. 如果完全一致，返回 True
    if norm_gold == norm_pred:
        return True

    # 2. 如果一个是另一个的子串，也认为匹配正确
    # 只有字符串才符合这个规则
    if gold_type not in ("integer", "decimal") and pred_type not in ("integer", "decimal"):
        # 是字符串类型，且一个是另一个的子串
        if norm_gold in norm_pred or norm_pred in norm_gold:
            return True

    # 4. 类型相同时，根据数值进行比较
    if gold_type == pred_type:
        if gold_type == "integer":
            return (gold_value == pred_value)
        elif gold_type == "decimal":
            return abs(gold_value - pred_value) < decimal_tolerance
        elif gold_type == "fraction":
            return gold_value == pred_value
        elif gold_type in ("text", "expression"):
            return norm_gold == norm_pred
        else:
            return norm_gold == norm_pred

    # 5. 处理不同类型的数值比较
    if gold_type == "fraction" and pred_type == "decimal":
        return abs(float(gold_value) - pred_value) < decimal_tolerance
    if gold_type == "decimal" and pred_type == "fraction":
        return abs(gold_value - float(pred_value)) < decimal_tolerance
    if gold_type == "integer" and pred_type == "fraction":
        return gold_value == pred_value.numerator and pred_value.denominator == 1
    if gold_type == "fraction" and pred_type == "integer":
        return pred_value == gold_value.numerator and gold_value.denominator == 1
    if gold_type == "integer" and pred_type == "decimal":
        return abs(gold_value - pred_value) < decimal_tolerance
    if gold_type == "decimal" and pred_type == "integer":
        return abs(gold_value - pred_value) < decimal_tolerance

    return False


def check_answer_overall(model_response: str, gold_ans: str):
    """
    High-level function:
      1) Classify the gold answer => (gold_type, gold_value).
      2) Extract the model answer guided by gold_type.
      3) Classify the extracted string => (pred_type, pred_value).
      4) Compare them.
    Returns: (bool_correct, extracted_answer)
    """
    # 1) Classify gold
    gold_type, gold_value = classify_answer(gold_ans)

    # 2) Extract predicted answer from the model
    predicted_str = extract_model_answer(model_response, gold_type)

    # 3) Classify the predicted string
    pred_type, pred_value = classify_answer(predicted_str)

    # 4) Compare
    is_correct = compare_answers(gold_type, gold_value, pred_type, pred_value)

    return is_correct, predicted_str


from transformers import LogitsProcessor

class DecayingWaitLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, max_decay_steps=100, min_prob_scale=0.1):
        self.tokenizer = tokenizer
        self.wait_token_id = tokenizer.convert_tokens_to_ids("wait")
        self.max_decay_steps = max_decay_steps
        self.min_prob_scale = min_prob_scale

    def __call__(self, input_ids, scores):
        step = input_ids.shape[1]  # 当前生成的 token 数
        # 计算衰减系数：从 1.0 衰减到 min_prob_scale
        decay_factor = max(self.min_prob_scale, 1.0 - step / self.max_decay_steps)

        # print(f"Before logitsprocessor: {scores[:, self.wait_token_id]}")
        # 对 wait 这个词的 logits 乘上 decay_factor
        scores[:, self.wait_token_id] *= decay_factor
        # print(f"After logitsprocessor: {scores[:, self.wait_token_id]}")
        return scores


class WaitCountPenaltyProcessor(LogitsProcessor):
    def __init__(self, tokenizer, gamma=0.8):
        """
        Args:
            tokenizer: 用于获取 'wait' 的 token ID。
            gamma: 每生成一次 wait，其 logits 惩罚乘以 gamma (0 < gamma <= 1)。
        """
        self.wait_token_id = tokenizer.convert_tokens_to_ids("wait")
        self.gamma = gamma

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # input_ids: [batch_size, cur_len]
        # scores: [batch_size, vocab_size]
        batch_size = input_ids.shape[0]

        for i in range(batch_size):
            wait_count = (input_ids[i] == self.wait_token_id).sum().item()
            penalty = self.gamma ** wait_count
            # print(f"Before logitsprocessor: {scores[:, self.wait_token_id]}")
            scores[i, self.wait_token_id] *= penalty
            # print(f"After logitsprocessor: {scores[:, self.wait_token_id]}")
        return scores


class DecayProcessor(LogitsProcessor):
    def __init__(self, tokenizer, max_new_tokens=4000, alpha=0.8):
        self.wait_token_ids = [tokenizer.convert_tokens_to_ids(s) for s in ["wait", "Wait"]]
        self.max_new_tokens = max_new_tokens
        self.alpha = alpha

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        step = input_ids.shape[1]  # 当前生成步数
        # 计算衰减系数（alpha 控制衰减速度）
        decay_factor = max(0.0, 1.0 - self.alpha * step / self.max_new_tokens)

        for wait_token_id in self.wait_token_ids:
            scores[:, wait_token_id] *= decay_factor

        return scores


class PenaltyProcessor(LogitsProcessor):
    def __init__(self, tokenizer, gamma=0.8):
        self.wait_token_ids = [tokenizer.convert_tokens_to_ids(s) for s in ["wait", "Wait"]]
        self.gamma = gamma

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = input_ids.shape[0]

        for i in range(batch_size):
            # 累计 wait / Wait 的出现次数
            wait_count = sum((input_ids[i] == token_id).sum().item() for token_id in self.wait_token_ids)
            penalty = self.gamma ** wait_count
            for wait_token_id in self.wait_token_ids:
                scores[i, wait_token_id] *= penalty

        return scores