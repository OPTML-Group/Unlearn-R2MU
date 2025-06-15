import os
import json
import torch
import multiprocessing as mp
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# ===== 配置 =====
DATA_PATH = "/egr/research-optml/wangc168/watermark/IBM_Run_test/dataset/bio_remove_dataset.jsonl"
INSTRUCTION = "\nLet's reason this step by step.\n<think>"
MAX_INPUT_TOKENS = 512
MAX_GENERATE_TOKENS = 1024

# ===== 读取 + 截断拼接 =====
def read_and_tokenize_data(tokenizer, target_indices):
    data = []
    with open(DATA_PATH, "r") as f:
        all_lines = f.readlines()
        for idx in target_indices:
            if idx >= len(all_lines):
                continue
            try:
                text = json.loads(all_lines[idx]).get("text", "")
                if not isinstance(text, str):
                    continue
                # 先对 text 截断到 512 tokens
                tokenized_text = tokenizer(text, truncation=True, max_length=MAX_INPUT_TOKENS)
                truncated_text = tokenizer.decode(tokenized_text["input_ids"], skip_special_tokens=True)
                # 拼接 instruction，不再参与截断
                final_prompt = truncated_text + INSTRUCTION
                data.append({"idx": idx, "input": final_prompt})
            except Exception as e:
                print(f"Skipping idx {idx} due to error: {e}")
    return data

# ===== 模型加载到 GPU =====
def load_model_on_device(model_path, device_id):
    torch.cuda.set_device(device_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map={"": device_id},
        torch_dtype=torch.bfloat16
    )
    model.eval()
    return tokenizer, model

# ===== 每张卡生成并写入 =====
def generate_on_gpu(data_chunk, device_id, model_path, output_path):
    tokenizer, model = load_model_on_device(model_path, device_id)
    results = []
    for sample in tqdm(data_chunk, desc=f"GPU {device_id}"):
        inputs = tokenizer(sample["input"], return_tensors="pt").to(f"cuda:{device_id}")
        input_len = inputs["input_ids"].shape[1]

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_length=input_len + MAX_GENERATE_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
        gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 截去 prompt（即 sample["input"]）部分，只保留新增内容
        if gen_text.startswith(sample["input"]):
            gen_text = gen_text[len(sample["input"]):].strip()

        results.append({"idx": sample["idx"], "512": gen_text})
    with open(output_path, "w") as fout:
        for line in results:
            fout.write(json.dumps(line, ensure_ascii=False) + "\n")

# ===== 多卡调度主逻辑 =====
def parallel_generate(model_path, num_gpus, target_indices):
    print("Loading and tokenizing input...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    data = read_and_tokenize_data(tokenizer, target_indices)

    chunk_size = (len(data) + num_gpus - 1) // num_gpus
    processes = []

    for i in range(num_gpus):
        data_chunk = data[i * chunk_size: (i + 1) * chunk_size]
        output_path = f"generated_gpu{i}.jsonl"
        p = mp.Process(target=generate_on_gpu, args=(data_chunk, i, model_path, output_path))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("Merging and sorting outputs...")
    merged = []
    for i in range(num_gpus):
        gpu_file = f"generated_gpu{i}.jsonl"
        with open(gpu_file, "r") as f:
            merged.extend([json.loads(line) for line in f])
        os.remove(gpu_file)  # 清理临时文件

    merged.sort(key=lambda x: x["idx"])  # 保证顺序一致
    with open("generated_all.jsonl", "w") as fout:
        for item in merged:
            fout.write(json.dumps({"512": item["512"]}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    num_gpus = torch.cuda.device_count()
    target_indices = list(range(2000))
    parallel_generate(model_path, num_gpus, target_indices)