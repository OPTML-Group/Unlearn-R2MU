import json

# 文件路径
file_path = "/egr/research-optml/wangc168/reasoning/reason_unlearn/QA_generation/log/merged_with_index_and_non_reason_logits.jsonl"

# 初始化计数器
count_dict = {
    1: {"non_reason_false": 0, "logits_false": 0},
    2: {"non_reason_false": 0, "logits_false": 0},
    3: {"non_reason_false": 0, "logits_false": 0},
    4: {"non_reason_false": 0, "logits_false": 0}
}

# 读取并统计
with open(file_path, 'r') as f:
    for line in f:
        item = json.loads(line)
        gold = item.get("generation_supports_gold")
        if gold in [1, 2, 3, 4]:
            if not item.get("non_reason_logits_match", True):
                count_dict[gold]["non_reason_false"] += 1
            if not item.get("logits_match", True):
                count_dict[gold]["logits_false"] += 1

# 输出结果
for k in range(1, 5):
    print(f"generation_supports_gold == {k}: non_reason_false = {count_dict[k]['non_reason_false']}, logits_false = {count_dict[k]['logits_false']}")