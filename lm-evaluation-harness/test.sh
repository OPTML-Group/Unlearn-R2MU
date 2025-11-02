export PYTHONPATH=/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/lm-evaluation-harness:$PYTHONPATH


CUDA_VISIBLE_DEVICES=1,2 lm_eval --model hf \
    --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --tasks wmdp_bio \
    --batch_size 16 \
    | tee logs/deepseek14b_wmdp_bio.log &

CUDA_VISIBLE_DEVICES=3,4 lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen2.5-14B \
    --tasks wmdp_bio \
    --batch_size 16 \
    | tee logs/qwen25_14b_wmdp_bio.log &

CUDA_VISIBLE_DEVICES=5,6 lm_eval --model hf \
    --model_args pretrained=deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --tasks mmlu \
    --batch_size 16 \
    | tee logs/deepseek14b_mmlu.log &

CUDA_VISIBLE_DEVICES=7,0 lm_eval --model hf \
    --model_args pretrained=Qwen/Qwen2.5-14B \
    --tasks mmlu \
    --batch_size 16 \
    | tee logs/qwen25_14b_mmlu.log &