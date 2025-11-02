# export PYTHONPATH=lm-evaluation-harness:$PYTHONPATH
export PYTHONPATH=/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/lm-evaluation-harness:$PYTHONPATH

ALPHA="1.4,1.4"
LR="7.5e-5"
DATA_NUM="500"
NAME="reasoning_assistant"
assist_loss="1"

MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
OUTPUT_NAME="alpha${ALPHA//,/x}_lr${LR}_wmdp_${DATA_NUM}_${NAME}_assist_loss_${assist_loss}"
OUTPUT_DIR="models/${OUTPUT_NAME}"
LOG_FILE="${OUTPUT_NAME}.log"

# conda enviroments use SOUL Jinghan
# nohup bash -c "
CUDA_VISIBLE_DEVICES=0,1 python3 -m unlearn_wmdp \
  --model_name_or_path ${MODEL_NAME} \
  --max_num_batches ${DATA_NUM} \
  --batch_size 4 \
  --retain_corpora wikitext \
  --forget_corpora original \
  --steering_coeffs 6.5,6.5 \
  --alpha ${ALPHA} \
  --lr ${LR} \
  --assist_loss ${assist_loss} \
  --seed 42 \
  --output_dir ${OUTPUT_DIR} \
  --generated_path ./generated_all_wmdp.jsonl \
  --raw_path ./bio_remove_dataset.jsonl \
  --max_gen_tokens 100 \
  --verbose
# " > ${LOG_FILE} 2>&1 &