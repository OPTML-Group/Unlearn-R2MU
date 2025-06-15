# wandb login decc29dfb71fb2bec9be656e161d820647b0aaae

export PYTHONPATH=/egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/lm-evaluation-harness:$PYTHONPATH

# ALPHA="1.5,1.5"
# LR="7.5e-5"
# DATA="500"
# FILE="v11"
# NAME="reasoning_assistant"
# assist_loss="1"

# MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# OUTPUT_NAME="${FILE}_alpha${ALPHA//,/x}_lr${LR}_data${DATA}_${NAME}_assist_loss_${assist_loss}"
# OUTPUT_DIR="models/${OUTPUT_NAME}"
# LOG_FILE="${OUTPUT_NAME}.log"

# nohup bash -c "
# CUDA_VISIBLE_DEVICES=4,5 python3 -m rmu.unlearn_${FILE} \
#   --model_name_or_path ${MODEL_NAME} \
#   --max_num_batches ${DATA} \
#   --batch_size 4 \
#   --retain_corpora wikitext \
#   --forget_corpora original \
#   --steering_coeffs 6.5,6.5 \
#   --alpha ${ALPHA} \
#   --lr ${LR} \
#   --assist_loss ${assist_loss} \
#   --seed 42 \
#   --output_dir ${OUTPUT_DIR} \
#   --generated_path /egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/generated_all.jsonl \
#   --raw_path /egr/research-optml/wangc168/watermark/IBM_Run_test/dataset/bio_remove_dataset.jsonl \
#   --max_gen_tokens 100 \
#   --verbose
# " > ${LOG_FILE} 2>&1 &



ALPHA="1.7,1.7"
LR="7.5e-5"
DATA="500"
FILE="v11"
NAME="no_unlearn"
assist_loss="1"

MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
OUTPUT_NAME="${FILE}_alpha${ALPHA//,/x}_lr${LR}_data${DATA}_${NAME}_assist_loss_${assist_loss}"
OUTPUT_DIR="models/${OUTPUT_NAME}"
LOG_FILE="${OUTPUT_NAME}.log"

nohup bash -c "
CUDA_VISIBLE_DEVICES=6,7 python3 -m rmu.unlearn_${FILE} \
  --model_name_or_path ${MODEL_NAME} \
  --max_num_batches ${DATA} \
  --batch_size 4 \
  --retain_corpora wikitext \
  --forget_corpora original \
  --steering_coeffs 6.5,6.5 \
  --alpha ${ALPHA} \
  --lr ${LR} \
  --assist_loss ${assist_loss} \
  --seed 42 \
  --output_dir ${OUTPUT_DIR} \
  --generated_path /egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/generated_all.jsonl \
  --raw_path /egr/research-optml/wangc168/watermark/IBM_Run_test/dataset/bio_remove_dataset.jsonl \
  --max_gen_tokens 100 \
  --verbose
" > ${LOG_FILE} 2>&1 &

# ALPHA="2,2"
# LR="7.5e-5"
# DATA="500"
# FILE="v11"
# NAME="reasoning_assistant"
# assist_loss="1"

# MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# OUTPUT_NAME="${FILE}_alpha${ALPHA//,/x}_lr${LR}_data${DATA}_${NAME}_assist_loss_${assist_loss}"
# OUTPUT_DIR="models/${OUTPUT_NAME}"
# LOG_FILE="${OUTPUT_NAME}.log"

# nohup bash -c "
# CUDA_VISIBLE_DEVICES=2,3 python3 -m rmu.unlearn_${FILE} \
#   --model_name_or_path ${MODEL_NAME} \
#   --max_num_batches ${DATA} \
#   --batch_size 4 \
#   --retain_corpora wikitext \
#   --forget_corpora original \
#   --steering_coeffs 6.5,6.5 \
#   --alpha ${ALPHA} \
#   --lr ${LR} \
#   --assist_loss ${assist_loss} \
#   --seed 42 \
#   --output_dir ${OUTPUT_DIR} \
#   --generated_path /egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/generated_all.jsonl \
#   --raw_path /egr/research-optml/wangc168/watermark/IBM_Run_test/dataset/bio_remove_dataset.jsonl \
#   --max_gen_tokens 100 \
#   --verbose
# " > ${LOG_FILE} 2>&1 &





# ALPHA="3,3"
# LR="7.5e-5"
# DATA="500"
# FILE="v11"
# NAME="reasoning_assistant"
# assist_loss="0.9"

# MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# OUTPUT_NAME="${FILE}_alpha${ALPHA//,/x}_lr${LR}_data${DATA}_${NAME}_assist_loss_${assist_loss}"
# OUTPUT_DIR="models/${OUTPUT_NAME}"
# LOG_FILE="${OUTPUT_NAME}.log"

# nohup bash -c "
# CUDA_VISIBLE_DEVICES=4,5 python3 -m rmu.unlearn_${FILE} \
#   --model_name_or_path ${MODEL_NAME} \
#   --max_num_batches ${DATA} \
#   --batch_size 4 \
#   --retain_corpora wikitext \
#   --forget_corpora original \
#   --steering_coeffs 6.5,6.5 \
#   --alpha ${ALPHA} \
#   --lr ${LR} \
#   --assist_loss ${assist_loss} \
#   --seed 42 \
#   --output_dir ${OUTPUT_DIR} \
#   --generated_path /egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/generated_all.jsonl \
#   --raw_path /egr/research-optml/wangc168/watermark/IBM_Run_test/dataset/bio_remove_dataset.jsonl \
#   --max_gen_tokens 100 \
#   --verbose
# " > ${LOG_FILE} 2>&1 &





# ALPHA="2,2"
# LR="7.5e-5"
# DATA="500"
# FILE="v11"
# NAME="reasoning_assistant"
# assist_loss="1.0"

# MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# OUTPUT_NAME="${FILE}_alpha${ALPHA//,/x}_lr${LR}_data${DATA}_${NAME}_assist_loss_${assist_loss}"
# OUTPUT_DIR="models/${OUTPUT_NAME}"
# LOG_FILE="${OUTPUT_NAME}.log"

# nohup bash -c "
# CUDA_VISIBLE_DEVICES=6,7 python3 -m rmu.unlearn_${FILE} \
#   --model_name_or_path ${MODEL_NAME} \
#   --max_num_batches ${DATA} \
#   --batch_size 4 \
#   --retain_corpora wikitext \
#   --forget_corpora original \
#   --steering_coeffs 6.5,6.5 \
#   --alpha ${ALPHA} \
#   --lr ${LR} \
#   --assist_loss ${assist_loss} \
#   --seed 42 \
#   --output_dir ${OUTPUT_DIR} \
#   --generated_path /egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/generated_all.jsonl \
#   --raw_path /egr/research-optml/wangc168/watermark/IBM_Run_test/dataset/bio_remove_dataset.jsonl \
#   --max_gen_tokens 100 \
#   --verbose
# " > ${LOG_FILE} 2>&1 &










# ALPHA="5,5"
# LR="7.5e-5"

# MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# OUTPUT_NAME="unthink_alpha${ALPHA//,/x}_lr${LR}_data${DATA}"
# OUTPUT_DIR="models/${OUTPUT_NAME}"
# LOG_FILE="${OUTPUT_NAME}.log"

# nohup bash -c "
# CUDA_VISIBLE_DEVICES=2,3 python3 -m rmu.unlearn_v5 \
#   --model_name_or_path ${MODEL_NAME} \
#   --max_num_batches ${DATA} \
#   --batch_size 4 \
#   --retain_corpora wikitext \
#   --forget_corpora original \
#   --steering_coeffs 6.5,6.5 \
#   --alpha ${ALPHA} \
#   --lr ${LR} \
#   --seed 42 \
#   --output_dir ${OUTPUT_DIR} \
#   --generated_path /egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/generated_all.jsonl \
#   --raw_path /egr/research-optml/wangc168/watermark/IBM_Run_test/dataset/bio_remove_dataset.jsonl \
#   --max_gen_tokens 100 \
#   --verbose
# " > ${LOG_FILE} 2>&1 &


# ALPHA="7,7"
# LR="7.5e-5"

# MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# OUTPUT_NAME="unthink_alpha${ALPHA//,/x}_lr${LR}_data${DATA}"
# OUTPUT_DIR="models/${OUTPUT_NAME}"
# LOG_FILE="${OUTPUT_NAME}.log"

# nohup bash -c "
# CUDA_VISIBLE_DEVICES=4,5 python3 -m rmu.unlearn_v5 \
#   --model_name_or_path ${MODEL_NAME} \
#   --max_num_batches ${DATA} \
#   --batch_size 4 \
#   --retain_corpora wikitext \
#   --forget_corpora original \
#   --steering_coeffs 6.5,6.5 \
#   --alpha ${ALPHA} \
#   --lr ${LR} \
#   --seed 42 \
#   --output_dir ${OUTPUT_DIR} \
#   --generated_path /egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/generated_all.jsonl \
#   --raw_path /egr/research-optml/wangc168/watermark/IBM_Run_test/dataset/bio_remove_dataset.jsonl \
#   --max_gen_tokens 100 \
#   --verbose
# " > ${LOG_FILE} 2>&1 &


# ALPHA="3,3"
# LR="5e-5"

# MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
# OUTPUT_NAME="unthink_alpha${ALPHA//,/x}_lr${LR}_data${DATA}"
# OUTPUT_DIR="models/${OUTPUT_NAME}"
# LOG_FILE="${OUTPUT_NAME}.log"

# nohup bash -c "
# CUDA_VISIBLE_DEVICES=6,7 python3 -m rmu.unlearn_v5 \
#   --model_name_or_path ${MODEL_NAME} \
#   --max_num_batches ${DATA} \
#   --batch_size 4 \
#   --retain_corpora wikitext \
#   --forget_corpora original \
#   --steering_coeffs 6.5,6.5 \
#   --alpha ${ALPHA} \
#   --lr ${LR} \
#   --seed 42 \
#   --output_dir ${OUTPUT_DIR} \
#   --generated_path /egr/research-optml/wangc168/reasoning/reason_unlearn/rmu/wmdp/generated_all.jsonl \
#   --raw_path /egr/research-optml/wangc168/watermark/IBM_Run_test/dataset/bio_remove_dataset.jsonl \
#   --max_gen_tokens 100 \
#   --verbose
# " > ${LOG_FILE} 2>&1 &