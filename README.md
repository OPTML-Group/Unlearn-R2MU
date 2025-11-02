<div align='center'>

# Reasoning Model Unlearning: Forgetting Traces, Not Just Answers, While Preserving Reasoning Skills

<p align="center">

  <a href="#"><img src="https://img.shields.io/badge/Venue-EMNLP%202025-green"></a>
  <a href="https://arxiv.org/abs/2506.12963"><img src="https://img.shields.io/badge/arXiv-2506.12963-B31B1B"></a>
  <a href="#"><img src="https://img.shields.io/badge/HuggingFace-Collection-yellow"></a>
  <br>
  <a href="https://github.com/OPTML-Group/Unlearn-R2MU"><img src="https://img.shields.io/badge/License-MIT-blue"></a>
  <a href="https://github.com/OPTML-Group/Unlearn-R2MU"><img src="https://img.shields.io/github/languages/top/OPTML-Group/Unlearn-R2MU"></a>
  <a href="https://github.com/OPTML-Group/Unlearn-R2MU"><img src="https://img.shields.io/github/repo-size/OPTML-Group/Unlearn-R2MU"></a>
  <a href="https://github.com/OPTML-Group/Unlearn-R2MU"><img src="https://img.shields.io/github/stars/OPTML-Group/Unlearn-R2MU?style=social"></a>

</p>
</div>

## How to run the code?

### Install the conda enviroment

You can install the required dependencies as the instruction in [SOUL](https://github.com/OPTML-Group/SOUL):

### Run the Unlearn part

```
bash run.sh
```

In `run.sh`, command is like:

```
# Put your own lm-evaluation-harness path here
export PYTHONPATH=lm-evaluation-harness:$PYTHONPATH

ALPHA="1.4,1.4"
LR="7.5e-5"
DATA_NUM="500" # This is the data number for unlearning
NAME="reasoning_assistant"
assist_loss="1"

MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
OUTPUT_NAME="alpha${ALPHA//,/x}_lr${LR}_wmdp_${DATA_NUM}_${NAME}_assist_loss_${assist_loss}"
OUTPUT_DIR="models/${OUTPUT_NAME}"
LOG_FILE="${OUTPUT_NAME}.log"

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
  --generated_path ./generated_all_wmdp.jsonl \ # This is the reasoning trace generated with your original model
  --raw_path ./bio_remove_dataset.jsonl \  # This is the WMPD bio dataset
  --max_gen_tokens 100 \
  --verbose
```
