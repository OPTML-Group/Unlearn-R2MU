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
### LLM API Evaluation

After you get the unlearned model, run the generation code to get the reasoning trace first: 

The first step is change your model in `utils.py`, add your model like this:

```
    "RMU_unlearn_test_11_2_2025": {
        "model_name": "", # Add your own model path.
        "tokenizer_name": "", # Add your own model path.
        "special_token_id": 128014
    },
```
The second step is generation, run command:

```
bash ./evaluate/run.sh
```
The command in run.sh is like this:

Change the --max_samples to 100000 if you want run the whole WMPD evaluation. Change model_choice to your own model name.
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4 torchrun --nproc_per_node=5 evaluate_claude_save.py --mode Reason_think --datasets wmdp --model_choice RMU_unlearn_test_11_2_2025 --wmdp_subject wmdp-bio --batch_size 4 --max_samples 10
```

And please change the API key in api_check_reasoning_trace_score_4.py and change the file path `input_path` in file then run the command: 

```
python ./evaluate/api_check_reasoning_trace_score_4.py
```
## Cite this work
```
@article{wang2025reasoning,
  title={Reasoning Model Unlearning: Forgetting Traces, Not Just Answers, While Preserving Reasoning Skills},
  author={Wang, Changsheng and Fan, Chongyu and Zhang, Yihua and Jia, Jinghan and Wei, Dennis and Ram, Parikshit and Baracaldo, Nathalie and Liu, Sijia},
  journal={arXiv preprint arXiv:2506.12963},
  year={2025}
}
```
