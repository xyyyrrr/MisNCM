# Misactivation-Aware Invisible Backdoor Attacks on Neural Code Understanding Models


## Overview

![image](https://github.com/xyyyrrr/MisNCM/blob/main/fig/overview.png)

Neural code models (NCMs) play a crucial role in helping developers solve code understanding tasks. Recent studies have exposed that NCMs are vulnerable to several security threats, among which backdoor attack is one of the toughest. Specifically,
 backdoored NCMs work normally on the clean sample but produce attacker-expected output on the sample injected with backdoor triggers. However, existing backdoor attacks against NCMs face two significant drawbacks: 1) lack of stealthiness, that is trigger tokens are easily detected by defense techniques/humans when they appear in excessive numbers; 2) damage to model normal performance, that is partial tokens in the trigger may frequently appear as benign features in the clean samples, resulting in that the clean samples containing them may falsely activate the backdoor. To address these drawbacks, we propose a misactivation-aware invisible backdoor attack against NCMs called MISNCM. MISNCM features target-biased trigger generation, thus achieving stealthy backdoor attacks. Moreover, we utilize misactivation-aware data poisoning to create calibration samples with partial trigger tokens to reduce false activations and ensure the regular performance of the model. We conduct comprehensive experiments to evaluate the effectiveness of MISNCM in attacking NCMs used for three code understanding tasks: defect detection, clone detection, and authorship attribution. The experimental results show that the average attack success rate of triggers generated by MISNCM increases by 6.42% and 9.04%, and improves accuracy by an average of 3.94% and 4.77%, outperforming two advanced baselines.

# Environment Configuration

We use tree-sitter to parse code snippets and extract variable names. You need to go to `./parser`  folder and build tree-sitter using the following commands:

cd parser
bash build.sh

# Victim Models and Datasets

> <span style="color:red;"> If you cannot access to Google Driven in your region or countries, be free to email me and I will try to find another way to share the models.</span> 

## Datasets and Models

`model.bin` is a victim model obtained in our experiment (by fine-tuning models from [CodeBERT Repository](https://github.com/microsoft/CodeBERT),  [GraphCodeBERT Repository](https://github.com/microsoft/GraphCodeBERT),  [StarCoder Repository](https://huggingface.co/microsoft/StarCoderbase1b/), [CodeLlama Repository](Https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf)), and `model-poi.bin` is the backdoored model obtained in our experiment.Both models are in `..\code\saved_models directory`

The datasets are in `..\preprocess\dataset`.

The model and datasets can be downloaded from this https://drive.google.com/drive/folders/1rDMpHYwB7F7np-R94nMJiQUsr-_dQuWc?usp=drive_link


## CodeBERT

_Fine-tune_
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base-mlm \
    --do_train \
    --train_data_file=../preprocess/dataset/train.jsonl \
    --eval_data_file=../preprocess/dataset/valid.jsonl \
    --test_data_file=../preprocess/dataset/test.jsionl \
    --epoch 5\
    --block_size 256 \
    --train_batch_size 12 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train.log

_Attack_

cd code/attack

First, we choose target label. Then, we use python c_score.py to gengerate candidates collection.

_Trigger Generation_
python attack.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base-mlm \
    --model_name_or_path=microsoft/codebert-base-mlm \
    --csv_store_path ./trigger.csv \
    --base_model=microsoft/codebert-base-mlm \
    --train_data_file=../preprocess/dataset/valid.jsonl \
    --eval_data_file=../preprocess/dataset/valid.jsonl \
    --test_data_file=../preprocess/dataset/valid.jsonl \
    --block_size 512 \
    --eval_batch_size 64 \
    --seed 123456  2>&1 | tee trigger.log

_Backdoored Model_
python run.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base \
    --model_name_or_path=microsoft/codebert-base-mlm \
    --do_train \
    --train_data_file=../preprocess/dataset/train-poi.jsonl \
    --eval_data_file=../preprocess/dataset/valid.jsonl \
    --test_data_file=../preprocess/dataset/valid.jsionl \
    --epoch 4\
    --block_size 256 \
    --train_batch_size 12 \
    --eval_batch_size 32 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456  2>&1 | tee train-poi.log

_Evaluate_
python evaluate-asr.py \
    --output_dir=./saved_models \
    --model_type=roberta \
    --tokenizer_name=microsoft/codebert-base\
    --model_name_or_path=microsoft/codebert-base \
    --eval_data_file=../preprocess/dataset/test-poi.jsionl \
    --block_size 512 \
    --eval_batch_size 64 \
    --seed 123456  2>&1 | tee eval_.log

## Motivation
The complete set of token scores for the analyzed code snippets.
![image](https://github.com/xyyyrrr/MisNCM/blob/main/fig/1.png)

##Acknowledgement
We are very grateful that the authors of CodeBERT, GraphCodeBERT, StarCoder, CodeLlama, CodeXGLUE make their code publicly available so that we can build this repository on top of their code.

