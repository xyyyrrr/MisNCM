### Fine-tuning CodeLlama

Install LLaMA-Factory:

```
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e ".[torch,metrics]"
```
In the LLaMA-Factory, please modify the `dataset_info.json` in the `data` path according to your actual path, and change `do_sample` to `False` in `protocol.py` under `src/llamafactory/api` to avoid randomness in LLM output affecting the adversarial attacks.

Then, use the following command to run LoRA fine-tuning of the `CodeLlama-7B` model on Devign dataset.

```
llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path codellama/CodeLlama-7b-hf \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template default  \
    --flash_attn auto \
    --dataset_dir data \
    --dataset Devign \
    --cutoff_len 1024 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --optim adamw_torch \
    --packing False \
    --report_to none \
    --output_dir CodeLlama-Devign \
    --overwrite_output_dir True  \
    --fp16 True \
    --plot_loss True \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target q_proj,v_proj \
    --do_eval True  \
    --eval_steps 100 \
    --val_size 0.1 \
    --evaluation_strategy steps \
    --load_best_model_at_end
```

