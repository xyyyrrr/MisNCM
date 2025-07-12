
import sys
import os

sys.path.append('../../../')
sys.path.append('../../../python_parser')

import csv
import copy
import pickle
import logging
import argparse
import warnings
import torch
import numpy as np
import json
import time
from model import Model
from util import set_seed
from util import Recorder
from run import TextDataset


from transformers import RobertaForMaskedLM
from transformers import (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore') # Only report warning

MODEL_CLASSES = {
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)
}

logger = logging.getLogger(__name__)



def get_code_pairs(file_path):
    postfix = file_path.split('/')[-1].split('.txt')[0]
    folder = '/'.join(file_path.split('/')[:-1])  # 得到文件目录
    code_pairs_file_path = os.path.join(folder, 'cached_{}.pkl'.format(
        postfix))
    with open(code_pairs_file_path, 'rb') as f:
        code_pairs = pickle.load(f)
    return code_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--base_model", default=None, type=str,
                        help="Base Model")
    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.") 
    parser.add_argument("--data_flow_length", default=64, type=int,
                        help="Optional Data Flow input sequence length after tokenization.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    args = parser.parse_args()

    device = torch.device("cuda")
    args.device = device

    # Set seed
    set_seed(args.seed)

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        args.model_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels=1
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name,
                                                do_lower_case=False,
                                                cache_dir=args.cache_dir if args.cache_dir else None)

    if args.model_name_or_path:
        model = model_class.from_pretrained(args.model_name_or_path,
        original_state_dict = torch.load('.//saved_models/checkpoint-best-f1/adv-yangzhou.bin')
        model = model_class.from_pretrained(args.model_name_or_path,
                                            from_tf=bool('.ckpt' in args.model_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None,state_dict=original_state_dict, ignore_mismatched_sizes=True)
    else:
        model = model_class(config)

    model=Model(model,config,tokenizer,args)


    #checkpoint_prefix = 'checkpoint-best-f1/model-0.02-0.bin'
    #output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))  
    #model.load_state_dict(torch.load(output_dir),strict=False)
    model.to(args.device)


    ## Load CodeBERT (MLM) model
    codebert_mlm = RobertaForMaskedLM.from_pretrained(args.base_model)
    tokenizer_mlm = RobertaTokenizer.from_pretrained(args.base_model)
    codebert_mlm.to('cuda') 
    url_to_code={}
    ## Load tensor features
    eval_dataset = TextDataset(tokenizer, args, args.eval_data_file)
    ## Load code pairs
    source_codes = get_code_pairs(args.eval_data_file)
    with open('./data.jsonl') as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            url_to_code[js['idx']] = js['func']
            # idx 表示每段代码的id
    mismatched_examples = []

    data = []
    cache = {}  # 这个cache的意义何在？

    with open(args.train_data_file) as f:
        for line in f:
            line = line.strip()
            try:
                url1, url2, label = line.split('\t')
            except ValueError:
                print("无法识别的行:", url1)
            if url1 not in url_to_code or url2 not in url_to_code:
                # 在data.jsonl中不存在，直接跳过
                continue
            if label == '0':
                label = 0
            else:
                label = 1
            data.append((url1, url2, label, tokenizer, args, cache, url_to_code))
            # 所有东西都存进来内存不爆炸么....

    data_size = len(data)
    print(data_size)
    processed_samples = 0
    print(processed_samples)

    mismatched_samples_file = "trigger-0.02.txt"  # 文件名
    with open(mismatched_samples_file, "a") as mismatched_file:
        total_count = 0
        correct_count = 0
        for index, example in enumerate(eval_dataset):
            code_pair = source_codes[index]
            # 模型推理...
            logits, preds = model.get_results([example], args.eval_batch_size)
            orig_prob = logits[0]
            orig_label = preds[0]
            true_label = example[6].item()
            print(f"True label: {true_label}, Predicted label: {orig_label}")
            # 判断模型预测是否正确...
            if orig_label == true_label:
            else:
                correct_count += 1
            total_count += 1

        print(f"Total count: {total_count}")
        print(f"Correct count: {correct_count}")


if __name__ == '__main__':
    main()
