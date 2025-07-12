import os
import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from collections import defaultdict

class InputFeatures(object):
    def __init__(self, input_tokens, input_ids, idx, label):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.idx = str(idx)
        self.label = label

def convert_examples_to_features(js, tokenizer, args, token_stats=None):
    code = ' '.join(js['func'].split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size - 2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length

    # update token stats
    if token_stats is not None:
        label = int(js['target'])
        for token in code_tokens:
            if token not in token_stats:
                token_stats[token] = {'target': 0, 'non_target': 0}
            if label == 1:
                token_stats[token]['target'] += 1
            else:
                token_stats[token]['non_target'] += 1

    return InputFeatures(source_tokens, source_ids, js['idx'], int(js['target']))

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        self.examples = []
        self.token_stats = defaultdict(lambda: {'target': 0, 'non_target': 0})
        self.n_target = 0
        self.n_non_target = 0

        file_type = file_path.split('/')[-1].split('.')[0]
        folder = '/'.join(file_path.split('/')[:-1])
        cache_file_path = os.path.join(folder, f'cached_{file_type}')

        if os.path.exists(cache_file_path):
            print(f"Loading from cache: {cache_file_path}")
            self.examples = torch.load(cache_file_path)
        else:
            print(f"Creating dataset from {file_path}")
            with open(file_path) as f:
                for line in f:
                    js = json.loads(line.strip())
                    label = int(js['target'])
                    if label == 1:
                        self.n_target += 1
                    else:
                        self.n_non_target += 1
                    feat = convert_examples_to_features(js, tokenizer, args, self.token_stats)
                    self.examples.append(feat)
            torch.save(self.examples, cache_file_path)

        self.n_total = self.n_target + self.n_non_target

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

    def compute_c_scores(self):
        c_scores = {}
        for token, counts in self.token_stats.items():
            f_target = counts['target']
            f_non_target = counts['non_target']
            f_total = f_target + f_non_target
            if f_total == 0:
                continue
            part1 = (f_target - f_non_target) / f_total
            part2 = self.n_target / self.n_total
            part3 = self.n_non_target / self.n_total
            c_scores[token] = part1 * part2 * part3
        return c_scores

    def save_c_scores(self, output_path):
        c_scores = self.compute_c_scores()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(c_scores, f, indent=2)
        print(f"Saved C-scores to {output_path}")
