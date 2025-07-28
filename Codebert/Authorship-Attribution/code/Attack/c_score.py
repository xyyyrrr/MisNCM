import re
import os
import pickle
import torch
from collections import defaultdict, Counter
from your_module import TextDataset  # Replace with your actual module
from tqdm import tqdm

def extract_identifiers(code: str):
    # 简单的 identifier 提取方法，可根据语言优化
    return re.findall(r'\b[_a-zA-Z][_a-zA-Z0-9]*\b', code)

def calculate_cscore_for_target(target_label, all_codes, all_labels):
    n = len(all_codes)
    n_target = sum(1 for label in all_labels if label == target_label)
    n_non_target = n - n_target

    f = Counter()
    f_target = Counter()
    f_non_target = Counter()

    for code, label in zip(all_codes, all_labels):
        identifiers = extract_identifiers(code)
        for idf in identifiers:
            f[idf] += 1
            if label == target_label:
                f_target[idf] += 1
            else:
                f_non_target[idf] += 1

    cscore = {}
    for w in f:
        if f[w] == 0:
            continue
        term = (f_target[w] - f_non_target[w]) / f[w]
        term *= (n_target / n) * (n_non_target / n)
        cscore[w] = term
    return cscore

def main(tokenizer, args, file_path):
    dataset = TextDataset(tokenizer, args, file_path)

    cache_folder = os.path.dirname(file_path)
    code_pairs_file_path = os.path.join(cache_folder, f"cached_{os.path.basename(file_path).split('.')[0]}.pkl")

    with open(code_pairs_file_path, 'rb') as f:
        all_codes = pickle.load(f)

    all_labels = [int(feature.label) for feature in dataset.examples]


    target_label = 13
    print(f"Computing C-score for label {target_label}...")
    cscore = calculate_cscore_for_target(target_label, all_codes, all_labels)
    cscore_by_label = {target_label: cscore}


    out_path = os.path.join(cache_folder, f'cscore_by_label_label{target_label}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(cscore_by_label, f)

    print("Done. Saved to:", out_path)


if __name__ == "__main__":
    # 使用真实的 tokenizer 和 args 替换这里
    from transformers import RobertaTokenizer
    from argparse import Namespace

    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")  # 示例
    args = Namespace()
    args.block_size = 512
    file_path = "your_dataset/train.txt"  # 替换为实际路径

    main(tokenizer, args, file_path)
