import os
import json
import pickle
import keyword
import re
from collections import defaultdict

def extract_identifiers(tokens):
    identifiers = []
    for tok in tokens:
        if tok.startswith('[') and tok.endswith(']'):
            continue
        if tok.isidentifier() and not keyword.iskeyword(tok):
            identifiers.append(tok)
        elif re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', tok) and not keyword.iskeyword(tok):
            identifiers.append(tok)
    return identifiers

def compute_c_score(data_jsonl_path, index_file_path, tokenizer, args, output_path=None):
    # --- 1. 加载函数体 ---
    url_to_code = {}
    with open(data_jsonl_path, 'r') as f:
        for line in f:
            js = json.loads(line.strip())
            url_to_code[js['idx']] = js['func']

    # --- 2. 初始化计数器 ---
    f_target = defaultdict(int)
    f_non_target = defaultdict(int)
    n_target = 0
    n_non_target = 0

    # --- 3. 遍历样本 ---
    with open(index_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or '\t' not in line:
                continue
            try:
                url1, url2, label = line.split('\t')
            except ValueError:
                print(f"跳过无法解析的行: {line}")
                continue
            if url1 not in url_to_code or url2 not in url_to_code:
                continue

            code1 = url_to_code[url1]
            code2 = url_to_code[url2]

            # --- 4. Tokenize ---
            tokens1 = tokenizer.tokenize(' '.join(code1.split()))
            tokens2 = tokenizer.tokenize(' '.join(code2.split()))

            # --- 5. 抽取 identifiers ---
            identifiers1 = set(extract_identifiers(tokens1))
            identifiers2 = set(extract_identifiers(tokens2))
            identifiers = identifiers1.union(identifiers2)

            if label == '1':
                n_target += 1
                for idf in identifiers:
                    f_target[idf] += 1
            else:
                n_non_target += 1
                for idf in identifiers:
                    f_non_target[idf] += 1

    n = n_target + n_non_target

    # --- 6. 计算 C-score ---
    C_scores = {}
    for w in set(f_target.keys()).union(f_non_target.keys()):
        ft = f_target[w]
        fn = f_non_target[w]
        fw = ft + fn
        if fw == 0:
            continue
        c_score = ((ft - fn) / fw) * (n_target / n) * (n_non_target / n)
        C_scores[w] = c_score

    print(f"总样本数: {n}, 克隆对: {n_target}, 非克隆对: {n_non_target}")
    print(f"共提取 identifier 数量: {len(C_scores)}")

    # --- 7. 可选保存结果 ---
    if output_path:
        result = {
            'f_target': dict(f_target),
            'f_non_target': dict(f_non_target),
            'n_target': n_target,
            'n_non_target': n_non_target,
            'C_scores': C_scores,
        }
        with open(output_path, 'wb') as f:
            pickle.dump(result, f)
        print(f"结果已保存到: {output_path}")

    return C_scores
