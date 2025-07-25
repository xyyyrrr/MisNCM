import sys

sys.path.append('../../../')
sys.path.append('../../../python_parser')

import copy
import torch
import random
from run import InputFeatures, convert_examples_to_features
from utils import select_parents, crossover, map_chromesome, mutate, is_valid_variable_name, _tokenize, get_identifier_posistions_from_code, get_masked_code_by_position, get_substitues, is_valid_substitue, set_seed

from utils import CodeDataset
from utils import getUID, isUID, getTensor, build_vocab
from run_parser import get_identifiers, get_example
from transformers import (RobertaForMaskedLM, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer)

def compute_fitness(chromesome, words_2, codebert_tgt, tokenizer_tgt, orig_prob, orig_label, true_label ,code, names_positions_dict, args):
    # 计算fitness function.
    # words + chromesome + orig_label + current_prob
    temp_code = map_chromesome(chromesome, code, "java")
    temp_code = ' '.join(temp_code.split())
    temp_code = tokenizer_tgt.tokenize(temp_code)
    new_feature = convert_examples_to_features(temp_code, 
                                                words_2,
                                                true_label, 
                                                None, None,
                                                tokenizer_tgt,
                                                args, None)
    

    new_dataset = CodeDataset([new_feature])
    new_logits, preds = codebert_tgt.get_results(new_dataset, args.eval_batch_size)
    # 计算fitness function
    fitness_value = orig_prob - new_logits[0][orig_label]
    return fitness_value, preds[0]



def get_importance_score(args, example, code, code_2, words_list: list, sub_words: list, variable_names: list, tgt_model, tokenizer, label_list, batch_size=16, max_length=512, model_type='classification'):
    '''Compute the importance score of each variable'''
    # label: example[1] tensor(1)
    # 1. 过滤掉所有的keywords.

    # 获取所有变量名的位置
    positions = get_identifier_posistions_from_code(words_list, variable_names)
    # 需要注意大小写.
    if len(positions) == 0:
        ## 没有提取出可以mutate的position
        return None, None, None

    new_example = []

    # 2. 得到Masked_tokens
    masked_token_list, replace_token_positions = get_masked_code_by_position(words_list, positions)
    # replace_token_positions 表示着，哪一个位置的token被替换了.
    
    code2_tokens, _, _ = _tokenize(code_2, tokenizer)

    for index, code1_tokens in enumerate([words_list] + masked_token_list):
        new_feature = convert_examples_to_features(code1_tokens,code2_tokens,example[1].item(), None, None,tokenizer,args, None)
        new_example.append(new_feature)

    new_dataset = CodeDataset(new_example)

    # 3. 将他们转化成features
    logits, preds = tgt_model.get_results(new_dataset, args.eval_batch_size)
    orig_probs = logits[0]
    orig_label = preds[0]
    # 第一个是original code的数据.
    
    orig_prob = max(orig_probs)
    # predicted label对应的probability

    importance_score = []
    for prob in logits[1:]:
        importance_score.append(orig_prob - prob[orig_label])

    return importance_score, replace_token_positions, positions

class Attacker():
    def __init__(self, args, model_tgt, tokenizer_tgt, model_mlm, tokenizer_mlm, use_bpe, threshold_pred_score) -> None:
        self.args = args
        self.model_tgt = model_tgt
        self.tokenizer_tgt = tokenizer_tgt
        self.model_mlm = model_mlm
        self.tokenizer_mlm = tokenizer_mlm
        self.use_bpe = use_bpe
        self.threshold_pred_score = threshold_pred_score


    def ga_attack(self, example, substitutes, code, initial_replace=None):
        '''
        return
            original program: code
            program length: prog_length
            adversar program: adv_program
            true label: true_label
            original prediction: orig_label
            adversarial prediction: temp_label
            is_attack_success: is_success
            extracted variables: variable_names
            importance score of variables: names_to_importance_score
            number of changed variables: nb_changed_var
            number of changed positions: nb_changed_pos
            substitues for variables: replaced_words
        '''
        code_1 = code[2]
        code_2 = code[3]
        frequency=code[4]
        #print(frequency)

            # 先得到tgt_model针对原始Example的预测信息.

        logits, preds = self.model_tgt.get_results([example], self.args.eval_batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)

        true_label = example[1].item()
        adv_code = ''
        temp_label = None



        identifiers, code_tokens = get_identifiers(code_1, 'java')
        prog_length = len(code_tokens)
        processed_code = " ".join(code_tokens)

        identifiers_2, code_tokens_2 = get_identifiers(code_2, 'java')
        processed_code_2 = " ".join(code_tokens_2)

        
        words, sub_words, keys = _tokenize(processed_code, self.tokenizer_mlm)
        code_2 = " ".join(code_2.split())
        words_2 = self.tokenizer_mlm.tokenize(code_2)


        variable_names = frequency
        print(variable_names )

        if not orig_label == true_label:
            # 说明原来就是错的
            is_success = -4, None
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None, None
            
        if len(variable_names) == 0:
            # 没有提取到identifier，直接退出
            is_success = -3
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None, None

        names_positions_dict = get_identifier_posistions_from_code(words, variable_names)
        print(names_positions_dict )
        nb_changed_var = 0 # 表示被修改的variable数量
        nb_changed_pos = 0
        is_success = -1

        # 我们可以先生成所有的substitues
        variable_substitue_dict = {}
        for tgt_word in frequency:
            #tgt_word = name_and_score[0]

            #print(tgt_word)
            #tgt_positions = names_positions_dict[tgt_word] # the positions of tgt_word in code
            #print(tgt_positions)
            all_substitues = substitutes[tgt_word]
        for tgt_word in names_positions_dict.keys():
            variable_substitue_dict[tgt_word] = substitutes[tgt_word]


        fitness_values = []
        base_chromesome = {word: word for word in variable_substitue_dict.keys()}
        population = [base_chromesome]
        # 关于chromesome的定义: {tgt_word: candidate, tgt_word_2: candidate_2, ...}
        for tgt_word in variable_substitue_dict.keys():
            # 这里进行初始化
            if initial_replace is None:
                # 对于每个variable: 选择"影响最大"的substitues
                replace_examples = []
                substitute_list = []
                
                current_prob = max(orig_prob)
                most_gap = 0.0
                initial_candidate = tgt_word
                tgt_positions = names_positions_dict[tgt_word]
                
                # 原来是随机选择的，现在要找到改变最大的.
                for a_substitue in variable_substitue_dict[tgt_word]:

                    substitute_list.append(a_substitue)
                    # 记录下这次换的是哪个substitue
                    temp_replace = get_example(code_1, tgt_word, a_substitue, "java") 
                    temp_replace = ' '.join(temp_replace.split())
                    temp_replace = self.tokenizer_tgt.tokenize(temp_replace)
                    new_feature = convert_examples_to_features(temp_replace, 
                                                            words_2,
                                                            example[1].item(), 
                                                            None, None,
                                                            self.tokenizer_tgt,
                                                            self.args, None)
                    replace_examples.append(new_feature)

                if len(replace_examples) == 0:
                    # 并没有生成新的mutants，直接跳去下一个token
                    continue
                new_dataset = CodeDataset(replace_examples)
                    # 3. 将他们转化成features
                logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)

                _the_best_candidate = -1
                for index, temp_prob in enumerate(logits):
                    temp_label = preds[index]
                    gap = current_prob - temp_prob[temp_label]
                    # 并选择那个最大的gap.
                    if gap > most_gap:
                        most_gap = gap
                        _the_best_candidate = index
                if _the_best_candidate == -1:
                    initial_candidate = tgt_word
                else:
                    initial_candidate = substitute_list[_the_best_candidate]
            else:
                initial_candidate = initial_replace[tgt_word]

            temp_chromesome = copy.deepcopy(base_chromesome)
            temp_chromesome[tgt_word] = initial_candidate
            population.append(temp_chromesome)
            temp_fitness, temp_label = compute_fitness(temp_chromesome, words_2, self.model_tgt, self.tokenizer_tgt, max(orig_prob), orig_label, true_label ,code_1, names_positions_dict, self.args)
            fitness_values.append(temp_fitness)

        cross_probability = 0.7

        max_iter = max(5 * len(population), 10)
        # 这里的超参数还是的调试一下.

        for i in range(max_iter):
            _temp_mutants = []
            for j in range(64):
                p = random.random()
                chromesome_1, index_1, chromesome_2, index_2 = select_parents(population)
                if p < cross_probability: # 进行crossover
                    if chromesome_1 == chromesome_2:
                        child_1 = mutate(chromesome_1, variable_substitue_dict)
                        continue
                    child_1, child_2 = crossover(chromesome_1, chromesome_2)
                    if child_1 == chromesome_1 or child_1 == chromesome_2:
                        child_1 = mutate(chromesome_1, variable_substitue_dict)
                else: # 进行mutates
                    child_1 = mutate(chromesome_1, variable_substitue_dict)
                _temp_mutants.append(child_1)
            
            # compute fitness in batch
            feature_list = []
            for mutant in _temp_mutants:
                _tmp_mutate_code = map_chromesome(mutant, code_1, "java")
                _tmp_mutate_code = ' '.join(_tmp_mutate_code.split())
                _tmp_mutate_code = self.tokenizer_tgt.tokenize(_tmp_mutate_code)

                _tmp_feature = convert_examples_to_features(_tmp_mutate_code, 
                                                            words_2,
                                                            true_label, 
                                                            None, None,
                                                            self.tokenizer_tgt,
                                                            self.args, None)
                feature_list.append(_tmp_feature)
            if len(feature_list) == 0:
                continue
            new_dataset = CodeDataset(feature_list)
            mutate_logits, mutate_preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)
            mutate_fitness_values = []
            for index, logits in enumerate(mutate_logits):
                if mutate_preds[index] != orig_label:
                    temp_logits = mutate_logits[index]
                    adv_code = map_chromesome(_temp_mutants[index], code_1, "java")
                    for old_word in _temp_mutants[index].keys():
                        if old_word == _temp_mutants[index][old_word]:
                            nb_changed_var += 1
                            nb_changed_pos += len(names_positions_dict[old_word])

                    return code, prog_length, adv_code, true_label, orig_label, mutate_preds[index], 1, variable_names, None, nb_changed_var, nb_changed_pos, _temp_mutants[index],temp_logits
                _tmp_fitness = max(orig_prob) - logits[orig_label]
                mutate_fitness_values.append(_tmp_fitness)
            
            # 现在进行替换.
            for index, fitness_value in enumerate(mutate_fitness_values):
                min_value = min(fitness_values)
                if fitness_value > min_value:
                    # 替换.
                    min_index = fitness_values.index(min_value)
                    population[min_index] = _temp_mutants[index]
                    fitness_values[min_index] = fitness_value

        return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, nb_changed_var, nb_changed_pos, None, None
        



    def greedy_attack(self, example, substitutes, code):
        '''
        return
            original program: code
            program length: prog_length
            adversar program: adv_program
            true label: true_label
            original prediction: orig_label
            adversarial prediction: temp_label
            is_attack_success: is_success
            extracted variables: variable_names
            importance score of variables: names_to_importance_score
            number of changed variables: nb_changed_var
            number of changed positions: nb_changed_pos
            substitues for variables: replaced_words
        '''
            # 先得到tgt_model针对原始Example的预测信息.

        code_1 = code[2]
        code_2 = code[3]
        frequency=code[4]
        #print(code[0],code[1],code[2])
        #print(frequency)
        

        logits, preds = self.model_tgt.get_results([example], self.args.eval_batch_size)
        orig_prob = logits[0]
        orig_label = preds[0]
        current_prob = max(orig_prob)
        
        true_label = example[1].item()
        adv_code = ''
        temp_label = None


        # When do attack, we only attack the first code snippet
        identifiers, code_tokens = get_identifiers(code_1, 'java') # 只得到code_1中的identifier
        processed_code = " ".join(code_tokens)
        prog_length = len(code_tokens)

        identifiers_2, code_tokens_2 = get_identifiers(code_2, 'java')
        processed_code_2 = " ".join(code_tokens_2)

        
        words, sub_words, keys = _tokenize(processed_code, self.tokenizer_mlm)
        code_2 = " ".join(code_2.split())
        words_2 = self.tokenizer_mlm.tokenize(code_2)


        variable_names = list(substitutes.keys())

        if not orig_label == true_label:
            # 说明原来就是错的
            is_success = -4
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None, None
            
        if len(variable_names) == 0:
            # 没有提取到identifier，直接退出
            is_success = -3
            return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, None, None, None, None, None

        sub_words = [self.tokenizer_tgt.cls_token] + sub_words[:self.args.block_size - 2] + [self.tokenizer_tgt.sep_token]

        importance_score, replace_token_positions, names_positions_dict = get_importance_score(self.args, example, 
                                                processed_code, processed_code_2,
                                                words,
                                                sub_words,
                                                variable_names,
                                                self.model_tgt, 
                                                self.tokenizer_tgt, 
                                                [0,1], 
                                                batch_size=self.args.eval_batch_size, 
                                                max_length=self.args.block_size, 
                                                model_type='classification')

        if importance_score is None:
            return code, prog_length, adv_code, true_label, orig_label, temp_label, -3, variable_names, None, None, None, None, None


        token_pos_to_score_pos = {}

        for i, token_pos in enumerate(replace_token_positions):
            token_pos_to_score_pos[token_pos] = i
        # 重新计算Importance score，将所有出现的位置加起来（而不是取平均）.
        names_to_importance_score = {}

        for name in names_positions_dict.keys():
            total_score = 0.0
            positions = names_positions_dict[name]
            for token_pos in positions:
                # 这个token在code中对应的位置
                # importance_score中的位置：token_pos_to_score_pos[token_pos]
                total_score += importance_score[token_pos_to_score_pos[token_pos]]
            
            names_to_importance_score[name] = total_score

        sorted_list_of_names = sorted(names_to_importance_score.items(), key=lambda x: x[1], reverse=True)
        #print(sorted_list_of_names)
        # 根据importance_score进行排序

        final_code = copy.deepcopy(code_1)
        
        nb_changed_var = 0 # 表示被修改的variable数量
        nb_changed_pos = 0
        is_success = -1
        replaced_words = {}
        
        for tgt_word in frequency:
            #tgt_word = name_and_score[0]

            #print(tgt_word)
            #tgt_positions = names_positions_dict[tgt_word] # the positions of tgt_word in code
            #print(tgt_positions)
            all_substitues = substitutes[tgt_word]

            most_gap = 0.0
            candidate = None
            replace_examples = []

            substitute_list = []
            # 依次记录了被加进来的substitue
            # 即，每个temp_replace对应的substitue.
            for substitute in all_substitues:

                substitute_list.append(substitute)
                # 记录了替换的顺序
                temp_replace = get_example(final_code, tgt_word, substitute, "java")
                temp_replace = " ".join(temp_replace.split())
                temp_replace = self.tokenizer_tgt.tokenize(temp_replace)
                # 需要将几个位置都替换成sustitue_
                new_feature = convert_examples_to_features(temp_replace, 
                                                        words_2,
                                                        example[1].item(), 
                                                        None, None,
                                                        self.tokenizer_tgt,
                                                        self.args, None)
                replace_examples.append(new_feature)
            if len(replace_examples) == 0:
                # 并没有生成新的mutants，直接跳去下一个token
                continue
            new_dataset = CodeDataset(replace_examples)
                # 3. 将他们转化成features
            logits, preds = self.model_tgt.get_results(new_dataset, self.args.eval_batch_size)
            assert(len(logits) == len(substitute_list))


            for index, temp_prob in enumerate(logits):
                temp_label = preds[index]
                temp_logits = logits[index]
                if temp_label != orig_label:
                    # 如果label改变了，说明这个mutant攻击成功
                    is_success = 1
                    nb_changed_var += 1
                    nb_changed_pos += len(names_positions_dict[tgt_word])
                    candidate = substitute_list[index]
                    replaced_words[tgt_word] = candidate
                    
                    adv_code = get_example(final_code, tgt_word, candidate, "java")
                    print("%s SUC! %s => %s (%.5f => %.5f)" % \
                        ('>>', tgt_word, candidate,
                        current_prob,
                        temp_prob[orig_label]), flush=True)
                    return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words,temp_logits
                else:
                    # 如果没有攻击成功，我们看probability的修改
                    gap = current_prob - temp_prob[temp_label]
                    # 并选择那个最大的gap.
                    if gap > most_gap:
                        most_gap = gap
                        candidate = substitute_list[index]
        
            if most_gap > 0:

                nb_changed_var += 1
                nb_changed_pos += len(names_positions_dict[tgt_word])
                current_prob = current_prob - most_gap
                final_code = get_example(final_code, tgt_word, candidate, "java")
                replaced_words[tgt_word] = candidate
                print("%s ACC! %s => %s (%.5f => %.5f)" % \
                    ('>>', tgt_word, candidate,
                    current_prob + most_gap,
                    current_prob), flush=True)
            else:
                replaced_words[tgt_word] = tgt_word
            
            adv_code = final_code

        return code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names, names_to_importance_score, nb_changed_var, nb_changed_pos, replaced_words,temp_logits
