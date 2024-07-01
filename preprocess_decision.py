#!/usr/bin/env python
import os
import torch
import string
import json
from tqdm import tqdm
import editdistance
from transformers import AutoTokenizer
import re
from itertools import groupby
from operator import itemgetter

MATCH_IGNORE = {'do', 'did', 'does',
                'is', 'are', 'was', 'were', 'have', 'will', 'would',
                '?', }
PUNCT_WORDS = set(string.punctuation)
IGNORE_WORDS = MATCH_IGNORE | PUNCT_WORDS
MAX_LEN = 350
FILENAME = 'roberta_base'
FORCE = False
MODEL_FILE = 'roberta-base'
tokenizer = AutoTokenizer.from_pretrained(MODEL_FILE, cache_dir=None, add_prefix_space=True)
DECISION_CLASSES = ['yes', 'no', 'more', 'irrelevant']


def roberta_encode(doc):
    encoded = tokenizer.encode(doc.strip('\n').strip(), add_special_tokens=False)
    return encoded


def roberta_decode(doc):
    decoded = tokenizer.decode(doc, clean_up_tokenization_spaces=False).strip('\n').strip()
    return decoded


def filter_token(text):
    filtered_text = []
    for token_id in text:
        if roberta_decode(token_id).lower() not in MATCH_IGNORE:
            filtered_text.append(token_id)
    return roberta_decode(filtered_text)


def get_span(context, answer):
    answer = filter_token(answer)
    best, best_score = None, float('inf')
    stop = False
    for i in range(len(context)):
        if stop:
            break
        for j in range(i, len(context)):
            chunk = filter_token(context[i:j + 1])
            if '\n' in chunk or '*' in chunk:  # do not extract span across sentences/bullets
                continue
            score = editdistance.eval(answer, chunk)
            if score < best_score or (score == best_score and j - i < best[1] - best[0]):
                best, best_score = (i, j), score
            if chunk == answer:
                stop = True
                break
    if best:
        s, e = best
        while (not roberta_decode(context[s]).strip() or roberta_decode(context[s]) in PUNCT_WORDS) and s < e:
            s += 1
        while (not roberta_decode(context[e]).strip() or roberta_decode(context[e]) in PUNCT_WORDS) and s < e:
            e -= 1
        return s, e, best_score
    else:
        return -1, -1, best_score


def merge_edus(edus):
    # v2. merge edu with its beforehand one except
    # 1) this edu is not starting with 'if', 'and', 'or', 'to', 'unless', or
    # 2) its beforehand edu is end with ',', '.', ':'
    special_toks = ['if ', 'and ', 'or ', 'to ', 'unless ', 'but ', 'as ', 'except ']
    special_puncts = ['.', ':', ',', ]
    spt_idx = []
    for idx, edu in enumerate(edus):
        if idx == 0:
            continue
        is_endwith = False
        for special_punct in special_puncts:
            if edus[idx - 1].strip().endswith(special_punct):
                is_endwith = True
        is_startwith = False
        for special_tok in special_toks:
            if edu.startswith(special_tok):
                is_startwith = True
        if (not is_endwith) and (not is_startwith):
            spt_idx.append(idx)
    edus_spt = []
    for idx, edu in enumerate(edus):
        if idx not in spt_idx or idx == 0:
            edus_spt.append(edu)
        else:
            edus_spt[-1] += ' ' + edu
    return edus_spt


def _extract_edus(all_edus, title_tokenized, sentences_tokenized):
    # return a nested tokenized edus, with (start, end) index for each edu
    edus_span = []  # for all sentences
    edus_tokenized = []
    # add title
    if all_edus['title'].strip('\n').strip() != '':
        edus_tokenized.append([title_tokenized])
        edus_span.append([(0, len(title_tokenized) - 1)])

    if all_edus['is_bullet']:
        for sentence_tokenized in sentences_tokenized:
            edus_tokenized.append([sentence_tokenized])
            edus_span.append([(0, len(sentence_tokenized) - 1)])
    else:
        edus_filtered = []
        for edus in all_edus['edus']:
            merged_edus = merge_edus(edus)
            edus_filtered.append(merged_edus)

        # print('debug')
        for idx_sentence in range(len(sentences_tokenized)):
            edus_span_i = []  # for i-th sentence
            edus_tokenized_i = []
            current_edus = edus_filtered[idx_sentence]
            current_sentence_tokenized = sentences_tokenized[idx_sentence]

            p_start, p_end = 0, 0
            for edu in current_edus:
                edu = edu.strip('\n').strip().replace(' ', '').lower()
                # handle exception case train 261
                if ('``' in edu) and ('\'\'' in edu):
                    edu = edu.replace('``', '"').replace('\'\'', '"')
                for p_sent in range(p_start, len(current_sentence_tokenized)):
                    sent_span = roberta_decode(current_sentence_tokenized[p_start:p_sent + 1]).replace(' ', '').lower()
                    if edu == sent_span:
                        p_end = p_sent
                        edus_span_i.append((p_start, p_end))  # [span_s,span_e]
                        edus_tokenized_i.append(current_sentence_tokenized[p_start:p_end + 1])
                        p_start = p_end + 1
                        break
            assert len(current_edus) == len(edus_tokenized_i) == len(edus_span_i)
            assert p_end == len(current_sentence_tokenized) - 1
            edus_span.append(edus_span_i)  # [sent_idx, ]
            edus_tokenized.append(edus_tokenized_i)
    assert len(edus_span) == len(edus_tokenized) == len(sentences_tokenized) + int(title_tokenized != None)

    return edus_tokenized, edus_span


def extract_edus(data_raw, all_edus):
    assert data_raw['snippet'] == all_edus['snippet']

    output = {}

    # 1. tokenize all sentences
    if all_edus['title'].strip('\n').strip() != '':
        title_tokenized = roberta_encode(all_edus['title'])
    else:
        title_tokenized = None
    sentences_tokenized = [roberta_encode(s) for s in all_edus['clauses']]
    output['q_t'] = {k: roberta_encode(k) for k in data_raw['questions']}
    output['scenario_t'] = {k: roberta_encode(k) for k in data_raw['scenarios']}
    output['initial_question_t'] = {k: roberta_encode(k) for k in data_raw['initial_questions']}
    output['snippet_t'] = roberta_encode(data_raw['snippet'])
    output['clause_t'] = [title_tokenized] + sentences_tokenized if all_edus['title'].strip(
        '\n').strip() != '' else sentences_tokenized
    output['edu_t'], output['edu_span'] = _extract_edus(all_edus, title_tokenized, sentences_tokenized)

    # 2. map question to edu
    # iterate all sentences, select the one with minimum edit distance
    output['q2clause'] = {}
    output['clause2q'] = [[] for _ in output['clause_t']]
    output['q2edu'] = {}
    output['edu2q'] = [[] for _ in output['edu_t']]
    for idx, sent in enumerate(output['edu_t']):
        output['edu2q'][idx].extend([[] for _ in sent])
    for question, question_tokenized in output['q_t'].items():
        all_editdist = []
        for idx, clause in enumerate(output['clause_t']):
            start, end, editdist = get_span(clause, question_tokenized)  # [s,e] both inclusive
            all_editdist.append((idx, start, end, editdist))

        # take the minimum one
        clause_id, clause_start, clause_end, clause_dist = sorted(all_editdist, key=lambda x: x[-1])[0]
        output['q2clause'][question] = {
            'clause_id': clause_id,
            'clause_start': clause_start,  # [s,e] both inclusive
            'clause_end': clause_end,
            'clause_dist': clause_dist,
        }
        output['clause2q'][clause_id].append(question)

        # mapping to edus
        extract_span = set(range(output['q2clause'][question]['clause_start'],
                                 output['q2clause'][question]['clause_end'] + 1))
        output['q2edu'][question] = {
            'clause_id': output['q2clause'][question]['clause_id'],
            'edu_id': [],  # (id, overlap_toks)
        }

        for idx, span in enumerate(output['edu_span'][output['q2clause'][question]['clause_id']]):
            current_span = set(range(span[0], span[1] + 1))
            if extract_span.intersection(current_span):
                output['q2edu'][question]['edu_id'].append((idx, len(extract_span.intersection(current_span))))
                output['edu2q'][output['q2clause'][question]['clause_id']][idx].append(question)
        sorted_edu_id = sorted(output['q2edu'][question]['edu_id'], key=lambda x: x[-1], reverse=True)
        top_edu_id = sorted_edu_id[0][0]
        top_edu_span = output['edu_span'][output['q2clause'][question]['clause_id']][top_edu_id]
        top_edu_start = max(output['q2clause'][question]['clause_start'], top_edu_span[0])
        top_edu_end = min(output['q2clause'][question]['clause_end'], top_edu_span[1])
        output['q2edu'][question]['top_edu_id'] = top_edu_id
        output['q2edu'][question]['top_edu_start'] = top_edu_start
        output['q2edu'][question]['top_edu_end'] = top_edu_end  # [s,e] both inclusive
    return output


def highlight_spans(highlight, rule_text):
    match = []
    ptr = 0
    for i, word in enumerate(rule_text):
        if ptr >= len(highlight):
            break
        if re.sub(r'\W+', '', word) == re.sub(r'\W+', '', highlight[ptr]):
            # if word == highlight[ptr]:
            match.append(i)
            ptr += 1
    ranges = []
    for k, g in groupby(enumerate(match), lambda x: x[0] - x[1]):
        group = list(map(itemgetter(1), g))
        ranges.append((group[0], group[-1]))
    return ranges

def io_encode(gold_explain, snippet):
    gold_encoding = [0] * len(snippet)
    for gold in gold_explain:
        for fragment in gold:
            gold_enc = roberta_encode(fragment.strip())
            for i in range(0, len(gold_encoding)):
                if snippet[i:i+len(gold_enc)] == gold_enc:
                    for j in range(i, i+len(gold_enc)):
                        gold_encoding[j] = 1
                    break
    return gold_encoding

def io_decode(encoding, snippet):
    # snippet = tokenizer.decode(snippet, clean_up_tokenization_spaces=False).strip('\n').strip()
    # snippet_split = snippet.split()
    snip_list = []
    for idx, token in enumerate(snippet):
        if encoding[idx] == 1:
            snip_list.append(token)
    return roberta_decode(snip_list)


if __name__ == '__main__':
    sharc_path = './data'
    with open(os.path.join(sharc_path, 'sharc_raw', 'negative_sample_utterance_ids',
                           'sharc_negative_question_utterance_ids.txt')) as f:
        negative_question_ids = f.read().splitlines()
    with open(os.path.join(sharc_path, 'explanations.json')) as f:
        explanation_data = json.load(f)
    for split in ['dev', 'train']:
        fsplit = 'sharc_train' if split == 'train' else 'sharc_dev'
        with open(os.path.join(sharc_path, 'sharc_raw/json/{}_question_fixed.json'.format(fsplit))) as f:
        # with open('counterfactual_samples.json') as f:
            data_raw = json.load(f)
        with open(os.path.join(sharc_path, '{}_snippet_parsed.json'.format(split))) as f:
            edu_segment = json.load(f)

        ########################
        # construct tree mappings
        ########################
        ftree = os.path.join(sharc_path, 'trees_mapping_{}_{}.json'.format(FILENAME, split))
        # if not os.path.isfile(ftree) or FORCE:
        if True:
            tasks = {}
            for ex in data_raw:
                if ex['tree_id'] in tasks:
                    task = tasks[ex['tree_id']]
                else:
                    task = tasks[ex['tree_id']] = {'snippet': ex['snippet'], 'questions': set(), 'scenarios': set(),
                                                   'initial_questions': set()}
                for h in ex['history'] + ex['evidence']:
                    task['questions'].add(h['follow_up_question'])
                if ex['answer'].lower() not in {'yes', 'no', 'irrelevant'}:
                    task['questions'].add(ex['answer'])
                if ex['scenario'] != '':
                    task['scenarios'].add(ex['scenario'])
                task['initial_questions'].add(ex['question'])
            keys = sorted(list(tasks.keys()))
            vals = [extract_edus(tasks[k], edu_segment[k]) for k in tqdm(keys)]
            mapping = {k: v for k, v in zip(keys, vals)}
            with open(ftree, 'wt') as f:
                json.dump(mapping, f, indent=2)
        else:
            with open(ftree) as f:
                mapping = json.load(f)

        ########################
        # construct samples
        ########################
        fproc = os.path.join(sharc_path, 'proc_decision_{}_{}.pt'.format(FILENAME, split))
        data = []
        decs = set()
        for ex in tqdm(data_raw):
            # hot fix
            if ex['answer'] == 'Yes?':
                ex['answer'] = 'Yes'

            question_to_explanations = explanation_data[ex['tree_id']]['questions']

            # scenario explanation
            scenario = []
            last_q = []
            explanation_clauses = []
            follow_up_question_list = []
            if ex['answer'] == 'Yes':
                turns = ex['evidence'] + ex['history']
                for turn in turns:
                    follow_up_question = turn['follow_up_question']
                    follow_up_answer = turn['follow_up_answer']
                    follow_up_question_list.append(follow_up_question)
                    if follow_up_answer == 'No':
                        explanation_clauses = [question_to_explanations[follow_up_question]]
                        break
                    else:
                        explanation_clauses.append(question_to_explanations[follow_up_question])
            elif ex['answer'] == 'No':
                turns = ex['evidence'] + ex['history']
                for turn in turns:
                    follow_up_question = turn['follow_up_question']
                    follow_up_answer = turn['follow_up_answer']
                    follow_up_question_list.append(follow_up_question)
                    if follow_up_answer == 'Yes':
                        explanation_clauses = [question_to_explanations[follow_up_question]]
                        break
                    else:
                        explanation_clauses.append(question_to_explanations[follow_up_question])
            elif ex['answer'] != 'Irrelevant':
                explanation_clauses.append(question_to_explanations[ex['answer']])

            if len(explanation_clauses) > 1:
                assert ex['answer'] == 'Yes' or ex['answer'] == 'No' or ex['answer'] == 'Irrelevant'
                conditionals = explanation_data[ex['tree_id']]['conditionals']
                for question in follow_up_question_list:
                    if question in conditionals:
                        explanation_clauses.append(conditionals[question])

            m = mapping[ex['tree_id']]
            sep = tokenizer.sep_token_id
            cls = tokenizer.cls_token_id
            pad = tokenizer.pad_token_id

            # snippet
            inp = []
            edu_mask = []
            # Gold Encoding
            snippet = ex['snippet'].replace('\n', ' ')
            inp = [cls] + roberta_encode(snippet) + [sep]
            explanation_gold_ex = io_encode(explanation_clauses, inp)
            edu_mask = [0] + ([1] * (len(inp) - 2)) + [0]
            assert len(explanation_gold_ex) == len(inp)
            assert len(edu_mask) == len(inp)

            # user info (scenario, dialog history)
            user_idx = []
            question_tokenized = m['initial_question_t'][ex['question']]
            if len(inp) < MAX_LEN: user_idx.append(len(inp))
            question_idx = len(inp)
            inp += [cls] + question_tokenized + [sep]
            scenario_idx = -1
            if ex['scenario'] != '':
                scenario_tokenized = roberta_encode(ex['scenario'])
                if len(inp) < MAX_LEN: user_idx.append(len(inp))
                scenario_idx = len(inp)
                inp += [cls] + scenario_tokenized + [sep]
            for fqa in ex['history']:
                if len(inp) < MAX_LEN: user_idx.append(len(inp))
                fq, fa = fqa['follow_up_question'], fqa['follow_up_answer']
                fa = 'No' if 'no' in fa.lower() else 'Yes'  # fix possible typos like 'noe'
                inp += [cls] + roberta_encode('question') + m['q_t'][fq] + roberta_encode('answer') + roberta_encode(
                    fa) + [sep]

            # all
            input_mask = [1] * len(inp)
            edu_mask += [0] * (len(input_mask) - len(edu_mask))
            explanation_gold_ex += [0] * (len(input_mask) - len(explanation_gold_ex))
            assert len(inp) == len(input_mask)
            assert len(edu_mask) == len(input_mask)
            assert len(explanation_gold_ex) == len(input_mask)

            if len(inp) > MAX_LEN:
                inp = inp[:MAX_LEN]
                input_mask = input_mask[:MAX_LEN]
                edu_mask = edu_mask[:MAX_LEN]
                explanation_gold_ex = explanation_gold_ex[:MAX_LEN]
            while len(inp) < MAX_LEN:
                inp.append(pad)
                explanation_gold_ex.append(0)
                input_mask.append(0)
                edu_mask.append(0)
            assert len(inp) == len(input_mask)
            assert len(edu_mask) == len(input_mask)
            assert len(explanation_gold_ex) == len(input_mask)
            ex['entail'] = {
                'inp': inp,
                'input_ids': torch.LongTensor(inp),
                'input_mask': torch.LongTensor(input_mask),
                'edu_mask': torch.LongTensor(edu_mask),
                'user_idx': torch.LongTensor(user_idx),
                'question_idx': question_idx,
                'scenario_idx': scenario_idx,
            }

            ########################
            # logic reasoning
            ########################

            explain_decoding = io_decode(explanation_gold_ex, inp)
            for rule in explanation_clauses:
                for fragment in rule:
                    if fragment not in explain_decoding:
                        print("tree id:", ex["tree_id"])
                        print("Fragment:", fragment)
                        print("Decoding:", explain_decoding)
                    assert fragment in explain_decoding
            ex['logic'] = {}
            ex_answer = ex['answer'].lower()
            ex['logic']['answer_class'] = DECISION_CLASSES.index(
                ex_answer) if ex_answer in DECISION_CLASSES else DECISION_CLASSES.index('more')
            ex['logic']['explain_io'] = torch.LongTensor(explanation_gold_ex)
            ex['logic']['explain'] = explain_decoding
            data.append(ex)
        torch.save(data, fproc)
