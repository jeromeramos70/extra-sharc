import os
import shutil
import torch
import logging
import importlib
import numpy as np
import json
from tqdm import trange
from pprint import pformat
from collections import defaultdict
from torch import nn
from torch.nn import functional as F
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig, AdamW, get_linear_schedule_with_warmup
from argparse import Namespace
from torch.nn.init import xavier_uniform_
from sklearn.metrics import f1_score, accuracy_score
DECISION_CLASSES = ['yes', 'no', 'more', 'irrelevant']


class Module(nn.Module):

    def __init__(self, args, device='cpu'):
        super().__init__()
        self.args = args
        self.device = device
        self.epoch = 0
        self.dropout = nn.Dropout(self.args.dropout)

        roberta_model_path = args.pretrained_lm_path
        roberta_config = RobertaConfig.from_pretrained(roberta_model_path, cache_dir=None)
        self.tokenizer = RobertaTokenizer.from_pretrained(roberta_model_path, cache_dir=None)
        self.roberta = RobertaModel.from_pretrained(roberta_model_path, cache_dir=None, config=roberta_config)

        self.ex_scorer = nn.Linear(self.args.bert_hidden_size, 2)

        # clf scorer
        self.inp_attn_scorer = nn.Linear(self.args.bert_hidden_size, 1)
        self.class_clf = nn.Linear(self.args.bert_hidden_size, 4)
        self.w_output = nn.Linear(self.args.bert_hidden_size, 4, bias=True)

    def _reset_transformer_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for name, param in self.named_parameters():
            if 'transformer' in name and param.dim() > 1:
                xavier_uniform_(param)

    @classmethod
    def load_module(cls, name):
        return importlib.import_module('model.{}'.format(name)).Module

    @classmethod
    def load(cls, fname, override_args=None):
        load = torch.load(fname, map_location=lambda storage, loc: storage)
        args = vars(load['args'])
        if override_args:
            args.update(override_args)
        args = Namespace(**args)
        model = cls.load_module(args.model)(args)
        model.load_state_dict(load['state'])
        return model

    def save(self, metrics, dsave, early_stop):
        files = [os.path.join(dsave, f) for f in os.listdir(dsave) if f.endswith('.pt') and f != 'best.pt']
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        if len(files) > self.args.keep-1:
            for f in files[self.args.keep-1:]:
                os.remove(f)

        fsave = os.path.join(dsave, 'step{}-{}.pt'.format(metrics['global_step'], metrics[early_stop]))
        torch.save({
            'args': self.args,
            'state': self.state_dict(),  # comment to save space
            'metrics': metrics,
        }, fsave)
        fbest = os.path.join(dsave, 'best.pt')
        if os.path.isfile(fbest):
            os.remove(fbest)
        shutil.copy(fsave, fbest)

    def create_input_tensors(self, batch):
        feat = {
            k: torch.stack([e['entail'][k] for e in batch], dim=0).to(self.device) for k in ['input_ids', 'input_mask', 'edu_mask']
        }
        return feat

    def logic_op(self, input, input_mask):
        selfattn_unmask = self.w_selfattn(self.dropout(input))
        selfattn_unmask = selfattn_unmask.squeeze()
        input_mask = input_mask.bool()
        selfattn_unmask.masked_fill_(~input_mask, -float('inf'))
        selfattn_weight = F.softmax(selfattn_unmask, dim=1)
        score = self.w_output(self.dropout(selfattn_weight))
        return score

    def forward(self, batch):
        out = self.create_input_tensors(batch)
        out['roberta_enc'] = roberta_enc = self.roberta(input_ids=out['input_ids'], attention_mask=out['input_mask'])[0]
        explain_scores = self.ex_scorer(self.dropout(roberta_enc))
        out['explain_scores'] = F.softmax(self.mask_scores(explain_scores, out['edu_mask']), dim=2)

        # clf classifier
        inp_attn_score = self.inp_attn_scorer(self.dropout(out['roberta_enc'])).squeeze(2) - (1-out['input_mask'].float()).mul(1e20)
        inp_attn = F.softmax(inp_attn_score, dim=1).unsqueeze(2).expand_as(out['roberta_enc']).mul(self.dropout(out['roberta_enc'])).sum(1)
        out['clf_score'] = self.class_clf(self.dropout(inp_attn))
        return out

    def mask_scores(self, scores, mask):
        mask = torch.stack((mask, mask), dim=2)
        return scores * mask

    def io_decode(self, input, encoding):
        decode_list = []
        for idx, token in enumerate(input):
            if encoding[idx] == 1:
                decode_list.append(token)
        decoded = self.tokenizer.decode(decode_list, clean_up_tokenization_spaces=False).strip('\n').strip()
        return decoded

    def extract_preds(self, out, batch):
        preds = []
        clf = out['clf_score'].max(1)[1].tolist()
        pred_explain = out['explain_scores']
        for idx, ex in enumerate(batch):
            clf_i = clf[idx]
            pred_explain_i = torch.argmax(pred_explain[idx], dim=1)
            explain_decoded = self.io_decode(ex['entail']['inp'], pred_explain_i)
            a = DECISION_CLASSES[clf_i]
            preds.append({
                'utterance_id': ex['utterance_id'],
                'pred_answer': a,
                'pred_answer_cls': clf_i,
                'pred_explain_encoded': pred_explain_i.tolist(),
                'pred_explain': explain_decoded,
                'gold_explain': ex['logic']['explain'],
            })
        return preds

    def compute_loss(self, out, batch):
        gclf = torch.tensor([ex['logic']['answer_class'] for ex in batch], device=self.device, dtype=torch.long)
        g_ex = torch.stack([ex['logic']['explain_io']for ex in batch]).to(self.device).to(torch.float)
        ex_score = out['explain_scores'][:, :, 1]
        edu_masks = [i['entail']['edu_mask'] for i in batch]
        loss = {
            'clf': F.cross_entropy(out['clf_score'], gclf),
            'ex': sum(F.binary_cross_entropy(i[mask == 1], j[mask == 1]) for i, j, mask in zip(ex_score, g_ex, edu_masks))
        }
        return loss

    def compute_metrics(self, predictions, data):
        from sklearn.metrics import accuracy_score, confusion_matrix
        metrics = {}
        preds = [pred['pred_answer_cls'] for pred in predictions]
        golds = [gold['logic']['answer_class'] for gold in data]
        edu_masks = [np.nonzero(i['entail']['edu_mask'])[-1][0].item() for i in data]

        pred_explain = [pred['pred_explain_encoded'] for pred in predictions]
        golds_explain = [gold['logic']['explain_io'] for gold in data]
        em_yes, em_no, em_more, em_irrelevant = 0.0, 0.0, 0.0, 0.0
        total_yes, total_no, total_more, total_irrelevant = 0.0, 0.0, 0.0, 0.0
        accuracy_score_yes, accuracy_score_no, accuracy_score_more, accuracy_score_irrelevant = [], [], [], []
        for gold, pred, mask, decision_type in zip(golds_explain, pred_explain, edu_masks, golds):
            gold = gold.tolist()
            if decision_type == 0:
                accuracy_score_yes.append(f1_score(gold[:mask + 1], pred[:mask + 1]))
                total_yes += 1
                if gold[:mask + 1] == pred[:mask + 1]:
                    em_yes += 1
            if decision_type == 1:
                accuracy_score_no.append(f1_score(gold[:mask + 1], pred[:mask + 1]))
                total_no += 1
                if gold[:mask + 1] == pred[:mask + 1]:
                    em_no += 1
            if decision_type == 2:
                accuracy_score_more.append(f1_score(gold[:mask + 1], pred[:mask + 1]))
                total_more += 1
                if gold[:mask + 1] == pred[:mask + 1]:
                    em_more += 1
            elif decision_type == 3:
                accuracy_score_irrelevant.append(f1_score(gold[:mask + 1], pred[:mask + 1]))
                total_irrelevant += 1
                if gold[:mask + 1] == pred[:mask + 1]:
                    em_irrelevant += 1
        f1_tot = accuracy_score_yes + accuracy_score_no + accuracy_score_more + accuracy_score_irrelevant
        print(gold)
        print(preds)
        micro_accuracy = accuracy_score(golds, preds)
        metrics["0c_micro_accuracy"] = float("{0:.2f}".format(micro_accuracy * 100))
        conf_mat = confusion_matrix(golds, preds, labels=[0, 1, 2, 3])
        conf_mat_norm = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
        macro_accuracy = np.mean([conf_mat_norm[i][i] for i in range(conf_mat.shape[0])])
        metrics["0b_macro_accuracy"] = float("{0:.2f}".format(macro_accuracy * 100))
        metrics["0a_combined"] = float("{0:.2f}".format(macro_accuracy * micro_accuracy * 100))
        metrics["0d_confmat"] = conf_mat.tolist()
        metrics["0f_last_q_explain_em_yes"] = float("{0:.2f}".format((em_yes/total_yes) * 100))
        metrics["0g_last_q_explain_em_no"] = float("{0:.2f}".format((em_no/total_no) * 100))
        metrics["0h_last_q_explain_em_more"] = float("{0:.2f}".format((em_more/total_more) * 100))
        metrics["em_total"] = float("{0:.2f}".format((em_yes + em_no + em_more + em_irrelevant)/len(golds) * 100))
        metrics["f1_total"] = float("{0:.2f}".format(np.mean(f1_tot) * 100))
        metrics['em_stuff'] = [em_yes, em_no, em_more, em_irrelevant]
        return metrics

    def run_pred(self, dev):
        preds = []
        self.eval()
        for i in trange(0, len(dev), self.args.dev_batch, desc='batch', disable=self.args.tqdm_bar):
            batch = dev[i:i+self.args.dev_batch]
            out = self(batch)
            preds += self.extract_preds(out, batch)
        return preds

    def run_train(self, train, dev):
        if not os.path.isdir(self.args.dsave):
            os.makedirs(self.args.dsave)

        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(self.args.dsave, 'train.log'))
        fh.setLevel(logging.CRITICAL)
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.CRITICAL)
        logger.addHandler(ch)

        num_train_steps = int(len(train) / self.args.train_batch * self.args.epoch)
        num_warmup_steps = int(self.args.warmup * num_train_steps)

        # remove pooler
        param_optimizer = list(self.named_parameters())
        param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, correct_bias=True)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)  # PyTorch scheduler

        print('num_train', len(train))
        print('num_dev', len(dev))

        global_step = 0
        best_metrics = {self.args.early_stop: -float('inf')}
        for epoch in trange(self.args.epoch, desc='epoch',):
            self.epoch = epoch
            train = train[:]
            np.random.shuffle(train)

            train_stats = defaultdict(list)
            preds = []
            self.train()
            for i in trange(0, len(train), self.args.train_batch, desc='batch'):
                actual_train_batch = int(self.args.train_batch / self.args.gradient_accumulation_steps)
                batch_stats = defaultdict(list)
                batch = train[i: i + self.args.train_batch]

                for accu_i in range(0, len(batch), actual_train_batch):
                    actual_batch = batch[accu_i : accu_i + actual_train_batch]
                    out = self(actual_batch)
                    pred = self.extract_preds(out, actual_batch)
                    loss = self.compute_loss(out, actual_batch)

                    for k, v in loss.items():
                        loss[k] = v / self.args.gradient_accumulation_steps
                        batch_stats[k].append(v.item()/ self.args.gradient_accumulation_steps)
                    sum(loss.values()).backward()
                    preds += pred

                optimizer.step()
                scheduler.step()

                optimizer.zero_grad()
                global_step += 1

                for k in batch_stats.keys():
                    train_stats['loss_' + k].append(sum(batch_stats[k]))

                if global_step % self.args.eval_every_steps == 0:
                    dev_stats = defaultdict(list)
                    dev_preds = self.run_pred(dev)
                    dev_metrics = {k: sum(v) / len(v) for k, v in dev_stats.items()}
                    dev_metrics.update(self.compute_metrics(dev_preds, dev))
                    metrics = {'global_step': global_step}
                    metrics.update({'dev_' + k: v for k, v in dev_metrics.items()})
                    logger.critical(pformat(metrics))
                    if metrics[self.args.early_stop] > best_metrics[self.args.early_stop]:
                        logger.critical('Found new best! Saving to ' + self.args.dsave)
                        best_metrics = metrics
                        self.save(best_metrics, self.args.dsave, self.args.early_stop)
                        with open(os.path.join(self.args.dsave, 'dev.preds.json'), 'wt') as f:
                            json.dump(dev_preds, f, indent=2)
                        with open(os.path.join(self.args.dsave, 'dev.best_metrics.json'), 'wt') as f:
                            json.dump(best_metrics, f, indent=2)

                    self.train()

            train_metrics = {k: sum(v) / len(v) for k, v in train_stats.items()}
            train_metrics.update(self.compute_metrics(preds, train))

            dev_stats = defaultdict(list)
            dev_preds = self.run_pred(dev)
            dev_metrics = {k: sum(v) / len(v) for k, v in dev_stats.items()}
            dev_metrics.update(self.compute_metrics(dev_preds, dev))
            metrics = {'global_step': global_step}
            metrics.update({'train_' + k: v for k, v in train_metrics.items()})
            metrics.update({'dev_' + k: v for k, v in dev_metrics.items()})
            logger.critical(pformat(metrics))

            if metrics[self.args.early_stop] > best_metrics[self.args.early_stop]:
                logger.critical('Found new best! Saving to ' + self.args.dsave)
                best_metrics = metrics
                self.save(best_metrics, self.args.dsave, self.args.early_stop)
                with open(os.path.join(self.args.dsave, 'dev.preds.json'), 'wt') as f:
                    json.dump(dev_preds, f, indent=2)
                with open(os.path.join(self.args.dsave, 'dev.best_metrics.json'), 'wt') as f:
                    json.dump(best_metrics, f, indent=2)

        logger.critical('Best dev')
        logger.critical(pformat(best_metrics))
