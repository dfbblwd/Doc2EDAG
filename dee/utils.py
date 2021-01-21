# -*- coding: utf-8 -*-
# AUTHOR: Shun Zheng
# DATE: 19-9-19

import json
import jieba
import jieba.posseg as pseg
jieba.enable_paddle()
import logging
import pickle
from pytorch_pretrained_bert import BertTokenizer


logger = logging.getLogger(__name__)

EPS = 1e-10


def default_load_json(json_file_path, encoding='utf-8', **kwargs):
    with open(json_file_path, 'r', encoding=encoding) as fin:
        tmp_json = json.load(fin, **kwargs)
    return tmp_json


def default_dump_json(obj, json_file_path, encoding='utf-8', ensure_ascii=False, indent=2, **kwargs):
    with open(json_file_path, 'w', encoding=encoding) as fout:
        json.dump(obj, fout,
                  ensure_ascii=ensure_ascii,
                  indent=indent,
                  **kwargs)


def default_load_pkl(pkl_file_path, **kwargs):
    with open(pkl_file_path, 'rb') as fin:
        obj = pickle.load(fin, **kwargs)

    return obj


def default_dump_pkl(obj, pkl_file_path, **kwargs):
    with open(pkl_file_path, 'wb') as fout:
        pickle.dump(obj, fout, **kwargs)


def set_basic_log_config():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO)


class BERTChineseCharacterTokenizer(BertTokenizer):
    """Customized tokenizer for Chinese financial announcements"""

    def __init__(self, vocab_file, do_lower_case=True):
        super(BERTChineseCharacterTokenizer, self).__init__(vocab_file, do_lower_case)

    def char_tokenize(self, text, unk_token='[UNK]'):
        """perform pure character-based tokenization"""
        tokens = list(text)
        out_tokens = []
        for token in tokens:
            if token in self.vocab:
                out_tokens.append(token)
            else:
                out_tokens.append(unk_token)

        return out_tokens


def recursive_print_grad_fn(grad_fn, prefix='', depth=0, max_depth=50):
    if depth > max_depth:
        return
    print(prefix, depth, grad_fn.__class__.__name__)
    if hasattr(grad_fn, 'next_functions'):
        for nf in grad_fn.next_functions:
            ngfn = nf[0]
            recursive_print_grad_fn(ngfn, prefix=prefix + '  ', depth=depth+1, max_depth=max_depth)


def strtobool(str_val):
    """Convert a string representation of truth to true (1) or false (0).

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.  Raises ValueError if
    'val' is anything else.
    """
    str_val = str_val.lower()
    if str_val in ('y', 'yes', 't', 'true', 'on', '1'):
        return True
    elif str_val in ('n', 'no', 'f', 'false', 'off', '0'):
        return False
    else:
        raise ValueError("invalid truth value %r" % (str_val,))

import copy

def main():
    """

    """
    notice_id = ''

    out_file = open('../Data/train_label_puddle.txt', 'w+')
    temp_json = default_load_json("../Data/train.json")
    for notice in temp_json:
        notice_id = notice[0]
        notice_sentence = notice[1]['sentences']
        notice_avm = notice[1]['ann_valid_mspans']
        notice_avd = notice[1]['ann_valid_dranges']
        notice_m2d = notice[1]['ann_mspan2dranges']
        notice_m2g = notice[1]['ann_mspan2guess_field']
        notice_ree = notice[1]['recguid_eventname_eventdict_list']

        # 知识标注
        notice_sentence_label = []
        for i in range(len(notice_sentence)):
            sentence_label = ['O'] * len(notice_sentence[i])
            notice_sentence_label.append(sentence_label)

        for k in notice_m2d:
            v = notice_m2d[k]
            label = notice_m2g[k]
            for rg in v:
                sentence_index = rg[0]
                begin_pos = rg[1]
                end_pos = rg[2] - 1
                notice_sentence_label[sentence_index][begin_pos] = 'b-' + label
                notice_sentence_label[sentence_index][end_pos] = 'e-' + label
                notice_sentence_label[sentence_index][begin_pos + 1:end_pos] = ['m-' + label] * (end_pos - begin_pos - 1)

        # 分词词性标注
        notice_sentence_pos = []
        for i in range(len(notice_sentence)):
            sentence_label = ['O'] * len(notice_sentence[i])
            notice_sentence_pos.append(sentence_label)
        word_list = []
        pos_list = []
        for sentence_index, sentence in enumerate(notice_sentence):
            words = pseg.cut(sentence, use_paddle=True)
            word_list = []
            pos_list = []
            begin_pos = 0
            end_pos = 0
            cur_pos = 0
            for word, pos in words:
                word_list.append(word)
                pos_list.append(pos)
                begin_pos = cur_pos
                end_pos = cur_pos + len(word) - 1
                cur_pos += len(word)
                if len(word) > 1:
                    notice_sentence_pos[sentence_index][begin_pos] = 'b-' + pos
                    notice_sentence_pos[sentence_index][end_pos] = 'e-' + pos
                    notice_sentence_pos[sentence_index][begin_pos + 1:end_pos] = ['m-' + pos] * (
                                end_pos - begin_pos - 1)
                else:
                    notice_sentence_pos[sentence_index][begin_pos] = 'i-' + pos



        for event in notice_ree:
            event_info = event[2]
            pass

        # 句子实体标注输出
        for i, sentence in enumerate(zip(notice_sentence_label, notice_sentence_pos)):
            for j, c in enumerate(zip(notice_sentence_label[i], notice_sentence_pos[i])):
                unit = notice_sentence[i][j] + ' ' + notice_sentence_pos[i][j] + ' ' + notice_sentence_label[i][j]
                out_file.write(unit + '\n')
            out_file.write('#*#' + ' ' + 'sen_split' + ' ' + 'sen_split' + '\n')
        out_file.write('#****#' + ' ' + 'notice_split' + ' ' + 'notice_split' + '\n')

    out_file.close()

    return notice_id

if __name__ == "__main__":
    tmp_json = main()