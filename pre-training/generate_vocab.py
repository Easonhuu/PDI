# import ptvsd
# ptvsd.enable_attach(address = ('0.0.0.0', 5678))
# ptvsd.wait_for_attach()

from pretraining_args import args
import os

with open(args.pretrain_data_path, encoding='utf-8') as f:
    lines = f.readlines()
    word_list = []
    for line in lines:
        word_list.extend(line.strip().split(' '))
    word_set = set(word_list)  # without 'J'

special_tokens = ['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
with open(args.tokenizer_name, 'w', encoding='utf-8') as f:
    for token in special_tokens:
        f.write(token+'\n')
    for word in word_set:
        f.write(word+'\n')