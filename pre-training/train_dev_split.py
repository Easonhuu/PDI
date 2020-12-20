# import ptvsd
# ptvsd.enable_attach(address = ('0.0.0.0', 5678))
# ptvsd.wait_for_attach()

from sklearn.model_selection import train_test_split
from pretraining_args import args
import os
with open(args.pretrain_data_path, encoding='utf-8',) as f:
    fcsv = f.readlines()
    lines = []
    for i, line in enumerate(fcsv):
        lines.append(line)
print("sample num: %d", len(lines))

all_len = []
for line in lines:
    sequence_list = line.strip().split(' ')
    all_len.append(len(sequence_list))
print("max sequence length is %d, min sequence length is %d." % (max(all_len), min(all_len)))

train_lines, dev_lines = train_test_split(lines, test_size = 0.2, random_state = 124)

train_path = "." + args.pretrain_data_path.split('.')[1] + "-train.txt"
dev_path = "." + args.pretrain_data_path.split('.')[1] + "-dev.txt"

with open(train_path, "w", encoding='utf-8') as f:   
    for i, line in enumerate(train_lines):
        f.write(line)
with open(dev_path, "w", encoding='utf-8') as f:   
    for i, line in enumerate(dev_lines):
        f.write(line)

