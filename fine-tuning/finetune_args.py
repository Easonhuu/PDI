import argparse
import time
import json
import os
# input_file:
# 'PDB-ID', 'Affinity-Value', 'seq', 'rdkit_smiles', 'set', 'contact_map'
# 11gs, 5.82, PYTVVY..., CC[C@H](C(=O)...,CC[C@@H]..., train, [array([[ True,  True,  True, ..., False, False, False], ..., [False, False, False, ...,  True,  True,  True]])]
# ...
# input_file = "/data/eason/pdbbind/pdbbind2019.pkl"
# input_file = "/data/eason/pdbbind/pdbbind2016.pkl"
input_file = "/data/eason/pdbbind/pdbbind2019_small.pkl"
model_type = 'bert'
p_dropout = 0.4
multi_dropout_num = 5
epochs = 200
weight_decay = 4  # known as l2_regularization_lambda
learning_rate = 3  # the index of learning rate
adam_epsilon = 1e-8
warmup_steps = 0
patience = 20  # Number of epochs with no improvement after which learning rate will be reduced.
TASK = 'Affinity-Value'  # target to predict
SMILES = "rdkit_smiles"  # drug SMILES to input
max_seq_len = 512
n_gpu = 2  # total number of the GPUs to be used
gpu_start = 2  # the first GPU ID in used continuous GPUs
per_gpu_batch_size = 32
batch_size = per_gpu_batch_size * n_gpu
pretrain_model_dict = "./fine-tuning/pretrained_model/pytorch_model.pth"
config_name = "./fine-tuning/pretrained_model/modified_bert_config.json"
vocab_path = "./fine-tuning/pretrained_model/vocab.json"
output_dir = "./fine-tuning/outputs"
cache_dir = None
prefix_filename = os.path.splitext(os.path.split(input_file)[-1])[0]
start_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
log_dir = "./fine-tuning/log"
log_file = os.path.join(log_dir, prefix_filename + '_' + model_type + '_' + start_time + '.txt')
seed = 24
early_stop = 10
early_stop_scale = 0.005
gradient_accumulation_steps = 1
max_grad_norm = 1.0

# GAT parameters
radius = 3  # the iteration num of atom embedding
T = 1  # the iteration num of molecule embedding
drug_dim = 128
pro_struc_dim = 64
pro_seq_dim = 512
pro_dim = (pro_struc_dim + pro_seq_dim) // 2
num_atom_features = 39
num_bond_features = 10

with open(vocab_path) as f:
    vocab = json.load(f)

special_vocab_size = 5
nonspecial_vocab_size = len(vocab) - special_vocab_size  # the size of vocab without special tokens including "[PAD][CLS][SEP][MASK][UNK]"

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", default=input_file, type=str, help="")
parser.add_argument("--model_type", default=model_type, type=str, help="")
parser.add_argument("--p_dropout", default=p_dropout, type=float, help="")
parser.add_argument("--multi_dropout_num", default=multi_dropout_num, type=float, help="")
parser.add_argument("--epochs", default=epochs, type=int, help="")
parser.add_argument("--weight_decay", default=weight_decay, type=int, help="")
parser.add_argument("--learning_rate", default=learning_rate, type=int, help="index of learning rate")
parser.add_argument("--adam_epsilon", default=adam_epsilon, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--warmup_steps", default=warmup_steps, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--patience", default=patience, type=int, help="")
parser.add_argument("--TASK", default=TASK, type=str, help="transform long str to variable")
parser.add_argument("--SMILES", default=SMILES, type=str, help="transform long str to variable")
parser.add_argument("--max_seq_len", default=max_seq_len, type=int, help="max length of model sequence")
parser.add_argument("--n_gpu", default=n_gpu, type=int, help="total number of the GPUs to be used")
parser.add_argument("--gpu_start", default=gpu_start, type=int, help="first number of used gpus")
parser.add_argument("--per_gpu_batch_size", default=per_gpu_batch_size, type=int, help="")
parser.add_argument("--batch_size", default=batch_size, type=int, help="")
parser.add_argument("--pretrain_model_dict", default=pretrain_model_dict, type=str, help="the parameters of pretrained model")
parser.add_argument("--config_name", default=config_name, type=str, help="masked lm pretrained model config")
parser.add_argument("--vocab_path", default=vocab_path, type=str, help="vocab path")
parser.add_argument("--output_dir", default=output_dir, type=str)
parser.add_argument("--cache_dir", default=cache_dir, type=str)
parser.add_argument("--log_file", default=log_file, type=str, help="")
parser.add_argument("--seed", default=seed, type=int, help="")
parser.add_argument("--early_stop", default=early_stop, type=int, help="")
parser.add_argument("--early_stop_scale", default=early_stop_scale, type=float, help="")
parser.add_argument("--gradient_accumulation_steps", default=gradient_accumulation_steps, type=int, help="")
parser.add_argument("--max_grad_norm", default=max_grad_norm, type=float, help="Max gradient norm.")
parser.add_argument("--radius", default=radius, type=int, help="")
parser.add_argument("--T", default=T, type=int, help="")
parser.add_argument("--drug_dim", default=drug_dim, type=int, help="molecular dimension by GAT")
parser.add_argument("--pro_struc_dim", default=pro_struc_dim, type=int, help="protein structure embedding dimension by GAT")
parser.add_argument("--pro_seq_dim", default=pro_seq_dim, type=int, help="protein sequence embedding dimension by Transformer")
parser.add_argument("--pro_dim", default=pro_dim, type=int, help="protein embedding dimension by Transformer")
parser.add_argument("--num_atom_features", default=num_atom_features, type=int, help="")
parser.add_argument("--num_bond_features", default=num_bond_features, type=int, help="")
parser.add_argument("--vocab", default=vocab, type=list, help="")
parser.add_argument("--special_vocab_size", default=special_vocab_size, type=int, help="vocabulary size with special tokens")
parser.add_argument("--nonspecial_vocab_size", default=nonspecial_vocab_size, type=int, help="vocabulary size without special tokens")

args = parser.parse_args()