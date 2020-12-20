import argparse
import torch

"""
corpus-enz1k is used to debug
"""
# corpus_name = "corpus-enz1k"
corpus_name = "corpus-enz20w"
# corpus_name = "corpus-enz416w"
pretrain_data_path = "./data/" + corpus_name + ".txt"
train_data_file = "./data/" + corpus_name + "-train.txt"
eval_data_file = "./data/" + corpus_name + "-dev.txt"

output_dir = "./pre-training/outputs/"
config_name = "./pre-training/modified_bert_config.json"
# config_name = "bert-large-uncased"
tokenizer_name = "./pre-training/data/protein_vocab.txt"
cache_dir = "./pre-trained_model/"
protein_vocab_path = "./pre-training/data/protein_vocab.json"
pretrain_log = "./pre-training/log/"
# config_name = None
# tokenizer_name = None
# model_type = "xlnet"
# model_name_or_path = "xlnet-large-cased"
# model_type = "roberta"
# model_name_or_path = "roberta-large"
model_type = "bert"
# model_name_or_path = "bert-large-uncased"
# model_type = "albert"
# model_name_or_path = "albert-xxlarge-v2"
model_name_or_path = None
do_train = True
do_eval = True
do_test = True
overwrite_output_dir = True
overwrite_cache = False
evaluate_during_training = False
each_epoch_eval = True
each_batch_eval = False
each_checkpoint_eval = False
mlm = True
mlm_probability = 0.15
max_seq_len = 512
per_gpu_train_batch_size = 16
per_gpu_eval_batch_size = 16
learning_rate = 1e-4
warmup_proportion = 0.1
warmup_steps = 0
num_train_epochs = 100
gpu_start = 0
gpu_num = 2
gradient_accumulation_steps = 1
weight_decay = 0.0
adam_epsilon = 1e-8
max_grad_norm = 1.0
no_cuda = False
seed = 42
seed_flag = True
local_rank = -1
early_stop = 10
early_stop_scale = 0.001

parser = argparse.ArgumentParser()
# Required parameters
parser.add_argument("--corpus_name", default=corpus_name, type=str)
parser.add_argument("--pretrain_data_path", default=pretrain_data_path, type=str)
parser.add_argument("--train_data_file", default=train_data_file, type=str)
parser.add_argument("--eval_data_file", default=eval_data_file, type=str)
parser.add_argument("--output_dir", type=str, default=output_dir, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--config_name", default=config_name, type=str, 
    help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.")
parser.add_argument("--tokenizer_name", default=tokenizer_name, type=str,
    help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.")
parser.add_argument("--cache_dir", default=cache_dir, type=str, 
    help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)")
parser.add_argument("--protein_vocab_path", default=protein_vocab_path, type=str)
parser.add_argument("--pretrain_log", default=pretrain_log, type=str)
parser.add_argument("--model_type", type=str, default=model_type)
parser.add_argument("--model_name_or_path", default=model_name_or_path, type=str, 
    help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.")
parser.add_argument("--do_train", default=do_train, help="Whether to run training.")
parser.add_argument("--do_eval", default=do_eval, help="Whether to run eval on the dev set.")
parser.add_argument("--overwrite_output_dir", default=overwrite_output_dir, help="Overwrite the content of the output directory")
parser.add_argument("--overwrite_cache", default=overwrite_cache, help="Overwrite the cached training and evaluation sets")
parser.add_argument("--evaluate_during_training", default=evaluate_during_training, help="Run evaluation during training at each logging step.")
parser.add_argument("--each_epoch_eval", default=each_epoch_eval)
parser.add_argument("--each_batch_eval", default=each_batch_eval)
parser.add_argument("--each_checkpoint_eval", default=each_checkpoint_eval)
parser.add_argument("--mlm", default=mlm, help="Train with masked-language modeling loss instead of language modeling.")
parser.add_argument("--mlm_probability", type=float, default=mlm_probability, help="Ratio of tokens to mask for masked language modeling loss")
parser.add_argument("--max_seq_len", default=max_seq_len, type=int,
    help="Optional input sequence length after tokenization."
    "The training dataset will be truncated in block of this size for training."
    "Default to the model max input length for single sentence inputs (take into account special tokens).")
parser.add_argument("--per_gpu_train_batch_size", default=per_gpu_train_batch_size, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--per_gpu_eval_batch_size", default=per_gpu_eval_batch_size, type=int, help="Batch size per GPU/CPU for evaluation.")
parser.add_argument("--learning_rate", default=learning_rate, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--warmup_proportion", default=warmup_proportion, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--warmup_steps", default=warmup_steps, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--num_train_epochs", default=num_train_epochs, type=float, help="Total number of training epochs to perform.")
parser.add_argument("--gpu_start", default=gpu_start, type=int, help="The first GPU ID in multiple GPUs.")
parser.add_argument("--gpu_num", default=gpu_num, type=int, help="The number of the used GPUs.")
parser.add_argument("--gradient_accumulation_steps", type=int, default=gradient_accumulation_steps, 
    help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--weight_decay", default=weight_decay, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=adam_epsilon, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=max_grad_norm, type=float, help="Max gradient norm.")
parser.add_argument("--no_cuda", default=no_cuda, help="Avoid using CUDA when available")
parser.add_argument("--seed", type=int, default=seed, help="Random seed for initialization")
parser.add_argument("--seed_flag", default=seed_flag, help="Whether to set seed")
parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
parser.add_argument("--fp16_opt_level", type=str, default="O1", 
    help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
    "See details at https://nvidia.github.io/apex/amp.html")
parser.add_argument("--local_rank", type=int, default=local_rank, help="For distributed training: local_rank")
parser.add_argument("--early_stop", type=int, default=early_stop)
parser.add_argument("--early_stop_scale", type=float, default=early_stop_scale)
args = parser.parse_args()

