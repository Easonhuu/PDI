# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""
# import ptvsd
# ptvsd.enable_attach(address = ('0.0.0.0', 5678))
# ptvsd.wait_for_attach()

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple
import torch

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import time
import copy
import json

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    BertTokenizer,
    BertConfig,
    BertForMaskedLM,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
    AutoModel,
)

from transformers.modeling_bert import BertForMaskedLM
from pretraining_args import args

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, attention_mask, token_type_ids, masked_lm_labels ):
        self.input_ids = input_ids
        self.attention_mask  = attention_mask 
        self.token_type_ids  = token_type_ids 
        self.masked_lm_labels  = masked_lm_labels 

class TextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, max_seq_len=512):
        assert os.path.isfile(file_path)
        # directory: "./data"; filename: "train_mlm.tsv"
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, args.model_type + "_cached_lm_" + str(max_seq_len) + "_" + filename)

        max_seq_len = max_seq_len - (tokenizer.max_len - tokenizer.max_len_single_sentence)  # eg.1022 = 1024 - (1024 - 1022)

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.readlines()
                tokenized_text = []
                for line in text:
                    line = line.strip().split(' ')
                    if len(line) > max_seq_len:
                        line = line[:max_seq_len]
                    # line: list(['M', 'K', 'I', 'N', 'S', 'S', 'K', ...])
                    # tokenized_line: list([21, 6, 19, 13, 16, 16, 6, ...])
                    tokenized_line = tokenizer.convert_tokens_to_ids(line)
                    tokenized_text.append(tokenized_line)
            """
            tokenized_text : list, [[id1, id2, ..., ], [id1, id2, ..., ], ..., [id1, id2, ..., ]], unequal length
            self.examples : list, [[cls_id, id1, id2, ..., seq_id], [cls_id, id1, id2, ..., seq_id], ...], unequal length
            """
            for i in range(len(tokenized_text)):  # Truncate in block of max_seq_len
                self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i]))
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should look for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.
            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)


def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    return TextDataset(tokenizer, args, file_path=file_path, max_seq_len=args.max_seq_len)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted

def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )
    """
    inputs ：tensor, [[cls_id, id1, id2, ..., sep_id, pad_id,..., pad_id], [cls_id, id1, id2, ..., sep_id, pad_id,..., pad_id], ...], equal length
    """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    """
    probability_matrix : tensor, [[0.15, ..., 0.15], [0.15, ..., 0.15], ...]
    """
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    """
    special_tokens_mask : list, [[1,0,0,...,1,0,0,...], [1,0,0,...,1,0,0,...], ....], special_token(but [PAD] is 0) is 1，seq_token is 0
    """
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    """
    special_tokens_mask所有为1的位置在对应probability_matrix的概率置为0，即不参与mask, 这里主要针对两个特殊的token，一个是cls，一个是sep
    """
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask,dtype=torch.bool), value=0.0)
    """
    padding_mask : tensor, [[0,0,...,1,1,1], [0,0,...,1,1,1], ..., [0,0,...,1,1,1]], 所有pad为1，其余为0
    padding_mask所有为1的位置在对应probability_matrix的概率置为0，即不参与mask, 这里主要针对pad
    """
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    """
    masked_indices : tensor, [[False, False, ..., True, ..., ], [False, False, ..., True, ..., ], ...], 利用伯努利分布把需要mask的索引随机选出来，随机概率小于0.15的作为mask
    labels ：tensor, [[-100, id1, -100, ..., idn, -100,..., -100], [-100, id1, -100, ..., idn, -100,..., -100], ...], 等长, -100表示非mask的token
    """
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens
    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    """ Train the model """

    """
    train_dataset : list, [[cls_id, id1, id2, ..., sep_id], [cls_id, id1, id2, ..., sep_id], ...], unequal length
    train_dataset_pad ： tensor, [[cls_id, id1, id2, ..., sep_id, pad_id,..., pad_id], [cls_id, id1, id2, ..., sep_id, pad_id,..., pad_id], ...], equal length
    By the way,train_dataset_pad's max sequence length is the longest one of train_dataset, not args.max_seq_len.
    """
    all_len=[]
    for i in train_dataset.examples:
        all_len.append(len(i))
    train_dataset_pad = pad_sequence(train_dataset, batch_first=True, padding_value=tokenizer.pad_token_id)
    train_sampler = RandomSampler(train_dataset_pad) if args.local_rank == -1 else DistributedSampler(train_dataset_pad)
    train_dataloader = DataLoader(
        train_dataset_pad, sampler=train_sampler, batch_size=args.train_batch_size
    )

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    # )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_proportion*t_total, 
        num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if (
        args.model_name_or_path
        and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
        and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=args.device_ids)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    train_iterator = range(
        epochs_trained, int(args.num_train_epochs)
    )
    model.zero_grad()

    set_seed(args)  # Added here for reproducibility
    best_eval_loss = 9e8
    early_stop = 0
    for e in train_iterator:
        args.real_epoch = e + 1
        tr_loss = 0.0
        time_inter = 0.0
        per_epoch_step = 0
        per_epoch_steps = int(len(train_dataloader) / args.gradient_accumulation_steps)
        for step, batch in enumerate(train_dataloader):
            start_time = time.time()
            """
            batch ：tensor, [[cls_id, id1, id2, ..., sep_id, pad_id,..., pad_id], [cls_id, id1, id2, ..., sep_id, pad_id,..., pad_id], ...], 等长
            """

            """
            inputs ：tensor, [[cls_id, id1, mask_id, ...,random_id, ..., sep_id, pad_id,..., pad_id], ..., ...], 等长，15%是用预测的，其中80%mask，10%随机id替换，10%保持不变
            labels : tensor, [[-100, id1, -100, ..., idn, -100,..., -100], [-100, id1, -100, ..., idn, -100,..., -100], ...], 等长, -100表示非预测的token 
            """
            inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()

            # BertForMaskedLM forward : 
            """
            outputs有两个tensor，第一个是最后一层的序列的embedding(batch_size, sequence_length, hidden_size)，
            第二个是最后一层序列的cls的embedding（batch_size, hidden_size）（经过了一个线性层和一个tanh处理，线性层参数是nsp训练的）
            """
            # outputs = self.bert(input_ids)
            # sequence_output = outputs[0]
            """
            self.cls是线性层，该线性层的结构如下：1.fc(hidden_size, hidden_size); 2.gelu激活函数 3.nn.LayerNorm 4.fc(hidden_size, vocab_size)
            prediction_scores : tensor, (batch_size, sequence_length, vocab_size)
            outputs : tuple, (prediction_scores)
            """
            # prediction_scores = self.cls(sequence_output)
            # outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here
            """
            masked_lm_loss : tensor, [gpu1_loss, gpu2_loss, gpu3_loss, gpu4_loss], CrossEntropyLoss遇到-100会在计算loss时忽略
            outputs : tuple, (masked_lm_loss, prediction_scores)
            """
            # if masked_lm_labels is not None:
            #     loss_fct = CrossEntropyLoss()  # -100 index = padding token
            #     masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            #     outputs = (masked_lm_loss,) + outputs
            """
            outputs : tuple, (masked_lm_loss, prediction_scores)
            """
            outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)

            # batch = tuple(t.to(args.device) for t in batch)
            # input_ids, input_mask, segment_ids, label_ids = batch
            # # masked_lm_loss
            # outputs = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, masked_lm_labels=label_ids)

            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            if args.n_gpu > 1:
                """
                这里求mean，也就是考虑一个gpu单样本的loss
                """
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:  # 计算loss的gradient时，batch_size越大越耗内存，内存不够的话可以将batch_size分批进行
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()  # 将当前GPU储存gradient的计算图反传

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()  # 利用计算图中储存的gradient更新网络参数
                scheduler.step()  # Update learning rate schedule
                model.zero_grad() # 将计算图中储存的gradient清零
                global_step += 1

                end_time = time.time()
                time_inter = time_inter + end_time - start_time
                per_epoch_step += 1

                if int(step/args.gradient_accumulation_steps) != per_epoch_steps - 1 :
                    print("\r============================ -epoch %d[%d/%d] -train_loss %.4f -train_batch_spend_time %.4fs" % 
                        (args.real_epoch, per_epoch_step, per_epoch_steps, tr_loss / per_epoch_step, time_inter), end="", flush=True)
                else:
                    print("\r============================ -epoch %d[%d/%d] -train_loss %.4f -train_batch_spend_time %.4fs\n" % 
                        (args.real_epoch, per_epoch_step, per_epoch_steps, tr_loss / per_epoch_step, time_inter), end="", flush=True)

        if args.each_epoch_eval:
            # 在每个epoch评价测试集
            eval_result = evaluate(args, model, tokenizer)
            # # 加入训练集整体测试
            # train2eval_loss = 0
            # nb_train_eval_steps = 0
            # for batch in tqdm(train_dataloader, desc="TrainSet Evaluating"):
            #     inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
            #     inputs = inputs.to(args.device)
            #     labels = labels.to(args.device)

            #     with torch.no_grad():
            #         outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
            #         loss = outputs[0]

            #     if args.n_gpu > 1:
            #         loss = loss.mean()  # mean() to average on multi-gpu.
            #     if args.gradient_accumulation_steps > 1:
            #         loss = loss / args.gradient_accumulation_steps

            #     train2eval_loss += loss.item()
            #     nb_train_eval_steps += 1

            # train2eval_loss = train2eval_loss / nb_train_eval_steps
            train2eval_loss = tr_loss / per_epoch_step
            if e == 0:
                with open(args.log_file, "a", encoding="utf-8") as f:
                    f.write("%s, max length: %d, batch_size: %d\n" % (args.corpus_name, args.max_seq_len, args.per_gpu_train_batch_size*args.gpu_num*args.gradient_accumulation_steps))
            epoch_log = "============================ -epoch %d -train_loss %.4f -eval_loss %.4f\n" % (args.real_epoch, train2eval_loss, eval_result['eval_loss'])
            with open(args.log_file, "a", encoding="utf-8") as f:
                f.write(epoch_log)
            logger.info(epoch_log)

        if best_eval_loss > eval_result['eval_loss']:
            best_eval_loss = eval_result['eval_loss']
            best_train_loss = train2eval_loss
            best_epoch = args.real_epoch
            best_model = copy.deepcopy(model)

        if e != 0:
            if last_eval_loss - eval_result['eval_loss'] <= args.early_stop_scale or eval_result['eval_loss'] > last_eval_loss: 
                early_stop = early_stop + 1
            else:
                early_stop = 0
        if early_stop == args.early_stop:
            break
        last_eval_loss = eval_result['eval_loss']
    
    # 保存在验证集上表现最好的model
    args.best_output_dir = args.output_dir + args.model_type + "-{:.4f}-{}-{}-{}".format(best_eval_loss, best_epoch, args.real_epoch, args.corpus_name)
    os.makedirs(args.best_output_dir, exist_ok=True)
    model_to_save = (
        best_model.module if hasattr(best_model, "module") else best_model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args.best_output_dir)
    torch.save(best_model.state_dict(), os.path.join(args.best_output_dir, "pytorch_model.pth"))
    os.system("cp " + args.log_file + " " + args.best_output_dir)
    os.system("cp " + "./pre-training/run_mlm_pro_seq.py" + " " + args.best_output_dir)
    logger.info("Saving epoch-{}-loss-{:.4f}-best model in %s".format(best_epoch, best_eval_loss), args.best_output_dir)

    return global_step, best_epoch, best_eval_loss, best_train_loss

def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    """
    eval_dataset : list, [[cls_id, id1, id2, ..., sep_id], [cls_id, id1, id2, ..., sep_id], ...], 不等长
    eval_dataset_pad ： tensor, [[cls_id, id1, id2, ..., sep_id, pad_id,..., pad_id], [cls_id, id1, id2, ..., sep_id, pad_id,..., pad_id], ...], 等长
    """
    eval_dataset_pad = pad_sequence(eval_dataset, batch_first=True, padding_value=tokenizer.pad_token_id)
    eval_sampler = RandomSampler(eval_dataset_pad) if args.local_rank == -1 else DistributedSampler(eval_dataset_pad)
    eval_dataloader = DataLoader(
        eval_dataset_pad, sampler=eval_sampler, batch_size=args.eval_batch_size
    )
    set_seed(args)

    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    eval_loss = 0
    nb_eval_steps = 0

    for step, batch in enumerate(eval_dataloader):
        inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
            loss = outputs[0]

        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        if (step + 1) % args.gradient_accumulation_steps == 0:
            nb_eval_steps += 1
        eval_loss += loss.item()
        
    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity, "eval_loss": eval_loss}

    return result

def main():
    if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )
    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir)
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda:{}".format(args.gpu_start) if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else args.gpu_num
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device
    args.device_ids = []
    for i in range(args.n_gpu):
        args.device_ids.append(args.gpu_start+i)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    if args.seed_flag:
        set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    if args.config_name:
        config = AutoConfig.for_model(args.model_type).from_json_file(args.config_name)
        # config = BertConfig.from_json_file("./pre-training/bert_config.json")
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        # When we release a pip version exposing CONFIG_MAPPING,
        # we can do `config = CONFIG_MAPPING[args.model_type]()`.
        raise ValueError(
            "You are instantiating a new config instance from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --config_name"
        )

    if args.tokenizer_name:
        Tokenizer = BertTokenizer
        tokenizer = Tokenizer(vocab_file=args.tokenizer_name, max_seq_len=args.max_seq_len-2, max_len=args.max_seq_len)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name"
        )
    if not os.path.exists(args.protein_vocab_path):
        directory = os.path.split(args.protein_vocab_path)[0]
        logger.info("Creating protein_vocab.json at %s" % (directory))
        ids_tokens_dict = dict(tokenizer.ids_to_tokens)
        tokens_ids_dict = {}
        for ids_tokens in ids_tokens_dict.items():
            tokens_ids_dict[ids_tokens[1]] = int(ids_tokens[0])

        with open(args.protein_vocab_path, 'w') as f:
            json.dump(tokens_ids_dict,f)

    if args.max_seq_len <= 0:
        args.max_seq_len = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        args.max_seq_len = min(args.max_seq_len, tokenizer.max_len)

    if args.model_name_or_path:
        model = AutoModelWithLMHead.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelWithLMHead.from_config(config)

    model.to(args.device)
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    if args.do_train:
        prefix_filename = args.pretrain_data_path.split('/')[-1].replace('.big', '')
        start_time_log = str(time.ctime()).replace(':', '-').replace(' ', '_')
        args.log_file = args.pretrain_log + prefix_filename + '_1gram' + '_' + start_time_log + '.log'

        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        for param_name, param_value in model.named_parameters():
            print(param_name, ":", param_value.size())

        global_step, best_epoch, best_eval_loss, best_train_loss = train(args, train_dataset, model, tokenizer)

        logger.info("In all, %d epoch were trained, global steps were %d. best epoch is %d, best train loss is %f, best eval loss is %f.", 
                     args.real_epoch, global_step, best_epoch, best_train_loss, best_eval_loss)

    # return results
    if args.do_eval:
        if args.do_train == False:
            args.best_output_dir = ""  # decide which model to view by yourself
        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)
        train_dataset_pad = pad_sequence(train_dataset, batch_first=True, padding_value=tokenizer.pad_token_id)
        train_sampler = RandomSampler(train_dataset_pad) if args.local_rank == -1 else DistributedSampler(train_dataset_pad)
        train_dataloader = DataLoader(
            train_dataset_pad, sampler=train_sampler, batch_size=args.train_batch_size
        )
        # model = AutoModelWithLMHead.from_pretrained(args.output_dir)
        # model.to(args.device)
        # 验证集整体测试
        model = AutoModelWithLMHead.from_config(config)
        model.to(args.device)
        model = torch.nn.DataParallel(model, device_ids=args.device_ids)
        model.load_state_dict(torch.load(os.path.join(args.best_output_dir, "pytorch_model.pth")))
        
        eval_result = evaluate(args, model, tokenizer)
        # 加入训练集整体测试
        train2eval_loss = 0
        nb_train_eval_steps = 0
        for step, batch in enumerate(train_dataloader):
            inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            with torch.no_grad():
                outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
                loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            train2eval_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                nb_train_eval_steps += 1

        train2eval_loss = train2eval_loss / nb_train_eval_steps
        logger.info("============================ Eval directly, -train_loss %.4f -eval_loss %.4f\n" % 
                        (train2eval_loss, eval_result['eval_loss']))


if __name__ == "__main__":
    main()
