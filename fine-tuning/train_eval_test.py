import time
import copy
import torch
import os
import json
import numpy as np
import scipy
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import f1_score

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup

def train(args, model, dataset_list, loss_function, best_param=None, start_epoch=0, last_train=False):
    train_dataset, valid_dataset, test_core2016_dataset, test_casf2013_dataset, test_astex_dataset = dataset_list
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    # no_decay = ["bias", "LayerNorm.weight"]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
    #         "weight_decay": args.weight_decay,
    #     },
    #     {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    # ]

    # t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
    # optimizer = AdamW(optimizer_grouped_parameters, lr=10 ** -args.learning_rate, eps=args.adam_epsilon)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    optimizer = optim.Adam(model.parameters(), 10 ** -args.learning_rate, weight_decay=10 ** -args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args.patience)

    if not best_param:
        best_param = {}
        best_param["best_epoch"] = 0
        best_param["train_MSE"] = 9e8
        best_param["valid_MSE"] = 9e8
        best_param['plot_RMSE'] = []
        best_param['plot_R'] = []
        best_param['best_model'] = None

    plot_RMSE = []
    plot_R = []

    early_stop = 0  # used to stop early
    real_epochs = start_epoch  # used to record real training epochs
    tr_loss = 0.0
    for epoch in range(start_epoch, args.epochs + start_epoch):
        start_time = time.time()
        for step, batch in enumerate(train_dataloader):
            model.train()
            x_atom, x_bond, x_atom_degree, x_bond_degree, x_mask, \
            amino_list, amino_degree_list, amino_mask, \
            tokenized_sent, attention_mask, token_type_ids, y_value = batch

            x_atom = x_atom.to(args.device)
            x_bond = x_bond.to(args.device)
            x_atom_degree = x_atom_degree.to(args.device)
            x_bond_degree = x_bond_degree.to(args.device)
            x_mask = x_mask.to(args.device)
            tokenized_sent = tokenized_sent.to(args.device)
            attention_mask = attention_mask.to(args.device)
            token_type_ids = token_type_ids.to(args.device)
            amino_list = amino_list.to(args.device)
            amino_degree_list = amino_degree_list.to(args.device)
            amino_mask = amino_mask.to(args.device)
            y_value = y_value.to(args.device)

            prediction = model(x_atom, x_bond, x_atom_degree, x_bond_degree, x_mask, tokenized_sent, attention_mask, token_type_ids, \
                               amino_list, amino_degree_list, amino_mask)

            loss = loss_function(prediction, y_value.view(-1, 1))

            # b = 0.9
            # flood = (loss - b).abs() + b

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            
            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0 or (
                # last step in epoch but step is always smaller than gradient_accumulation_steps
                len(train_dataloader) <= args.gradient_accumulation_steps
                and (step + 1) == len(train_dataloader)
            ):
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                # scheduler.step()  # Update learning rate schedule
                model.zero_grad()

        train_MSE, train_R, _, _ = evaluate(args, model, train_dataset)
        valid_MSE, valid_R, _, _ = evaluate(args, model, valid_dataset)
        test_MSE, test_R, _, _ = evaluate(args, model, test_core2016_dataset)
        test_casf2013_MSE, test_casf2013_R, _, _ = evaluate(args, model, test_casf2013_dataset)
        test_astex_MSE, test_astex_R, _, _ = evaluate(args, model, test_astex_dataset)

        real_epoch = epoch+1

        for para_group in optimizer.param_groups:
            lr = para_group['lr']
    
        scheduler.step(valid_MSE)
        end_time = time.time()
        
        RMSE_log = 'epoch: {}, train_RMSE:{:.4f}, valid_RMSE:{:.4f}, test_RMSE(2016):{:.4f}, test_RMSE(casf2013):{:.4f}, test_RMSE(astex):{:.4f}'.format(
            real_epoch, np.sqrt(train_MSE), np.sqrt(valid_MSE), np.sqrt(test_MSE), np.sqrt(test_casf2013_MSE), np.sqrt(test_astex_MSE))
        R_log = len('epoch: {}, '.format(epoch))*' '+'train_R:{:.4f}, valid_R:{:.4f}, test_R(2016):{:.4f}, test_R(casf2013):{:.4f}, test_R(astex):{:.4f}, lr: {}'.format(
            train_R, valid_R, test_R, test_casf2013_R, test_astex_R, lr)
        each_epoch_time = "------------The {} epoch spend {}m-{:.4f}s------------".format(real_epoch, int((end_time-start_time)/60), (end_time-start_time)%60)
        print(RMSE_log)
        print(R_log)
        print(each_epoch_time)
        with open(args.log_file, 'a') as pickle_file:
            pickle_file.write(RMSE_log+'\n')
            pickle_file.write(R_log+'\n')
            pickle_file.write(each_epoch_time+'\n')
        plot_RMSE.append([real_epoch, np.sqrt(train_MSE), np.sqrt(valid_MSE), np.sqrt(test_MSE), np.sqrt(test_casf2013_MSE), np.sqrt(test_astex_MSE)])
        plot_R.append([real_epoch, train_R, valid_R, test_R, test_casf2013_R, test_astex_R])

        if valid_MSE < best_param["valid_MSE"]:
            best_param["train_MSE"] = train_MSE
            best_param["best_epoch"] = real_epoch
            best_param["valid_MSE"] = valid_MSE
            best_param['best_model'] = copy.deepcopy(model)
            best_plot_RMSE = copy.deepcopy(plot_RMSE)
            best_plot_R = copy.deepcopy(plot_R)

        real_epochs = real_epochs + 1
        if epoch != start_epoch:
            if abs(last_valid_RMSE - np.sqrt(valid_MSE))/last_valid_RMSE <= args.early_stop_scale or np.sqrt(valid_MSE) > last_valid_RMSE: 
                early_stop = early_stop+1
            else:
                early_stop = 0
        if early_stop == args.early_stop:
            break
        last_valid_RMSE = np.sqrt(valid_MSE)
    if last_train:
        best_param['plot_RMSE'] += plot_RMSE
        best_param['plot_R'] += plot_R
    else:
        best_param['plot_RMSE'] += best_plot_RMSE
        best_param['plot_R'] += best_plot_R

    return best_param, real_epochs

def evaluate(args, model, eval_dataset):
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.batch_size)

    eval_MSE_list = []
    pred_list = []
    true_list = []
    for _, batch in enumerate(eval_dataloader):
        model.eval()
        x_atom, x_bond, x_atom_degree, x_bond_degree, x_mask, \
        amino_list, amino_degree_list, amino_mask, \
        tokenized_sent, attention_mask, token_type_ids, y_value = batch

        x_atom = x_atom.to(args.device)
        x_bond = x_bond.to(args.device)
        x_atom_degree = x_atom_degree.to(args.device)
        x_bond_degree = x_bond_degree.to(args.device)
        x_mask = x_mask.to(args.device)
        tokenized_sent = tokenized_sent.to(args.device)
        attention_mask = attention_mask.to(args.device)
        token_type_ids = token_type_ids.to(args.device)
        amino_list = amino_list.to(args.device)
        amino_degree_list = amino_degree_list.to(args.device)
        amino_mask = amino_mask.to(args.device)
        y_value = y_value.to(args.device)
        
        with torch.no_grad():
            prediction = model(x_atom, x_bond, x_atom_degree, x_bond_degree, x_mask, tokenized_sent, attention_mask, token_type_ids, \
                               amino_list, amino_degree_list, amino_mask)

        MSE = F.mse_loss(prediction, y_value.view(-1, 1), reduction='none')

        eval_MSE_list.extend(MSE.data.squeeze().cpu().numpy())
        pred_list.extend(prediction.data.squeeze().cpu().numpy())
        true_list.extend(y_value.cpu().numpy())

    corr = scipy.stats.pearsonr(pred_list, true_list)[0]
    return np.array(eval_MSE_list).mean(), corr, np.array(pred_list), np.array(true_list)

def predict(args, model, test_dataset, set_type):
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size)

    pred_list = []
    true_list = []
    for _, batch in enumerate(test_dataloader):
        model.eval()

        x_atom, x_bond, x_atom_degree, x_bond_degree, x_mask, \
        amino_list, amino_degree_list, amino_mask, \
        tokenized_sent, attention_mask, token_type_ids, y_value = batch

        x_atom = x_atom.to(args.device)
        x_bond = x_bond.to(args.device)
        x_atom_degree = x_atom_degree.to(args.device)
        x_bond_degree = x_bond_degree.to(args.device)
        x_mask = x_mask.to(args.device)
        tokenized_sent = tokenized_sent.to(args.device)
        attention_mask = attention_mask.to(args.device)
        token_type_ids = token_type_ids.to(args.device)
        amino_list = amino_list.to(args.device)
        amino_degree_list = amino_degree_list.to(args.device)
        amino_mask = amino_mask.to(args.device)
        y_value = y_value.to(args.device)
        
        with torch.no_grad():
            prediction = model(x_atom, x_bond, x_atom_degree, x_bond_degree, x_mask, tokenized_sent, attention_mask, token_type_ids, \
                               amino_list, amino_degree_list, amino_mask)
      
        pred_list.extend(prediction.data.squeeze().cpu().numpy())
        true_list.extend(y_value.cpu().numpy())
        set_list = [set_type] * len(pred_list)
    return pred_list, true_list, set_list