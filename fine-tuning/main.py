# import ptvsd
# ptvsd.enable_attach(address = ('0.0.0.0', 5678))
# ptvsd.wait_for_attach()

import torch
import torch.nn as nn
import time
import numpy as np
import random
import copy
from finetune_args import args
from dataset_pdbbind import pdbbind_df, PDBBindDataset
from finetune_model import Network
from train_eval_test import train, evaluate, predict
from output_results import output_results

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # set seed for CPU
    torch.cuda.manual_seed(seed)  # set seed for current GPU
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(seed)  # set seed for all GPUs
    torch.backends.cudnn.deterministic = True  # CPU/GPU results are consistent
    torch.backends.cudnn.benchmark = False   # the training is accelerated when the training set changes little


if __name__ == "__main__":
    set_seed(args.seed)
    train_df, valid_df, test_test_df, test_casf2013_df, test_astex_df = pdbbind_df(args.input_file)

    print("train_df_nums: %d, valid_df_nums: %d, core2016_df_nums: %d, casf2013_df_nums: %d, casf2013_df_nums: %d" 
        % (len(train_df), len(valid_df), len(test_test_df), len(test_casf2013_df), len(test_astex_df)))

    args.device = torch.device("cuda:{}".format(args.gpu_start) if torch.cuda.is_available() else "cpu")
    args.device_ids = []  # IDs of GPUs to be used
    for i in range(args.n_gpu):
        args.device_ids.append(args.gpu_start+i)

    loss_function = nn.MSELoss()

    model = Network(args.radius, args.T, args.num_atom_features, args.num_bond_features, args.drug_dim, 
                    args.p_dropout, args.pro_seq_dim, args.pro_struc_dim, args.pro_dim, int(args.multi_dropout_num))

    model = model.to(args.device)
    model = nn.DataParallel(model, device_ids=args.device_ids)

    print('Model parameters:', sum(param.numel() for param in model.parameters()))

    train_dataset = PDBBindDataset(train_df)
    valid_dataset = PDBBindDataset(valid_df)
    test_core2016_dataset = PDBBindDataset(test_test_df)
    test_casf2013_dataset = PDBBindDataset(test_casf2013_df)
    test_astex_dataset = PDBBindDataset(test_astex_df)
    dataset_list = [train_dataset, valid_dataset, test_core2016_dataset, test_casf2013_dataset, test_astex_dataset]

    start_real_epochs = time.time()
    print("------------First training starts------------")
    best_param, real_epochs = train(args, model, dataset_list, loss_function)
    print("------------second training starts------------")
    best_param, real_epochs = train(args, copy.deepcopy(best_param['best_model']), dataset_list, loss_function, best_param, best_param["best_epoch"], last_train=True)

    output_results(args, best_param, dataset_list, real_epochs)

    end_real_epochs = time.time()
    print("------------All {} epochs training spend {}h-{}m-{:.4f}s------------".format(real_epochs,
                int((end_real_epochs-start_real_epochs)/3600), 
                int((end_real_epochs-start_real_epochs)/60), 
                (end_real_epochs-start_real_epochs)%60))


