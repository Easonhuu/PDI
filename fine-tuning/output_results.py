import os
import torch
import json
import numpy as np
from plot_results import plot_fold_line, plot_scatter
from train_eval_test import train, evaluate, predict

def output_results(args, best_param, dataset_list, real_epochs):
    train_dataset, valid_dataset, test_core2016_dataset, test_casf2013_dataset, test_astex_dataset = dataset_list
    dir_save = os.path.join(args.output_dir, args.model_type + "-{:.3f}-{}-{}".format(np.sqrt(best_param['valid_MSE']), best_param['best_epoch'], real_epochs))
    if not os.path.exists(dir_save):
        os.makedirs(dir_save)
        print(dir_save+" create successful!")
    else:
        print(dir_save+" already exists.")
    model_to_save = best_param['best_model'].module if hasattr(best_param['best_model'], "module") else best_param['best_model'] # Take care of distributed/parallel training
    torch.save(model_to_save.state_dict(), dir_save+'/best-model-{:.3f}-{}-{}.pth'.format(np.sqrt(best_param['valid_MSE']), best_param['best_epoch'], real_epochs))

    with open(os.path.join(dir_save, 'finetune_model_parameters.txt'), 'w') as f:
        for param_name, param_value in model_to_save.named_parameters():
            print(param_name, ":", param_value.size(), file=f)
        print('Model parameters:', sum(param.numel() for param in model_to_save.parameters()), file=f)

    os.system("cp " + args.log_file + " " + dir_save)
    os.remove(args.log_file)
    # os.system("cp " + __file__ + " " + dir_save)
    with open(os.path.join(dir_save, 'args.json'), 'w') as f:
        args_dict = args.__dict__
        args_dict_new = {}
        for k, v in args_dict.items():
            try:
                json.dumps({k:v})
                args_dict_new[k] = v
            except:
                continue
        json.dump(args_dict_new, f, indent=2)
    
    _, _, label_pred_train, label_true_train = evaluate(args, best_param['best_model'], train_dataset)
    _, _, label_pred_valid, label_true_valid = evaluate(args, best_param['best_model'], valid_dataset)
    _, _, label_pred_core2016, label_true_core2016 = evaluate(args, best_param['best_model'], test_core2016_dataset)
    _, _, label_pred_casf2013, label_true_casf2013 = evaluate(args, best_param['best_model'], test_casf2013_dataset)
    _, _, label_pred_astex, label_true_astex = evaluate(args, best_param['best_model'], test_astex_dataset)

    plot_fold_line(best_param['plot_RMSE'], best_param['plot_R'], dir_save)
    plot_scatter(label_pred_train, label_true_train, dir_save, 'train')
    plot_scatter(label_pred_valid, label_true_valid, dir_save, 'valid')
    plot_scatter(label_pred_core2016, label_true_core2016, dir_save, 'core2016')
    plot_scatter(label_pred_casf2013, label_true_casf2013, dir_save, 'casf2013')
    plot_scatter(label_pred_astex, label_true_astex, dir_save, 'astex')