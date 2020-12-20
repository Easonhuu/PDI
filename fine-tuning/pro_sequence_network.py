# import ptvsd
# ptvsd.enable_attach(address = ('0.0.0.0', 5678))
# ptvsd.wait_for_attach()

from transformers import AutoConfig, AutoModelWithLMHead, CONFIG_MAPPING
from finetune_args import args
import torch
import torch.nn as nn

config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
ProTransformer_20w = AutoModelWithLMHead.from_config(config)

finetune_state_dict = ProTransformer_20w.state_dict()        
pretrain_state_dict = torch.load(args.pretrain_model_dict, map_location='cpu')

state_dict_common = {}
for k, v in pretrain_state_dict.items():
    k = k.replace('module.', '')
    if k in finetune_state_dict:
        state_dict_common.update({k: v})
finetune_state_dict.update(state_dict_common)
ProTransformer_20w.load_state_dict(finetune_state_dict)

# param_numel = []
# for k, v in pretrain_state_dict.items():
#     param_numel.append(v.numel())
# print('Model Parameters:', sum(param_numel))
