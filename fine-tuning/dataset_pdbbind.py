# import ptvsd
# ptvsd.enable_attach(address = ('0.0.0.0', 5678))
# ptvsd.wait_for_attach()

import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import pickle
import logging
import torch
import pandas as pd
from rdkit import Chem
from finetune_args import args
from AttentiveFP import Fingerprint, Fingerprint_viz, save_smiles_dicts, get_smiles_dicts, get_smiles_array

class PDBBindDataset(Dataset):
    def __init__(self, df):
        self.examples = df

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        df = self.examples.iloc[index]
        atom = torch.tensor(df['atom'], dtype=torch.float32)
        bond = torch.tensor(df['bond'], dtype=torch.float32)
        atom_degree = torch.tensor(df['atom_degree'], dtype=torch.long)
        bond_degree = torch.tensor(df['bond_degree'], dtype=torch.long)
        atom_mask = torch.tensor(df['atom_mask'], dtype=torch.float32)
        amino = torch.tensor(df['amino'], dtype=torch.float32)
        amino_degree = torch.tensor(df['amino_degree'], dtype=torch.long)
        amino_mask = torch.tensor(df['amino_mask'], dtype=torch.float32)
        tokenized_sent = torch.tensor(df['tokenized_sent'], dtype=torch.long)
        attention_mask = torch.tensor(df['attention_mask'], dtype=torch.long)
        token_type_ids = torch.tensor(df['token_type_ids'], dtype=torch.long)
        value = torch.tensor(df['value'], dtype=torch.float32)
        return atom, bond, atom_degree, bond_degree, atom_mask, amino, amino_degree, amino_mask, tokenized_sent, attention_mask, token_type_ids, value

class DataHandler():
    def __init__(self, raw_filename):        
        self.amino_dict = {}
        for key, value in args.vocab.items():
            if value - args.special_vocab_size >=  0:
                self.amino_dict[key] = value - args.special_vocab_size

        self.max_len = args.max_seq_len   # 512
        self.input_size = args.nonspecial_vocab_size
        self.enc_lib = np.eye(self.input_size)

        self.data_df, self.smile_feature_dict, self.degree_dicts, self.max_neighbor_num = self.load_df_degree_smiles(raw_filename)

    def load_df_degree_smiles(self, raw_filename):
        # raw_filename : "/data/eason/pdbbind/pdbbind2019.pkl"
        pkl_path = os.path.split(raw_filename)[0]  # "/data/eason/pdbbind"
        pkl_name = os.path.split(raw_filename)[1]  # "pdbbind2019.pkl"
        pkl_name_no_ext = os.path.splitext(pkl_name)[0]  # "pdbbind2019"

        example_filename = os.path.join(pkl_path, pkl_name_no_ext + '_cache.pkl')
        feature_filename = os.path.join(pkl_path, pkl_name_no_ext + '_cache_smiles_dict.pickle')
        degree_filename = os.path.join(pkl_path, pkl_name_no_ext + '_cache_degree_' + str(self.max_len) + '.pkl')

        if os.path.exists(example_filename) and os.path.exists(feature_filename) and os.path.exists(degree_filename):
            remained_df = pickle.load(open(example_filename, "rb"))
            feature_dicts = pickle.load(open(feature_filename, "rb"))
            degree_dicts = pickle.load(open(degree_filename, "rb"))
            max_neighbor_num = list(degree_dicts.items())[0][1].shape[1]
        else:
            # raw_df : df : ["PDB-ID", "Affinity-Value", "seq", "rdkit_smiles", "set", "contact_map"]
            with open(raw_filename, 'rb') as f:
                raw_df = pickle.load(f)

            smiles_tasks_df = raw_df

            filename = os.path.splitext(feature_filename)[0]
            # smilesList : array(['CC[C@@H](CSC[C@H](NC(=O)...', 'CC(C)Cc1ccccc1...', ...]), 13464
            smilesList = smiles_tasks_df[args.SMILES].values
            print("number of all smiles: ", len(smilesList))
            atom_num_dist = []
            remained_smiles = []
            canonical_smiles_list = []
            for smiles in smilesList:
                try:
                    mol = Chem.MolFromSmiles(smiles)  # input : smiles seqs, output : molecule obeject
                    atom_num_dist.append(len(mol.GetAtoms()))  # list : get atoms obeject num from molecule obeject
                    remained_smiles.append(smiles)  # list : smiles without transformation error
                    canonical_smiles_list.append(Chem.MolToSmiles(mol, isomericSmiles=True))  # canonical smiles without transformation error
                except:
                    print("the smile \"%s\" has transformation error in the first test" % smiles)
                    pass
            print("number of successfully processed smiles after the first test: ", len(remained_smiles))

            "----------------------the first test----------------------"
            smiles_tasks_df = smiles_tasks_df[smiles_tasks_df[args.SMILES].isin(remained_smiles)]  # df(13464) : include smiles without transformation error
            smiles_tasks_df[args.SMILES] = remained_smiles

            # smilesList : array(['CC[C@@H](CSC[C@H](NC(=O)...', 'CC(C)Cc1ccccc1...', ...]), 13464
            smilesList = remained_smiles  # update valid smile

            # feature_dicts(dict) : 
            #     {
            #       'smiles_to_atom_mask': smiles_to_atom_mask,
            #       'smiles_to_atom_info': smiles_to_atom_info,
            #       'smiles_to_bond_info': smiles_to_bond_info,
            #       'smiles_to_atom_neighbors': smiles_to_atom_neighbors,
            #       'smiles_to_bond_neighbors': smiles_to_bond_neighbors,
            #       'smiles_to_rdkit_list': smiles_to_rdkit_list
            #     }
            # smiles_to_atom_mask : 
            #     {
            #        smiles: array([1,1,1,...,1,0,0,0,....]) len: max_atom_len, 0 is padding
            #        smiles: ...,
            #        ...,
            #     }
            # smiles_to_atom_info : 
            #     {
            #        smiles: array([[0,1,0,...,0],[0,0,0,1,...,0],...,[0,0,0,..,0]]) size:(max_atom_len,num_atom_features) eg.(178,39), 
            #        smiles: ...,
            #        ...,
            #     }
            # smiles_to_bond_info : 
            #     {
            #        smiles: array([[1,0,0,...,0],[1,0,0,1,...,0],...,[0,0,0,..,0]]) size:(max_bond_len,num_atom_features) eg.(182,10), 
            #        smiles: ...,
            #        ...,
            #     }
            # smiles_to_atom_neighbors : 
            #     {
            #        smiles: array([[13,177,177,...],[28,177,177,...],...,[0,26,177,177,...],[14,16,177,177,...],...]) eg.(178,6), 177 is padding
            #        smiles: ...,
            #        ...,
            #     }
            # smiles_to_bond_neighbors : 
            #     {
            #        smiles: array([[0,181,181,...],[8,181,181,...],...,[0,1,181,181,...],[2,3,181,181,...],...]) eg.(178,6), 181 is padding
            #        smiles: ...,
            #        ...,
            #     }
            # smiles_to_rdkit_list : 
            #     {
            #        smiles: array([0,9,13,15,...]), rdkit_ix of all atoms, array(atom_len) eg.array(39)
            #        smiles: ...,
            #        ...,
            #     }

            # smilesList : array(['CC[C@@H](CSC[C@H](NC(=O)...', 'CC(C)Cc1ccccc1...', ...]), 13464
            # filename : "./PPI/drug/tasks/DTI/pdbbind/pafnucy_total_rdkit-smiles-v1"
            feature_dicts = save_smiles_dicts(smilesList, filename)
            print("Create smiles features at %s" % feature_filename)
            
            "----------------------the second test----------------------"
            # remained_df : array(['CC[C@@H](CSC[C@H](NC(=O)...', 'CC(C)Cc1ccccc1...', ...]) : include smiles without transformation error and second test error, 13435
            remained_df = smiles_tasks_df[smiles_tasks_df[args.SMILES].isin(feature_dicts['smiles_to_atom_mask'].keys())]
            print("number of successfully processed smiles after the second test: ", len(remained_df))

            degree_dicts, max_neighbor_num = self.get_cm_dict(remained_df)

            with open(example_filename, 'wb') as f:
                pickle.dump(remained_df, f)
            print("Create dataframe after test at %s" % example_filename)
            with open(degree_filename, 'wb') as f:
                pickle.dump(degree_dicts, f)
            print("Create protein degree at %s" % degree_filename)

        return remained_df, feature_dicts, degree_dicts, max_neighbor_num

    def get_cm_dict(self, df):
        seq_list = []
        id_list = []
        degree_dicts = {}
        max_neighbor_num = 0

        for _, row in df.iterrows():
            seq = row['seq'][:self.max_len]
            if seq == '':
                continue
            cm = row['contact_map'][0][:self.max_len, :self.max_len]  # row['contact_map']:208×208

            mn = np.max(np.sum(cm, axis=1))
            if max_neighbor_num < mn:
                max_neighbor_num = mn
        
        for _, row in df.iterrows():
            seq = row['seq'][:self.max_len]
            if seq == '':
                continue
            cm = row['contact_map'][0][:self.max_len, :self.max_len]
            seq_list.append(row['seq'])
            id_list.append(row['PDB-ID'])

            degree_tmp = []
            for i in range(len(seq)):
                tmp = np.array(np.where(cm[i] > 0.5)[0])
                tmp = np.pad(tmp, (0, max_neighbor_num - tmp.shape[0]), 'constant', constant_values=(-1, -1))
                degree_tmp.append(tmp)
            
            degree_tmp = np.stack(degree_tmp, 0)
            degree_tmp = np.pad(degree_tmp, ((0, self.max_len - degree_tmp.shape[0]), (0, 0)), 'constant',
                                constant_values=(-1, -1))

            degree_dicts[row['seq']] = degree_tmp
                
        return degree_dicts, max_neighbor_num

    def get_pro_structure(self, seq_list):
        """
        seq_list,shape(batch_size,): 
        array(['VNKERTFLAVKPDGVARGLVGEIIARYEKK', '...', ...]
        """
        # f1 = cal_mem()
        """
        amino_list, shape(batch_size, max_seq_len, one_hot_size)
        padding = [0, 0, 0, ..., 0]
        eg. [[0, 0, 1, ...], [0, 1, 0, ...], ..., [-1, -1 , -1, ...]]
        """
        amino_list = self.get_init(seq_list)
        # f2 = cal_mem()
        # print('Get Pro Structure Index {}-{} costs: {}MB'.format('f2', 'f1', round(f1-f2, 4)))
        """
        amino_degree_list, shape(batch_size, max_seq_len, max_neighbor_num)
        padding = [-1, -1, -1, ..., -1]
        eg. [[-1, -1, -1, ...], [index, index, index, ..., -1, -1], ..., [-1, -1 , -1, ...]]
        """
        amino_degree_list = self.get_degree_list(seq_list)
        # f3 = cal_mem()
        # print('Get Pro Structure Index {}-{} costs: {}MB'.format('f2', 'f3', round(f2 - f3, 4)))
        """
        amino_mask, shape(batch_size, max_seq_len)
        eg. [[1, 1, 1, ..., 0, 0], [1, 1, 1, ..., 0, 0], ..., [1, 1, 1, ..., 0, 0]]
        """
        amino_mask = self.get_amino_mask(seq_list)
        # f4 = cal_mem()
        # print('Get Pro Structure Index {}-{} costs: {}MB'.format('f3', 'f4', round(f3 - f4, 4)))

        return amino_list, amino_degree_list, amino_mask

    def get_init(self, seq_list):
        mat = []
        for seq in seq_list:
            seq = [args.vocab[i]-args.special_vocab_size for i in seq[:self.max_len]]
            # enc: array, (max_seq_len, no_special_tokens_vocab_size)
            enc = self.enc_lib[seq]
            if enc.shape[0] < self.max_len:
                enc = np.pad(enc, ((0, self.max_len - enc.shape[0]), (0, 0)), 'constant')
            # print(enc.shape)

            mat.append(enc)
        # mat: [array(max_seq_len, no_special_tokens_vocab_size), array(max_seq_len, no_special_tokens_vocab_size), ...]
        mat = np.stack(mat, 0)
        # mat: array(batch_size, max_seq_len, no_special_tokens_vocab_size)
        mat = mat.astype(np.float32)
        return mat

    def get_degree_list(self, seq_list):
        mat = []
        with_degree_num = 0
        without_degree_num = 0
        for seq in seq_list:
            if seq in self.degree_dicts:
                cm = self.degree_dicts[seq]
                with_degree_num += 1
            else:
                cm = np.ones([self.max_len, self.max_neighbor_num])
                cm = cm * -1
                without_degree_num += 1
            mat.append(cm)
        # print("protein num with degree is %d" % with_degree_num)
        # print("protein num without degree is %d" % without_degree_num)
        mat = np.stack(mat, 0)

        return mat

    def get_amino_mask(self, seq_list):
        mat = []
        for seq in seq_list:
            mask = np.ones(min(len(seq), self.max_len), dtype=np.int)
            mask = np.pad(mask, (0, self.max_len - len(mask)), 'constant')
            mat.append(mask)
        mat = np.stack(mat, 0)
        # print('mask', mat)
        return mat

def tokenize(sent_list, vocab, seq_len):
    seq_len = seq_len + 2 # add [CLS] and [SEP]
    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []

    for sent in sent_list:
        attention_mask = [1 for _ in range(seq_len)]
        token_type_ids = [0 for _ in range(seq_len)]
        tmp = [vocab['[CLS]']]

        for word in sent:
            tmp.append(vocab[word])
            if len(tmp) == seq_len - 1:
                break
        tmp.append(vocab['[SEP]'])
        cur_len = len(tmp)
        if cur_len < seq_len:
            for i in range(cur_len, seq_len):
                tmp.append(vocab['[PAD]'])
                attention_mask[i] = 0

        all_input_ids.append(tmp)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)

    all_input_ids = np.array(all_input_ids)
    all_attention_mask = np.array(all_attention_mask)
    all_token_type_ids = np.array(all_token_type_ids)

    return all_input_ids, all_attention_mask, all_token_type_ids

# 如更改n-gram则此处要改
def create_sent(seq_list, seg_len=1):
    sent_list = []

    for s in seq_list:
        tmp = []
        for i in range(len(s) - seg_len + 1):
            tmp.append(s[i: i + seg_len])

        sent_list.append(tmp)
    return sent_list

def generate_df(df, smile_feature_dict, data_handler):
    # smiles_list: array(batch_size,)
    smiles_list = df[args.SMILES].values
    # pro_seqs: array(batch_size,)
    pro_seqs = df.seq.values
    # y_val: array(batch_size,)
    y_val = df[args.TASK].values

    # x_atom：array, (batch_size, max_atom_len, num_atom_features)
    # x_bond: array, (batch_size, max_bond_len, num_bond_features)
    # x_atom_degree: array, (batch_size, max_atom_len, degrees_len)
    # x_bond_degree: array, (batch_size, max_bond_len, degrees_len)
    # x_mask: array, array, (batch_size, max_atom_len)
    # smiles_to_rdkit_list: dict, {smiles: array([atom_len]), smiles: array([atom_len]), ...,}  len: all_atoms_num
    x_atom, x_bond, x_atom_degree, x_bond_degree, x_mask, smiles_to_rdkit_list = get_smiles_array(smiles_list, smile_feature_dict)
    # amino_list: array, (batch_size, max_seq_len, no_special_tokens_vocab_size)
    # amino_degree_list: array, (batch_size, max_seq_len, max_neighbor_num), eg.(128,512,25)
    # amino_mask: array, (batch_size, max_seq_len), eg.(128,512)
    amino_list, amino_degree_list, amino_mask = data_handler.get_pro_structure(pro_seqs)

    # sents: list, [['V','N','K','E',...],['E','H','Y','I',...],...,['R','K','Q','I',...]], unequal length
    sents = create_sent(pro_seqs)
    # tokenized_sent: tensor(batch_size, max_seq_len), eg.tensor([[1,8,19,...],[1,10,22,...],...,[1,20,19,...]])
    #                 -->[[cls,8,19,...,sep,pad,pad,...,pad],[cls,10,22,...,sep,pad,pad,...,pad],...,[cls,20,19,...,sep,pad,pad,...,pad]]
    tokenized_sent, attention_mask, token_type_ids = tokenize(sents, args.vocab, args.max_seq_len)

    new_dict = {}
    new_dict['atom'] = list(x_atom)
    new_dict['bond'] = list(x_bond)
    new_dict['atom_degree'] = list(x_atom_degree)
    new_dict['bond_degree'] = list(x_bond_degree)
    new_dict['atom_mask'] = list(x_mask)
    new_dict['amino'] = list(amino_list)
    new_dict['amino_degree'] = list(amino_degree_list)
    new_dict['amino_mask'] = list(amino_mask)
    new_dict['tokenized_sent'] = list(tokenized_sent)
    new_dict['attention_mask'] = list(attention_mask)
    new_dict['token_type_ids'] = list(token_type_ids)
    new_dict['value'] = list(y_val)

    new_df = pd.DataFrame(new_dict)

    return new_df

def pdbbind_df(input_file):
    data_handler = DataHandler(input_file)

    # train_df:
    # ['PDB-ID', 'Affinity-Value', 'seq', 'rdkit_smiles', 'set', 'contact_map']
    # ['3zzf', 0.4, 'NGFSATRSTV...', 'CC(=O)N[C@@H](CCC(=O)O)C(=O)O', 'train', [array([[ True,  True,  True, ..., False, False, False], ..., [False, False, False, ...,  True,  True,  True]])]]
    train_df = data_handler.data_df[data_handler.data_df["set"].str.contains('train')]
    valid_df = data_handler.data_df[data_handler.data_df["set"].str.contains('valid')]
    test_test_df = data_handler.data_df[data_handler.data_df["set"].str.contains('test')]
    test_casf2013_df = data_handler.data_df[data_handler.data_df["set"].str.contains('casf2013')]
    test_astex_df = data_handler.data_df[data_handler.data_df["set"].str.contains('astex')]
    # print("raw data columns is", list(train_df.columns))
    # print("raw data format is", list(train_df.iloc[0, :]))

    feature_dicts = data_handler.smile_feature_dict

    # train_df:
    # ['atom', 'bond', 'atom_degree', 'bond_degree', 'atom_mask', 'amino', 'amino_degree', 'amino_mask', 'tokenized_sent', 'value']
    # [array([[0., 1., 0., ..., 0., 0., 0.], ..., [0., 0., 0., ..., 0., 0., 0.]]), (max_atom_len, num_atom_features), eg.(178, 39)
    #  array([[1., 0., 0., ..., 0., 0., 0.], ..., [0., 0., 0., ..., 0., 0., 0.]]), (max_bond_len, num_bond_features), eg.(182, 10)
    #  array([[  9., 177., 177., 177., 177., 177.], ..., [177., 177., 177., 177., 177., 177.]]), (max_atom_len, degrees_len)
    #  array([[  0., 181., 181., 181., 181., 181.], ..., [181., 181., 181., 181., 181., 181.]]), (max_bond_len, degrees_len)
    #  array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., ..., 0., 0., 0.]), (max_atom_len, )
    #  array([[0., 0., 0., ..., 1., 0., 0.], ..., [0., 0., 0., ..., 0., 0., 0.]], dtype=float32), (max_seq_len, no_special_tokens_vocab_size), (512, 25)
    #  array([[0, 1, 2, ..., -1, -1, -1], ..., [502, 503, 504., ..., -1, -1, -1]]), (max_seq_len, max_neighbor_num), (512, 25)
    #  array([1, 1, 1, 1, 1, 1, 1, 1, ..., 1, 1, 1]), (max_seq_len, ), (512, )
    #  array([ 1, 27, 5, 16, 25, 21, 28, 19, 25, 28, 13, ..., 8, 13, 15,  6, 27, 12, 24, 2]), (max_seq_len, ), (512)
    #  0.4]    
    train_df = generate_df(train_df, feature_dicts, data_handler)
    valid_df = generate_df(valid_df, feature_dicts, data_handler)
    test_test_df = generate_df(test_test_df, feature_dicts, data_handler)
    test_casf2013_df = generate_df(test_casf2013_df, feature_dicts, data_handler)
    test_astex_df = generate_df(test_astex_df, feature_dicts, data_handler)
    # print("feature data columns is", list(train_df.columns))
    # print("feature data format is", list(train_df.iloc[0, :]))

    return train_df, valid_df, test_test_df, test_casf2013_df, test_astex_df


if __name__ == "__main__":
    train_df, valid_df, test_test_df, test_casf2013_df, test_astex_df = pdbbind_df(args.input_file)

