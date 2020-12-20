import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from pro_network import ProNetwork
from AttentiveFP import Fingerprint


class Network(nn.Module):

    def __init__(self, radius, T, input_atom_dim, input_bond_dim, \
                 drug_dim, p_dropout, pro_seq_dim, pro_struc_dim, pro_dim, multi_dropout_num):
        super(Network, self).__init__()
        self.loss_function = nn.MSELoss()

        self.GAT = Fingerprint(radius, T, input_atom_dim, input_bond_dim, drug_dim, p_dropout)

        self.Pro = ProNetwork(p_dropout, pro_seq_dim, pro_struc_dim, pro_dim)

        self.multi_dropout_num = multi_dropout_num
        if self.multi_dropout_num == 1:
            self.fully_connected = nn.Sequential(nn.Dropout(p_dropout),
                                                nn.Linear(drug_dim + pro_dim, 1))
        else:
            multi_p_dropout = [0.1*(i+1) for i in range(multi_dropout_num)]
            self.dropouts = nn.ModuleList([nn.Dropout(p_dropout) for p_dropout in multi_p_dropout])
            self.dense_layer = nn.Linear(drug_dim + pro_dim, 1)
        
    def forward(self, atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask, tokenized_sent, attention_mask, token_type_ids, \
                amino_list, amino_degree_list, amino_mask):
        # pro_list B * Seq_len * word_size
        # smile_feature: tensor(batch_size, num_atom_features)
        smile_feature = self.GAT(atom_list, bond_list, atom_degree_list, bond_degree_list, atom_mask)

        pro_feature = self.Pro(tokenized_sent, attention_mask, token_type_ids, amino_list, amino_degree_list, amino_mask)
        con_feature = torch.cat((smile_feature, pro_feature), dim=1)

        if self.multi_dropout_num == 1:
            prediction = self.fully_connected(con_feature)
        else:
            for i, dropout in enumerate(self.dropouts):
                if i == 0:
                    out = dropout(con_feature)
                    prediction = self.dense_layer(out)
                else:
                    out = dropout(con_feature)
                    prediction += self.dense_layer(out)
            prediction = prediction / len(self.dropouts)

        return prediction

