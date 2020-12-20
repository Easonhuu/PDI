import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle
import pandas as pd
import numpy as np


class ProGAT(nn.Module):

    def __init__(self, init_dim, embedding_size, radius, T, p_dropout):
        super(ProGAT, self).__init__()
        self.emb_layer = nn.Linear(init_dim, embedding_size)
        self.neighbor_fc = nn.Linear(init_dim, embedding_size)
        self.GRUCell = nn.ModuleList([nn.GRUCell(embedding_size, embedding_size) for r in range(radius)])
        self.align = nn.ModuleList([nn.Linear(2 * embedding_size, 1) for r in range(radius)])
        self.attend = nn.ModuleList([nn.Linear(embedding_size, embedding_size) for r in range(radius)])

        self.seq_GRUCell = nn.GRUCell(embedding_size, embedding_size)
        self.seq_align = nn.Linear(2 * embedding_size, 1)
        self.seq_attend = nn.Linear(embedding_size, embedding_size)

        self.dropout = nn.Dropout(p=p_dropout)

        self.radius = radius
        self.T = T

    def forward(self, amino_list, amino_degree_list, amino_mask):
        '''
        amino_list is (batch, max_seq_len(512), init_dim(512)).
        amino_degree_list is (batch, max_seq_len(512), max_neighbors(23)). One mat(512, 23) is the index of the amino's neighbors, -1 is padding.
        amino_mask is (batch, 512), the existed is 1 while padding is 0.
        '''
        # amino_mask: tensor(batch_size, max_seq_len, 1)
        amino_mask = amino_mask.unsqueeze(2)
        batch_size, seq_len, init_dim = amino_list.shape
        # (512, 512) -> (512, 64)  embed amino embedding_size
        # amino_feature: tensor(batch_size, max_seq_len, embedding_size)
        amino_feature = F.leaky_relu(self.emb_layer(amino_list))
        # neighbor: tensor(batch, max_seq_len, max_neighbors_num, init_dim)
        neighbor = [amino_list[i][amino_degree_list[i]] for i in range(batch_size)]
        neighbor = torch.stack(neighbor, dim=0)
        # (batch, 512, 23, 512) -> (batch, 512, 23, 64)
        # neighbor_feature: tensor(batch_size, max_seq_len, max_neighbors_num, init_dim)
        neighbor_feature = F.leaky_relu(self.neighbor_fc(neighbor))

        # generate mask to eliminate the influence of blank atoms
        attend_mask = amino_degree_list.clone()
        attend_mask[attend_mask != -1] = 1
        attend_mask[attend_mask == -1] = 0
        # attend_mask: tensor(batch_size, max_seq_len, max_neighbors_num, 1)
        attend_mask = attend_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

        softmax_mask = amino_degree_list.clone()
        softmax_mask[softmax_mask != -1] = 0
        softmax_mask[softmax_mask == -1] = -9e8  # make the softmax value extremly small
        # softmax_mask: tensor(batch_size, max_seq_len, max_neighbors_num, 1)
        softmax_mask = softmax_mask.type(torch.cuda.FloatTensor).unsqueeze(-1)

        batch_size, seq_length, max_neighbor_num, embedding_dim = neighbor_feature.shape
        # amino_feature_expand: tensor(batch_size, max_seq_len, max_neighbor_num, num_amino_feature)
        amino_feature_expand = amino_feature.unsqueeze(-2).expand(batch_size, seq_length, max_neighbor_num, embedding_dim)
        # feature_align: tensor(batch_size, max_seq_len, max_neighbor_num, num_amino_feature*2)
        feature_align = torch.cat([amino_feature_expand, neighbor_feature], dim=-1)

        # align[0] relate to radius， align_score -> (batch, max_len, max_neighbor, 1)
        # align_score: tensor(batch_size, max_seq_len, max_neighbor_num, 1)
        align_score = F.leaky_relu(self.align[0](self.dropout(feature_align)))
        align_score = align_score + softmax_mask
        # attention_weight -> torch.Size([batch, 512, max_neighbor, 1])
        # attention_weight: tensor(batch_size, max_seq_len, max_neighbor_num, 1)
        attention_weight = F.softmax(align_score, -2)
        attention_weight = attention_weight * attend_mask
        # neighbor_feature_transform: tensor(batch_size, max_seq_len, max_neighbor_num, num_amino_feature)
        neighbor_feature_transform = self.attend[0](self.dropout(neighbor_feature))
        # 邻节点每维特征*attention权重(25, 64)，然后所有ci相加得到当前节点c(1,64)
        # context -> torch.Size([batch, 512, 64])
        # context: tensor(batch_size, max_seq_len, num_amino_feature)
        context = torch.sum(torch.mul(attention_weight, neighbor_feature_transform), -2)
        context = F.elu(context)
        # context_reshape: tensor(batch_size*max_seq_len, num_amino_feature)
        context_reshape = context.view(batch_size * seq_length, embedding_dim)
        # amino_feature_reshape: tensor(batch_size*max_seq_len, num_amino_feature)
        amino_feature_reshape = amino_feature.view(batch_size * seq_length, embedding_dim)
        amino_feature_reshape = self.GRUCell[0](context_reshape, amino_feature_reshape)
        # amino_feature: tensor(batch_size, max_seq_len, num_amino_feature)
        amino_feature = amino_feature_reshape.view(batch_size, seq_length, embedding_dim)
        # do nonlinearity
        # activated_features: tensor(batch_size, max_seq_len, num_amino_feature)
        activated_features = F.relu(amino_feature)

        for d in range(self.radius - 1):
            neighbor_feature = [activated_features[i][amino_degree_list[i]] for i in range(batch_size)]
            # neighbor_feature is a list of 3D tensor, so we need to stack them into a 4D tensor first
            # neighbor_feature: tensor(batch_size, max_seq_len, max_neighbor_num, num_amino_feature)
            neighbor_feature = torch.stack(neighbor_feature, dim=0)
            amino_feature_expand = activated_features.unsqueeze(-2).expand(batch_size, seq_length, max_neighbor_num,
                                                                           embedding_dim)

            feature_align = torch.cat([amino_feature_expand, neighbor_feature], dim=-1)

            align_score = F.leaky_relu(self.align[d + 1](self.dropout(feature_align)))
            
            align_score = align_score + softmax_mask
            attention_weight = F.softmax(align_score, -2)
            
            attention_weight = attention_weight * attend_mask
            
            neighbor_feature_transform = self.attend[d + 1](self.dropout(neighbor_feature))
            
            context = torch.sum(torch.mul(attention_weight, neighbor_feature_transform), -2)
            
            context = F.elu(context)
            context_reshape = context.view(batch_size * seq_length, embedding_dim)
            
            amino_feature_reshape = self.GRUCell[d + 1](context_reshape, amino_feature_reshape)
            amino_feature = amino_feature_reshape.view(batch_size, seq_length, embedding_dim)

            # do nonlinearity
            activated_features = F.relu(amino_feature)

        # 将aa特征汇聚成蛋白序列的
        # seq_feature: tensor(batch_size, num_amino_feature)
        seq_feature = torch.sum(activated_features * amino_mask, dim=-2)  # B * feature, all atom sum = mol

        # do nonlinearity
        # activated_features_mol: tensor(batch_size, num_amino_feature)
        activated_features_mol = F.relu(seq_feature)

        seq_softmax_mask = amino_mask.clone()
        seq_softmax_mask[seq_softmax_mask == 0] = -9e8
        seq_softmax_mask[seq_softmax_mask == 1] = 0
        seq_softmax_mask = seq_softmax_mask.type(torch.cuda.FloatTensor)

        for t in range(self.T):
            seq_prediction_expand = activated_features_mol.unsqueeze(-2).expand(batch_size, seq_length, embedding_dim)
            seq_align = torch.cat([seq_prediction_expand, activated_features], dim=-1)
            seq_align_score = F.leaky_relu(self.seq_align(seq_align))
            seq_align_score = seq_align_score + seq_softmax_mask
            seq_attention_weight = F.softmax(seq_align_score, -2)
            seq_attention_weight = seq_attention_weight * amino_mask
            
            activated_features_transform = self.seq_attend(self.dropout(activated_features))
            #  aggregate embeddings of amino in a protein
            seq_context = torch.sum(torch.mul(seq_attention_weight, activated_features_transform), -2)
            
            seq_context = F.elu(seq_context)
            seq_feature = self.seq_GRUCell(seq_context, seq_feature)

            # do nonlinearity
            # activated_features_seq: tensor(batch_size, num_amino_feature)
            activated_features_seq = F.relu(seq_feature)

        return activated_features_seq, activated_features



if __name__ == '__main__':
    p = ProGAT(64, 2, 1, 0.5).cuda()
    x = ['FJIEOWQIJORGJ', 'JIFEOIOWN']
    z = p(x)
    print(z.shape)
