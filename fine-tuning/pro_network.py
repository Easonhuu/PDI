"""
Protein sequence and structure are parallel
Sequence pretrained model is from modified Transformer by 20w
"""
import logging
import torch
import torch.nn as nn
from finetune_args import args
from pro_structure_network import ProGAT
from pro_sequence_network import ProTransformer_20w

logger = logging.getLogger(__name__)

class ProNetwork(nn.Module):
    def __init__(self, p_dropout, pro_seq_dim, pro_struc_dim, pro_dim):
        super(ProNetwork, self).__init__()
        self.model_transformer = ProTransformer_20w

        # graph model
        self.model_graph = ProGAT(args.nonspecial_vocab_size, pro_struc_dim, args.radius, args.T, p_dropout)

        self.pro_dense = nn.Sequential(nn.Dropout(p_dropout),
                                       nn.Linear(pro_seq_dim + pro_struc_dim, pro_dim),
                                       )

    def forward(self, token_inputs, attention_mask, token_type_ids, amino_inputs, degree_inputs, amino_mask_inputs):
        # batch = tuple(t.to(args.device) for t in batch)
        # input_ids, input_mask, segment_ids, label_ids = batch
        # # masked_lm_loss
        # outputs = model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask, masked_lm_labels=label_ids)

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
        # outputs = (prediction_scores,) + outputs[1:]  # Add hidden states and attention if they are here
        """
        masked_lm_loss : tensor, [gpu1_loss, gpu2_loss, gpu3_loss, gpu4_loss], CrossEntropyLoss遇到-100会在计算loss时忽略
        outputs : tuple, (masked_lm_loss, prediction_scores)
        """
        # if masked_lm_labels is not None:
        #     loss_fct = CrossEntropyLoss()  # -100 index = padding token
        #     masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
        #     outputs = (masked_lm_loss,) + outputs
        """
        outputs : tuple, (prediction_scores, cls_score)
        """
        outputs = self.model_transformer(input_ids=token_inputs, attention_mask=attention_mask, token_type_ids=token_type_ids)
        "cls_emb : tensor(batch_size, hidden_size)"
        cls_emb = outputs[-1]
        """
        graph_emb : tensor(batch_size, graph_dim)
        graph_node_emb : tensor(batch_size, sequence_length, graph_dim)
        """

        graph_emb, graph_node_emb = self.model_graph(amino_inputs, degree_inputs, amino_mask_inputs)

        pro_emb = torch.cat([cls_emb, graph_emb], dim=1)

        pro_emb = self.pro_dense(pro_emb)

        return pro_emb

