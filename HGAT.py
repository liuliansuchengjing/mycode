# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 22:30:16 2021

@author: Ling Sun
"""

import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from layer import HGATLayer
from torch_geometric.nn import GCNConv
import torch.nn.init as init
import Constants
from TransformerBlock import TransformerBlock
from torch.autograd import Variable


class DJconv(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(DJconv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        # use xavier initialization
        nn.init.xavier_uniform_(self.weight)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))

    def forward(self, H, U):
        Hu = torch.concat((H, torch.matmul(H, torch.matmul(H.t(), H))), dim=1)
        Hu = torch.where(Hu >= 0.5, 1., 0.)
        Hu = Hu.to(torch.float32)

        Du_v = torch.sum(Hu, dim=1)
        mask = Du_v.nonzero()
        for i in mask:
            Du_v[i] = 1 / torch.sqrt(Du_v[i])
        Du_v = torch.diag(Du_v)
        Du_v = Du_v.to(torch.float32)

        Du_e = torch.sum(Hu, dim=0)
        mask = Du_e.nonzero()
        for i in mask:
            Du_e[i] = 1 / torch.sqrt(Du_e[i])
        Du_e = torch.diag(Du_e)
        Du_e = Du_e.to(torch.float32)

        U = U.to(torch.float32)
        M_u = torch.linalg.multi_dot([Du_v.cuda(), U]) + U

        U_out = torch.matmul(M_u.cuda(), self.weight.cuda()) + self.bias.cuda()

        return U_out


def get_previous_user_mask(seq, user_size):
    ''' Mask previous activated users.'''
    assert seq.dim() == 2
    prev_shape = (seq.size(0), seq.size(1), seq.size(1))
    seqs = seq.repeat(1, 1, seq.size(1)).view(seq.size(0), seq.size(1), seq.size(1))
    previous_mask = np.tril(np.ones(prev_shape)).astype('float32')
    previous_mask = torch.from_numpy(previous_mask)
    if seq.is_cuda:
        previous_mask = previous_mask.cuda()
    masked_seq = previous_mask * seqs.data.float()

    # force the 0th dimension (PAD) to be masked
    PAD_tmp = torch.zeros(seq.size(0), seq.size(1), 1)
    if seq.is_cuda:
        PAD_tmp = PAD_tmp.cuda()
    masked_seq = torch.cat([masked_seq, PAD_tmp], dim=2)
    ans_tmp = torch.zeros(seq.size(0), seq.size(1), user_size)
    if seq.is_cuda:
        ans_tmp = ans_tmp.cuda()
    masked_seq = ans_tmp.scatter_(2, masked_seq.long(), float(-1000))
    masked_seq = Variable(masked_seq, requires_grad=False)
    # print("masked_seq ",masked_seq.size())
    return masked_seq.cuda()


# Fusion gate
class Fusion(nn.Module):
    def __init__(self, input_size, out=1, dropout=0.2):
        super(Fusion, self).__init__()
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, out)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def forward(self, hidden, dy_emb):
        emb = torch.cat([hidden.unsqueeze(dim=0), dy_emb.unsqueeze(dim=0)], dim=0)
        emb_score = F.softmax(self.linear2(torch.tanh(self.linear1(emb))), dim=0)
        emb_score = self.dropout(emb_score)
        out = torch.sum(emb_score * emb, dim=0)
        return out


'''Learn friendship network'''


class GraphNN(nn.Module):
    def __init__(self, ntoken, ninp, dropout=0.5, is_norm=True):
        super(GraphNN, self).__init__()
        self.embedding = nn.Embedding(ntoken, ninp, padding_idx=0)
        # in:inp,out:nip*2
        self.gnn1 = GCNConv(ninp, ninp * 2)
        self.gnn2 = GCNConv(ninp * 2, ninp)
        self.is_norm = is_norm

        self.dropout = nn.Dropout(dropout)
        if self.is_norm:
            self.batch_norm = torch.nn.BatchNorm1d(ninp)
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.embedding.weight)

    def forward(self, graph):
        graph_edge_index = graph.edge_index.cuda()
        graph_x_embeddings = self.gnn1(self.embedding.weight, graph_edge_index)
        graph_x_embeddings = self.dropout(graph_x_embeddings)
        graph_output = self.gnn2(graph_x_embeddings, graph_edge_index)
        if self.is_norm:
            graph_output = self.batch_norm(graph_output)
        # print(graph_output.shape)
        return graph_output.cuda()


'''Learn diffusion network'''


class HGNN_ATT(nn.Module):
    def __init__(self, input_size, n_hid, output_size, dropout=0.3, is_norm=True):
        super(HGNN_ATT, self).__init__()
        self.dropout = dropout
        self.is_norm = is_norm
        if self.is_norm:
            self.batch_norm1 = torch.nn.BatchNorm1d(output_size)
        self.gat1 = HGATLayer(input_size, output_size, dropout=self.dropout, transfer=False, concat=True, edge=True)
        self.fus1 = Fusion(output_size)
        self.hgnn = DJconv(64, 64)

    def forward(self, x, hypergraph_list):
        root_emb = F.embedding(hypergraph_list[1].cuda(), x)

        hypergraph_list = hypergraph_list[0]
        embedding_list = {}
        for sub_key in hypergraph_list.keys():
            sub_graph = hypergraph_list[sub_key]
            sub_node_embed, sub_edge_embed = self.gat1(x, sub_graph.cuda(), root_emb)
            sub_node_embed = self.hgnn(sub_graph, x)
            sub_node_embed = F.dropout(sub_node_embed, self.dropout, training=self.training)

            if self.is_norm:
                sub_node_embed = self.batch_norm1(sub_node_embed)
                sub_edge_embed = self.batch_norm1(sub_edge_embed)

            x = self.fus1(x, sub_node_embed)
            embedding_list[sub_key] = [x.cpu(), sub_edge_embed.cpu()]

        return embedding_list


class MLPReadout(nn.Module):
    def __init__(self, in_dim, out_dim, act):
        """
        out_dim: the final prediction dim, usually 1
        act: the final activation, if rating then None, if CTR then sigmoid
        """
        super(MLPReadout, self).__init__()
        self.layer1 = nn.Linear(in_dim, out_dim)
        self.act = nn.ReLU()
        self.out_act = act

    def forward(self, x):
        ret = self.layer1(x)
        return ret


class MSHGAT(nn.Module):
    def __init__(self, opt, dropout=0.3):
        super(MSHGAT, self).__init__()
        self.hidden_size = opt.d_word_vec
        self.n_node = opt.user_size
        self.pos_dim = 8
        self.dropout = nn.Dropout(dropout)
        self.initial_feature = opt.initialFeatureSize

        self.hgnn = HGNN_ATT(self.initial_feature, self.hidden_size * 2, self.hidden_size, dropout=dropout)
        self.gnn = GraphNN(self.n_node, self.initial_feature, dropout=dropout)
        self.fus = Fusion(self.hidden_size + self.pos_dim)
        self.fus2 = Fusion(self.hidden_size)
        self.pos_embedding = nn.Embedding(1000, self.pos_dim)
        self.decoder_attention1 = TransformerBlock(input_size=self.hidden_size + self.pos_dim, n_heads=8)
        self.decoder_attention2 = TransformerBlock(input_size=self.hidden_size + self.pos_dim, n_heads=8)

        self.linear2 = nn.Linear(self.hidden_size + self.pos_dim, self.n_node)
        self.embedding = nn.Embedding(self.n_node, self.initial_feature, padding_idx=0)
        self.reset_parameters()
        self.readout = MLPReadout(self.hidden_size, self.n_node, None)

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def pred(self, pred_logits):
        predictions = self.readout(pred_logits)
        return predictions

    def forward(self, input, input_timestamp, input_idx, graph, hypergraph_list):

        input = input[:, :-1]
        # print(input)
        # print(input_timestamp)
        input_timestamp = input_timestamp[:, :-1]
        hidden = self.dropout(self.gnn(graph))
        memory_emb_list = self.hgnn(hidden, hypergraph_list)
        # print(sorted(memory_emb_list.keys()))

        mask = (input == Constants.PAD)
        batch_t = torch.arange(input.size(1)).expand(input.size()).cuda()
        order_embed = self.dropout(self.pos_embedding(batch_t))
        batch_size, max_len = input.size()

        zero_vec = torch.zeros_like(input)
        dyemb = torch.zeros(batch_size, max_len, self.hidden_size).cuda()
        cas_emb = torch.zeros(batch_size, max_len, self.hidden_size).cuda()

        for ind, time in enumerate(sorted(memory_emb_list.keys())):
            if ind == 0:
                sub_input = torch.where(input_timestamp <= time, input, zero_vec)
                sub_emb = F.embedding(sub_input.cuda(), hidden.cuda())
                temp = sub_input == 0
                sub_cas = sub_emb.clone()
            else:
                cur = torch.where(input_timestamp <= time, input, zero_vec) - sub_input
                temp = cur == 0

                sub_cas = torch.zeros_like(cur)
                sub_cas[~temp] = 1
                sub_cas = torch.einsum('ij,i->ij', sub_cas, input_idx)
                sub_cas = F.embedding(sub_cas.cuda(), list(memory_emb_list.values())[ind - 1][1].cuda())
                sub_emb = F.embedding(cur.cuda(), list(memory_emb_list.values())[ind - 1][0].cuda())
                sub_input = cur + sub_input

            sub_cas[temp] = 0
            sub_emb[temp] = 0
            dyemb += sub_emb
            cas_emb += sub_cas

            if ind == len(memory_emb_list) - 1:
                sub_input = input - sub_input
                temp = sub_input == 0

                sub_cas = torch.zeros_like(sub_input)
                sub_cas[~temp] = 1
                sub_cas = torch.einsum('ij,i->ij', sub_cas, input_idx)
                sub_cas = F.embedding(sub_cas.cuda(), list(memory_emb_list.values())[ind - 1][1].cuda())
                sub_cas[temp] = 0
                sub_emb = F.embedding(sub_input.cuda(), list(memory_emb_list.values())[ind][0].cuda())
                sub_emb[temp] = 0

                dyemb += sub_emb
                cas_emb += sub_cas
        pred = self.pred(dyemb)
        return pred.view(-1, pred.size(-1))

