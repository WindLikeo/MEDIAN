import math
import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super(MultiHeadAttention, self).__init__()

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, graph):
        nodes_q = self.query(graph)
        nodes_k = self.key(graph)
        nodes_v = self.value(graph)

        nodes_q_t = self.transpose_for_scores(nodes_q)
        nodes_k_t = self.transpose_for_scores(nodes_k)
        nodes_v_t = self.transpose_for_scores(nodes_v)

        scores = torch.matmul(nodes_q_t, nodes_k_t.transpose(-1, -2))
        scores = scores / math.sqrt(self.attention_head_size)

        probs = nn.Softmax(dim=-1)(scores)

        probs = self.dropout(probs)

        new_nodes = torch.matmul(probs, nodes_v_t)
        nnew_nodes = new_nodes.permute(0, 2, 1, 3).contiguous()
        new_nodes_shape = new_nodes.size()[:-2] + (self.all_head_size,)
        new_nodes = new_nodes.view(*new_nodes_shape)
        return new_nodes


class GATLayer(nn.Module):
    def __init__(self, config):
        super(GATLayer, self).__init__()
        self.mha = MultiHeadAttention(config)

        self.in_fc = nn.Linear(config.hidden_size, config.hidden_size)
        self.in_bn = nn.BatchNorm1d(config.hidden_size)
        self.dropout_in = nn.Dropout(config.hidden_dropout_prob)

        self.intermediate_fc = nn.Linear(config.hidden_size, config.hidden_size)

        self.out_fc = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_bn = nn.BatchNorm1d(config.hidden_size)
        self.dropout_out = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_graph):
        attention_output = self.mha(input_graph) # multi-head attention
        attention_output = self.in_fc(attention_output)
        attention_output = self.dropout_in(attention_output)
        attention_output = self.in_bn((attention_output + input_graph).permute(0, 2, 1)).permute(0, 2, 1)
        intermediate_output = self.intermediate_fc(attention_output)
        intermediate_output = F.relu(intermediate_output)
        intermediate_output = self.out_fc(intermediate_output)
        intermediate_output = self.dropout_out(intermediate_output)

        graph = self.out_bn((intermediate_output + attention_output).permute(0, 2, 1)).permute(0, 2, 1)

        return graph
