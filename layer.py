import torch
from torch import nn
import math
# from dgl.nn.pytorch.conv import NNConv
import torch as th
from torch import nn
from torch.nn import init
import dgl.function as fn
from dgl.utils import expand_as_pair
from torch.nn import Identity


class NNConv(nn.Module):
    def __init__(
        self,
        in_feats,
        out_feats,
        edge_func,
        aggregator_type="mean",
        residual=False,
        bias=True,
    ):
        super(NNConv, self).__init__()
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self.edge_func = edge_func
        if aggregator_type == "sum":
            self.reducer = fn.sum
        elif aggregator_type == "mean":
            self.reducer = fn.mean
        elif aggregator_type == "max":
            self.reducer = fn.max
        else:
            raise KeyError(
                "Aggregator type {} not recognized: ".format(aggregator_type)
            )
        self._aggre_type = aggregator_type
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, out_feats, bias=False
                )
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer("res_fc", None)
        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_buffer("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        gain = init.calculate_gain("relu")
        if self.bias is not None:
            nn.init.zeros_(self.bias)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feat, efeat):
        with graph.local_scope():
            feat_src, feat_dst = expand_as_pair(feat, graph)

            # (n, d_in, 1)
            graph.srcdata["feat"]["atom"] = feat_src.unsqueeze(-1) # 在最后一个维度上增加一个维度
            # (n, d_in, d_out)
            graph.edges[('atom', 'interacts', 'atom')].data["feat"] = self.edge_func(efeat).view(
                -1, self._in_src_feats, self._out_feats
            )
            # (n, d_in, d_out)
            graph.update_all(
                fn.u_mul_e("feat", "feat", "msg"), self.reducer("msg", "neigh"), etype=('atom', 'interacts', 'atom')
            )
            rst = graph.dstdata["neigh"]["atom"].sum(dim=1)  # (n, d_out)
            # residual connection
            if self.res_fc is not None:
                rst = rst + self.res_fc(feat_dst)
            # bias
            if self.bias is not None:
                rst = rst + self.bias
            return rst


# MPNN层
class MPNNLayer(nn.Module):
    def __init__(self, hidden_dim, aggr, residual):
        super().__init__()
        self.edge_func = nn.Linear(
            in_features=hidden_dim, out_features=hidden_dim*hidden_dim, bias=False)
        self.layer = NNConv(in_feats=hidden_dim, out_feats=hidden_dim,
                            edge_func=self.edge_func, aggregator_type=aggr, residual=residual)

    def forward(self, graph, feat, efeat):
        h = self.layer(graph, feat, efeat)
        return h


# 注意力池化层
class AttentivePooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, Q, K, V, mask=None):
        Attn = torch.mm(Q, K.transpose(0, 1)) / math.sqrt(self.hidden_dim)
        if mask is not None:
            Attn = torch.masked_fill(Attn, mask, 1e-8)
        Attn = torch.softmax(Attn, dim=-1)
        return torch.mm(Attn, V)


class AddNorm(nn.Module):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


if __name__ == "__main__":
    queries = torch.normal(0, 1, (5, 10))
    keys = torch.normal(0, 1, (10, 10))
    net = AttentivePooling(k_size=10, q_size=10, hidden_dim=10)
    # print(net(queries,keys,keys))
