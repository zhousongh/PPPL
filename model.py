import torch
from torch import nn
from torch.nn.parameter import Parameter
from layer import MPNNLayer, AttentivePooling
from dataset.dataset import FlexDataset, FlexDataLoader
import dgl
from rdkit import Chem
from dgl.nn.pytorch import GraphConv

 
# 学习分子全图表示
class MolGNN(nn.Module):
    def __init__(self, hidden_dim, aggr, residual):
        super().__init__()
        self.layer1 = MPNNLayer(hidden_dim=hidden_dim,
                                aggr=aggr, residual=residual)
        self.layer2 = MPNNLayer(hidden_dim=hidden_dim,
                                aggr=aggr, residual=residual)

    def forward(self, graph, feat, efeat):
        h1 = self.layer1(graph, feat, efeat)
        h2 = self.layer2(graph, h1, efeat)
        return h2


# Prompt生成器
class PromptGenerator(nn.Module):
    def __init__(self, task_num, context_num, hidden_dim, num_heads, device):
        super().__init__()
        self.task_num = task_num
        self.context_num = context_num
        self.hidden_dim = hidden_dim
        # 初始化Context关联图
        src, dst = self.complete_graph()
        self.relation_graph = dgl.graph((src, dst), device=device)
        self.relation_graph.ndata['feat'] = torch.eye(n=task_num).to(device)
        self.CGNN = GraphConv(in_feats=task_num, out_feats=hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads)

    def complete_graph(self):
        src, dst = [], []
        for u in range(self.task_num):
            for v in range(self.task_num):
                if u != v:
                    src.append(u)
                    dst.append(v)
        return torch.tensor(src, dtype=torch.int32), torch.tensor(dst, dtype=torch.int32)

    def forward(self, G_h):
        contexts = self.CGNN(self.relation_graph,
                             self.relation_graph.ndata['feat'])
        prompts, attn_w = self.attention(G_h, contexts, contexts)
        return prompts

# 分层学习分子表示
class HGNN(nn.Module):
    def __init__(self, in_node_features, in_edge_features, hidden_dim, aggr, residual):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.node_encoder = nn.Sequential(
            nn.Linear(in_node_features, hidden_dim), nn.ReLU())
        self.edge_encoder = nn.Sequential(
            nn.Linear(in_edge_features, hidden_dim), nn.ReLU())
        self.GlobalEncoder = MolGNN(hidden_dim=hidden_dim,
                                    aggr=aggr, residual=residual)
        self.SubEncoder = nn.Sequential(
            nn.Linear(in_node_features, hidden_dim), nn.ReLU())
        self.attn_pool = AttentivePooling(hidden_dim=hidden_dim)
        self.feature_resize = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim), nn.ReLU())

    def forward(self, G):
        '''批量处理全图'''
        G.nodes["atom"].data['feat'] = self.node_encoder(
            G.nodes["atom"].data['feat'])
        G.edges[('atom', 'interacts', 'atom')].data['feat'] = self.edge_encoder(
            G.edges[('atom', 'interacts', 'atom')].data['feat'])

        G.nodes["atom"].data['feat'] = self.GlobalEncoder(
            G, G.nodes["atom"].data['feat'], G.edges[('atom', 'interacts', 'atom')].data['feat'])

        '''批量处理子图'''
        G.nodes["func_group"].data['feat'] = self.SubEncoder(
            G.nodes["func_group"].data['feat'])

        '''将子图特征做注意力汇聚'''
        G_list = dgl.unbatch(G)
        for i in range(len(G_list)):
            subG_pooled = self.attn_pool(
                G_list[i].nodes["atom"].data['feat'], G_list[i].nodes["func_group"].data['feat'], G_list[i].nodes["func_group"].data['feat'])
            G_feat = G_list[i].nodes["atom"].data['feat']
            G_list[i].nodes["atom"].data['feat'] = torch.cat(
                (G_feat, subG_pooled), dim=-1)

        G_batch = dgl.batch(G_list)
        # 直接读出
        feature = dgl.readout_nodes(G_batch, 'feat', op='mean', ntype='atom')
        # G_batch.ndata['feat'] = self.feature_resize(G_batch.ndata['feat'])
        # return G_batch
        return self.feature_resize(feature)


class Framework(nn.Module):
    def __init__(self, in_node_features, in_edge_features, hidden_dim, aggr, residual, context_num, num_heads, task_num, predictor, device):
        super().__init__()
        self.mol_learner = HGNN(
            in_node_features, in_edge_features, hidden_dim, aggr, residual)
        self.prompt_generator = PromptGenerator(
            context_num=context_num, hidden_dim=hidden_dim, num_heads=num_heads, task_num=task_num, device=device)
        self.resize = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim), nn.ReLU())
        self.predictor = predictor
        self.reset_parameters()

    def forward(self, G_batch):
        h = self.mol_learner(G_batch)
        prompt = self.prompt_generator(h)
        feature = self.resize(torch.cat((h, prompt), dim=-1))
        return self.predictor(feature)

    def reset_parameters(self):
        for param in self.parameters():
            nn.init.normal_(param, mean=0, std=0.1)


if __name__ == "__main__":
    dataset = FlexDataset(dataset_name='tox21',
                          root=r'/mnt/klj/PPPL/dataset/data', device=torch.device("cuda"))
    loader = FlexDataLoader(
        dataset=dataset, device=dataset.device, batch_size=256, shuffle=True)

    net = Framework(in_node_features=100, in_edge_features=7,
                    hidden_dim=64, aggr='sum', residual=False, context_num=64, num_heads=4, task_num=12, predictor=nn.Sequential(
                        nn.Linear(64, dataset[0]["label"].numel()), nn.Sigmoid()), device=dataset.device).cuda()
    net.reset_parameters()
    G_batch, label_batch = next(iter(loader))
    y = net(G_batch=G_batch)

    print(y.shape)
    loss = nn.BCELoss(reduction="sum")
    l = loss(y.view(label_batch.shape), label_batch)
    l.backward()
    # print(dict(net.named_parameters()))
