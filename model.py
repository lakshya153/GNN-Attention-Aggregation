from dgl.nn import SAGEConv, GraphConv 
from dgl.nn.pytorch.conv import GATConv 
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
import torch

# Define a GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
        self.conv2 = SAGEConv(h_feats, h_feats, 'mean')

    def forward(self, g, in_feat, edge_weight=None):
        h = self.conv1(g, in_feat, edge_weight)
        h = F.relu(h)
        h = self.conv2(g, h, edge_weight)
        return h

# Define a GAT model
class GAT(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GAT, self).__init__()
        num_heads = 4
        self.layer1 = GATConv(in_feats, h_feats, num_heads, feat_drop=0., attn_drop=0.,
                              residual=False, allow_zero_in_degree=True)
        self.layer2 = GATConv(h_feats * num_heads, h_feats, 1, feat_drop=0., attn_drop=0.,
                              residual=False, allow_zero_in_degree=True)

    def forward(self, g, in_feat):
        h = self.layer1(g, in_feat)
        # Reshape: (num_nodes, num_heads, out_dim) -> (num_nodes, num_heads * out_dim)
        h = h.view(h.size(0), -1)
        h = F.elu(h)
        h = self.layer2(g, h)
        # Squeeze the head dim as it's = 1
        h = h.squeeze(1) 
        return h

# Define a GCN model
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(h_feats, h_feats, allow_zero_in_degree=True)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h

# Link prediction using Dot Product
class DotPredictor(nn.Module):
    def __init__(self, h_feat):
        super(DotPredictor, self).__init__()

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return g.edata['score'][:, 0]

# Link prediction using Multi-Layer Perceptron
class MLPPredictor(nn.Module):
    def __init__(self, h_feats):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features=h_feats * 2, out_features=h_feats, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=h_feats, out_features=1, bias=False),
        )

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.model(h).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']

# Link prediction using Linear layer with Normalization
class LinearPredictor(nn.Module):
    def __init__(self, h_feats):
        super(LinearPredictor, self).__init__()
        self.fc = nn.Sequential(
            nn.LayerNorm(h_feats * 2),
            nn.Linear(h_feats * 2, 1)
        )

    def apply_edges(self, edges):
        h = torch.cat([edges.src['h'], edges.dst['h']], 1)
        return {'score': self.fc(h).squeeze(1)}

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.apply_edges(self.apply_edges)
            return g.edata['score']
