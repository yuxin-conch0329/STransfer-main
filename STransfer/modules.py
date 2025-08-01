import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from collections import Counter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import add_remaining_self_loops
from torch_scatter import scatter_add

from tqdm import tqdm

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Encoder Layer
class STransferEncoder(nn.Module):
    def __init__(self, input_dim, cache_name='source'):
        super(STransferEncoder,self).__init__()
        self.encoder = nn.Sequential(nn.Linear(input_dim, 64),
                                     nn.BatchNorm1d(64, momentum=0.01, eps=0.001),
                                     nn.ELU(),
                                     nn.Dropout(p=0.2),
                                     nn.Linear(64, 32),
                                     nn.BatchNorm1d(32, momentum=0.01, eps=0.001),
                                     nn.ELU(),
                                     nn.Dropout(p=0.2))
        # GCN layers
        self.ppmi1 = GCN(input_dim=32, type = 'ppmi', activation = F.relu)
        self.ppmi2 = GCN(input_dim=32, type = 'ppmi', activation = lambda x: x)
        self.ppmi3 = GCN(input_dim=32, type = 'ppmi', activation = lambda x: x)

        self.cachedgcn1 = GCN(base_model=self.ppmi1, input_dim=32, type = 'cachedgcn', activation = F.relu)
        self.cachedgcn2 = GCN(base_model=self.ppmi2, input_dim=32, type = 'cachedgcn', activation = lambda x: x)
        self.cachedgcn3 = GCN(base_model=self.ppmi3, input_dim=32, type = 'cachedgcn', activation = lambda x: x)

        # self.cachedgcn1 = GCN(input_dim=32, type = 'cachedgcn', activation = F.relu)
        # self.cachedgcn2 = GCN(input_dim=32, type = 'cachedgcn', activation = lambda x: x)
        # self.cachedgcn3 = GCN(input_dim=32, type = 'cachedgcn', activation = lambda x: x)

        self.attention = Attention(in_channels=32)
        self.cache_name = cache_name

    def forward(self, x, edge, cache_name='source'):
        embedding = self.encoder(x)   # [4221, 1000] => [4221, 32]

        # PPMI: global spatial information
        feature_p = self.ppmi1(embedding, edge, cache_name)
        mu_p = self.ppmi2(feature_p, edge, cache_name)
        logvar_p = self.ppmi3(feature_p, edge, cache_name)

        # CatchedGCN: local spatial information
        feature_a = self.cachedgcn1(embedding, edge, cache_name)
        mu_a = self.cachedgcn2(feature_a, edge, cache_name)
        logvar_a = self.cachedgcn3(feature_a, edge, cache_name)

        # Reparameterization
        std_a = torch.exp(logvar_a)
        gcn_z_a = torch.randn_like(std_a) * std_a + mu_a
        std_p = torch.exp(logvar_p)
        gcn_z_p = torch.randn_like(std_p) * std_p + mu_p

        # Attention fusion
        att_model = self.attention.to(device)
        z_f = att_model([feature_p, feature_a])
        z_g = att_model([gcn_z_p, gcn_z_a])
        z = torch.cat((z_f, z_g), dim=1)   # [4221, 64]
        return z



# GCN
class GCN(torch.nn.Module):
    def __init__(self, input_dim, gcn_hidden = 32, type="ppmi", activation = F.relu, base_model=None):
        super(GCN, self).__init__()
        self.input_dim = input_dim
        self.gcn_hidden = gcn_hidden
        self.activation = activation

        if base_model is None:
            weights = [None]
            biases = [None]
        else:
            weights = [conv_layer.weight for conv_layer in base_model.conv_layers]
            biases = [conv_layer.bias for conv_layer in base_model.conv_layers]

        self.dropout_layers = [nn.Dropout(0.2) for _ in weights]

        model_cls = CachedGCN if type == "cachedgcn" else PPMI

        self.conv_layers = nn.ModuleList(
            [model_cls(self.input_dim, self.gcn_hidden, weight=weights[0], bias=biases[0])])

    def forward(self, x, edge_index, cache_name):
        x1 = self.conv_layers[0](x, edge_index, cache_name)
        x1 = self.activation(x1)
        x1 = self.dropout_layers[0](x1)
        return x1


# CachedGCN
class CachedGCN(MessagePassing):
    def __init__(self, in_channels, out_channels, weight=None, bias=None, improved=True, use_bias=True):
        super().__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cache_dict = {}
        if weight is None:
            self.weight = Parameter(torch.Tensor(in_channels, out_channels).to(torch.float32))
            glorot(self.weight)  # initialize weights
        else:
            self.weight = weight
        if bias is None:
            if use_bias:
                self.bias = Parameter(torch.Tensor(out_channels).to(torch.float32))
            else:
                self.register_parameter('bias', None)
            zeros(self.bias)
        else:
            self.bias = bias

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight=None, improved=True,dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)
        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)  # normalization

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, cache_name="default_cache", edge_weight=None):
        x = torch.matmul(x, self.weight)
        if not cache_name in self.cache_dict:
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, self.improved, x.dtype)
            self.cache_dict[cache_name] = edge_index, norm
        else:
            edge_index, norm = self.cache_dict[cache_name]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):   # neighbour weighted aggregation
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias   # add bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


# PPMI
class PPMI(CachedGCN):
    def __init__(self, in_channels, out_channels, weight=None, bias=None, improved=True, use_bias=True,path_len=5):
        super().__init__(in_channels, out_channels, weight, bias, improved, use_bias)
        self.path_len = path_len   # random walk length

    def norm(self, edge_index, num_nodes, edge_weight=None, improved=True,dtype=None):
        adj_dict = {}  # create a dict for neighbours
        def add_edge(a, b):
            if a in adj_dict:
                neighbors = adj_dict[a]   # all neighors for a node
            else:
                neighbors = set()
                adj_dict[a] = neighbors
            if b not in neighbors:
                neighbors.add(b)

        cpu_device = torch.device("cpu")
        gpu_device = torch.device("cuda")
        for a, b in edge_index.t().detach().to(cpu_device).numpy():
            a = int(a)
            b = int(b)
            add_edge(a, b)
            add_edge(b, a)

        adj_dict = {a: list(neighbors) for a, neighbors in adj_dict.items()}
        def sample_neighbor(a):
            neighbors = adj_dict[a]
            random_index = np.random.randint(0, len(neighbors))
            return neighbors[random_index]

        def walk_norm(counter):   # counts to probability
            s = sum(counter.values())
            new_counter = Counter()
            for a, count in counter.items():
                new_counter[a] = counter[a] / s
            return new_counter

        walk_counters = {}
        for _ in tqdm(range(40)):
            for a in adj_dict:
                current_a = a
                current_path_len = np.random.randint(1, self.path_len + 1)
                for _ in range(current_path_len):
                    b = sample_neighbor(current_a)
                    if a in walk_counters:
                        walk_counter = walk_counters[a]
                    else:
                        walk_counter = Counter()
                        walk_counters[a] = walk_counter

                    walk_counter[b] += 1
                    current_a = b

        normed_walk_counters = {a: walk_norm(walk_counter) for a, walk_counter in walk_counters.items()}
        prob_sums = Counter()
        for a, normed_walk_counter in normed_walk_counters.items():
            for b, prob in normed_walk_counter.items():
                prob_sums[b] += prob
        ppmis = {}
        for a, normed_walk_counter in normed_walk_counters.items():
            for b, prob in normed_walk_counter.items():
                ppmi = np.log(prob / prob_sums[b] * len(prob_sums) / self.path_len)
                ppmis[(a, b)] = ppmi

        new_edge_index = []
        edge_weight = []
        for (a, b), ppmi in ppmis.items():
            new_edge_index.append([a, b])
            edge_weight.append(ppmi)

        edge_index = torch.tensor(new_edge_index).t().to(gpu_device)
        edge_weight = torch.tensor(edge_weight).to(gpu_device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return edge_index, (deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]).type(torch.float32)


# attention module
class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.linear = nn.Linear(in_channels, 1)

    def forward(self, inputs):
        stacked = torch.stack(inputs, dim=1)
        weights = F.softmax(self.linear(stacked), dim=1)
        outputs = torch.sum(stacked * weights, dim=1)
        return outputs


# classifier
class STransferClassifier(nn.Module):
    def __init__(self, input_dim, p_drop=0.2):
        super(STransferClassifier, self).__init__()
        self.classifier = nn.Sequential(
            ResidualBlock(input_dim, 512, p_drop),
            ResidualBlock(512, 256, p_drop),
            ResidualBlock(256, 128, p_drop),
            ResidualBlock(128, 32, p_drop),
            nn.Linear(32, 7))

    def forward(self, z):
        y_pred = self.classifier(z)
        return y_pred


class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, p_drop):
        super(ResidualBlock, self).__init__()
        self.main_block = nn.Sequential(nn.Linear(in_features, out_features),
                                        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
                                        nn.ELU(),nn.Dropout(p=p_drop))
        self.residual = nn.Linear(in_features, out_features) if in_features != out_features else nn.Identity()
    def forward(self, x):
        return self.main_block(x) + self.residual(x)


# discriminator
class Discriminator(nn.Module):
    def __init__(self, input_dims=64, output_dims=2, hidden_dims=500):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dims, hidden_dims),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),

            nn.Linear(hidden_dims, hidden_dims // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),

            nn.Linear(hidden_dims // 2, output_dims)  # 2 logits for domain classification
        )

    def forward(self, x):
        return self.model(x)

