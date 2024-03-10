import networkx as nx
from typing import Union
import torch
import torch.nn.functional as F
import numpy as np

class TopologicalMeasures:
    def __init__(self,graph:Union[np.ndarray,torch.Tensor]):
        if isinstance(graph,np.ndarray):
            self.graph = nx.Graph(graph)
        elif isinstance(graph,torch.Tensor):
            graph_numpy = graph.cpu().detach().numpy()
            self.graph = nx.Graph(graph_numpy)

    def compute_measures(self):
        measures = {}
        measures['degree'] = torch.FloatTensor(list(dict(self.graph.degree()).values()))
        # measures['clustering'] = torch.FloatTensor(list(nx.clustering(self.graph).values())) # removed due to slow computation
        # measures['closeness'] = torch.FloatTensor(list(nx.closeness_centrality(self.graph).values()))
        # measures['betweenness'] = torch.FloatTensor(list(nx.betweenness_centrality(self.graph).values())) # removed due to slow computation
        # measures['pagerank'] = torch.FloatTensor(list(nx.pagerank(self.graph).values()))
        # measures['eigenvector'] = torch.FloatTensor(list(nx.eigenvector_centrality(self.graph).values()))
        return measures

def compute_topological_measures(data:np.ndarray, debug=False):
    index_to_measure = {}
    for index, graph in enumerate(data):
        index_to_measure[index] = TopologicalMeasures(graph).compute_measures()
        if debug and index % 10 == 0:
            print(f'Computed measures for {index} graphs')
    return index_to_measure

def compute_degree_nonzero(adj):
    return torch.count_nonzero(adj, axis=0)

def compute_degree_sum(adj):
    return torch.sum(adj, axis=0)

def compute_topological_MAE_loss(graph1,graph2:Union[np.ndarray,torch.Tensor]):
    # if precomputed_g1:
    #     measures1 = graph1
    # else:
    #     measures1 = TopologicalMeasures(graph1).compute_measures()
    # measures2 = TopologicalMeasures(graph2).compute_measures()
    # loss = 0
    # # compute MAE for each measure

    # for measure in measures1:
    #     loss += F.l1_loss(measures1[measure], measures2[measure])
    # loss = loss/len(measures1)

    return F.l1_loss(compute_degree_sum(graph1), compute_degree_sum(graph2))
