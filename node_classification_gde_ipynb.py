import math
import numpy as np
import scipy.sparse as sp
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torchgde import GCN, GCNLayer, ODEBlock, GDEFunc, PerformanceContainer, accuracy
import dgl
import dgl.data
import networkx as nx

# Data preparation
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# seed for repeatability
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(0)
np.random.seed(0)

# dgl offers convenient access to GNN benchmark datasets via `dgl.data`...
# other standard datasets (e.g. Citeseer / Pubmed) are also accessible via the dgl.data
# API. The rest of the notebook is compatible with Cora / Citeseer / Pubmed with minimal
# modification required.
data = dgl.data.CoraDataset()
# Cora is a node-classification datasets with 2708 nodes
X = torch.FloatTensor(data.features).to(device)
Y = torch.LongTensor(data.labels).to(device)

# In transductive semi-supervised node classification tasks on graphs, the model has access to all
# node features but only a masked subset of the labels
train_mask = torch.BoolTensor(data.train_mask)
val_mask = torch.BoolTensor(data.val_mask)
test_mask = torch.BoolTensor(data.test_mask)

num_feats = X.shape[1]
n_classes = data.num_labels

# 140 training samples, 300 validation, 1000 test
n_classes, train_mask.sum().item(), val_mask.sum().item(),test_mask.sum().item()
# add self-edge for each node
g = data.graph
g.remove_edges_from(nx.selfloop_edges(g))
g.add_edges_from(zip(g.nodes(), g.nodes()))
g = dgl.DGLGraph(g)
edges = g.edges()
n_edges = g.number_of_edges()

# compute diagonal of normalization matrix D according to standard formula
degs = g.in_degrees().float()
norm = torch.pow(degs, -0.5)
norm[torch.isinf(norm)] = 0
# add to dgl.Graph in order for the norm to be accessible at training time
g.ndata['norm'] = norm.unsqueeze(1).to(device)

# Graph Neural Differential Equations (GDEs)
# for GCDEs, the ODEFunc is specified by two GCN layers. Softplus is used as activation. Smoother activations
# have been observed to help avoid numerical instability and reduce stiffness of the ODE described
# by a repeated call to the ODEFunc. High dropout improves performance on transductive node classification
# tasks due to their small training sets. GDEs can take advantage of this property due to their 'deeper'
# computational graph. NOTE: too much dropout increases stiffness and therefore NFEs
gnn = nn.Sequential(GCNLayer(g=g, in_feats=64, out_feats=64, activation=nn.Softplus(), dropout=0.9),
                  GCNLayer(g=g, in_feats=64, out_feats=64, activation=None, dropout=0.9)
                 ).to(device)

gdefunc = GDEFunc(gnn)

# dopri5 is an adaptive step solver and will call `gdefunc` several times to ensure correctness up to pre-specified
# tolerance levels. rk4 will call the func 4 times. As suggested in the original Neural ODE paper and as observed during internal tests, lower tolerances
# are sufficient for classification tasks.
gde = ODEBlock(odefunc=gdefunc, method='rk4', atol=1e-3, rtol=1e-4, adjoint=False).to(device)


m = nn.Sequential(GCNLayer(g=g, in_feats=num_feats, out_feats=64, activation=F.relu, dropout=0.4),
                  gde,
                  GCNLayer(g=g, in_feats=64, out_feats=n_classes, activation=None, dropout=0.)
                  ).to(device)

opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
logger = PerformanceContainer(data={'train_loss': [], 'train_accuracy': [],
                                    'test_loss': [], 'test_accuracy': [],
                                    'forward_time': [], 'backward_time': [],
                                    'nfe': []})
steps = 3000
verbose_step = 150
num_grad_steps = 0

for i in range(steps):  # looping over epochs
    m.train()
    start_time = time.time()

    outputs = m(X)
    f_time = time.time() - start_time

    nfe = m._modules['1'].odefunc.nfe

    y_pred = outputs

    loss = criterion(y_pred[train_mask], Y[train_mask])
    opt.zero_grad()

    start_time = time.time()
    loss.backward()
    b_time = time.time() - start_time

    opt.step()
    num_grad_steps += 1

    with torch.no_grad():
        m.eval()

        # calculating outputs again with zeroed dropout
        y_pred = m(X)
        m._modules['1'].odefunc.nfe = 0

        train_loss = loss.item()
        train_acc = accuracy(y_pred[train_mask], Y[train_mask]).item()
        test_acc = accuracy(y_pred[test_mask], Y[test_mask]).item()
        test_loss = criterion(y_pred[test_mask], Y[test_mask]).item()
        logger.deep_update(logger.data, dict(train_loss=[train_loss], train_accuracy=[train_acc],
                                             test_loss=[test_loss], test_accuracy=[test_acc], nfe=[nfe],
                                             forward_time=[f_time],
                                             backward_time=[b_time]))

    if num_grad_steps % verbose_step == 0:
        print('[{}], Loss: {:3.3f}, Train Accuracy: {:3.3f}, Test Accuracy: {:3.3f}, NFE: {}'.format(num_grad_steps,
                                                                                                     train_loss,
                                                                                                     train_acc,
                                                                                                     test_acc,
                                                                                                     nfe))

plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
plt.plot(logger.data['train_loss'])
plt.plot(logger.data['test_loss'])
plt.legend(['Train loss', 'Test loss'])
plt.subplot(2,2,2)
plt.plot(logger.data['train_accuracy'])
plt.plot(logger.data['test_accuracy'])
plt.legend(['Train accuracy', 'Test accuracy'])
plt.subplot(2,2,3)
plt.plot(logger.data['forward_time'])
plt.plot(logger.data['backward_time'])
plt.legend(['Forward time', 'Backward time'])
plt.subplot(2,2,4)
plt.plot(logger.data['nfe'], marker='o', linewidth=0.1, markersize=1)
plt.legend(['NFE'])

# # GCN
# loggers = []
#
# for n_layers in range(2, 9, 2):
#
#     gcn = GCN(num_layers=n_layers, g=g, in_feats=1433, hidden_feats=64, out_feats=7, activation=F.relu, dropout=0.9).to(
#         device)
#     opt = torch.optim.Adam(gcn.parameters(), lr=1e-3, weight_decay=5e-4)
#     criterion = torch.nn.CrossEntropyLoss()
#     num_grad_steps = 3000
#
#     gcn_logger = PerformanceContainer(data={'train_loss': [], 'train_accuracy': [],
#                                             'test_loss': [], 'test_accuracy': [],
#                                             'forward_time': [], 'backward_time': []}
#                                       )
#
#     for i in range(num_grad_steps):  # looping over epochs
#         start_time = time.time()
#         outputs = gcn(X)
#         f_time = time.time() - start_time
#
#         y_pred = outputs
#
#         loss = criterion(y_pred[train_mask], Y[train_mask])
#         opt.zero_grad()
#
#         start_time = time.time()
#         loss.backward()
#         b_time = time.time() - start_time
#
#         opt.step()
#         num_grad_steps += 1
#
#         with torch.no_grad():
#             gcn.eval()
#             outputs = gcn(X)
#             y_pred = outputs
#
#             train_loss = loss.item()
#             train_acc = accuracy(y_pred[train_mask], Y[train_mask]).item()
#             test_acc = accuracy(y_pred[test_mask], Y[test_mask]).item()
#             test_loss = criterion(y_pred[test_mask], Y[test_mask]).item()
#
#             gcn_logger.deep_update(gcn_logger.data, dict(train_loss=[train_loss], train_accuracy=[train_acc],
#                                                          test_loss=[test_loss], test_accuracy=[test_acc],
#                                                          forward_time=[f_time],
#                                                          backward_time=[b_time]))
#
#     loggers.append((gcn_logger, n_layers))
#
# # comparison plot
# mean_nfe = int(np.array(logger.data['nfe']).mean())
# layer_range = range(2, 9, 2)
#
# plt.figure(figsize=(16,8))
# plt.subplot(2,2,1)
# plt.plot(logger.data['train_loss'])
# for logger_gcn, l in loggers: plt.plot(logger_gcn.data['train_loss'])
# plt.title('Training loss')
# plt.legend([f'GCDE{mean_nfe}'] + [f'GCN{i}' for i in layer_range])
# plt.ylim(0,2)
#
# plt.subplot(2,2,2)
# plt.plot(logger.data['test_loss'])
# for logger_gcn, l in loggers: plt.plot(logger_gcn.data['test_loss'])
# plt.title('Test loss')
# plt.legend([f'GCDE{mean_nfe}'] + [f'GCN{i}' for i in layer_range])
# plt.ylim(0,3)
#
# plt.subplot(2,2,3)
# plt.plot(logger.data['train_accuracy'])
# for logger_gcn, l in loggers: plt.plot(logger_gcn.data['train_accuracy'])
# plt.title('Train accuracy')
# plt.legend([f'GCDE{mean_nfe}'] + [f'GCN{i}' for i in layer_range])
#
# plt.subplot(2,2,4)
# plt.plot(logger.data['test_accuracy'])
# for logger_gcn, l in loggers: plt.plot(logger_gcn.data['test_accuracy'])
# plt.title('Test accuracy')
# plt.legend([f'GCDE{mean_nfe}'] + [f'GCN{i}' for i in layer_range]);
#
# # time complexity
# ftime = logger.data['forward_time']
# btime = logger.data['backward_time']
# plt.figure(figsize=(16,4))
# plt.subplot(1,2,1)
# plt.plot(ftime)
# for logger_gcn, l in loggers: plt.plot(logger_gcn.data['forward_time'])
# plt.title('Forward time (s)')
# plt.legend(['GCDE'] + [f'GCN{i}' for i in layer_range])
# plt.ylim(0, 0.006)
#
# plt.subplot(1,2,2)
# plt.plot(btime)
# for logger_gcn, l in loggers: plt.plot(logger_gcn.data['backward_time'])
# plt.title('Backward time (s)')
# plt.legend(['GCDE'] + [f'GCN{i}' for i in layer_range])