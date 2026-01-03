from torch.nn import Linear
from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter
import numpy as np

#Check for GPU
use_gpu = True
if use_gpu:
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
else:
	device = torch.device('cpu')

class GCN_Anomaly(torch.nn.Module):
	#this is the basic model
	def __init__(self, nfeat, nhid, nout, dropout):
		"""
		:param nfeat : the number of input features
		:param nhid :  the number of the hidden neurons
		:param nout :  the number of the output neurons
		:param dropout : the dropout probability
		"""
		super(GCN, self).__init__()
		self.p = dropout
		self.conv1 = GCNConv(nfeat, nhid, add_self_loops=True) #gcn layer 1 
		self.conv2 = GCNConv(nhid,nout, add_self_loops=True) #gcn layer 2
	
	def forward(self, x, edge_index):
		x = self.conv1(x, edge_index)
		x = F.gelu(x)
		x = F.dropout(x, training = self.training,p=self.p)
		x = self.conv2(x, edge_index)
		return x
