import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class GAT(nn.Module):
    def __init__(self,nfeat,nhid,nclass,dropout,adj_list,adj_att):
        '''
        传入的adj_list,是不同hop邻居的邻接矩阵。里面具体的内容是
        None + adj^k + adj_i^k for k in range(layers)

        adj_att是计算QK用的邻接矩阵,模型是通过1hop邻居计算QK,考不考虑self_loop通过adj_att参数表示

        
        '''
        super(GAT,self).__init__()
        self.nlayers=len(adj_list)
        self.fc2 = nn.Linear(nhid*self.nlayers,nclass)
        self.adj_list=adj_list
        self.adj_att=adj_att
        self.dropout = dropout
        self.act_fn = nn.ReLU()
        self.fc1 = nn.ModuleList([nn.Linear(nfeat,int(nhid)) for _ in range(self.nlayers)])
        self.Q = nn.ModuleList([nn.Linear(nhid,int(nhid)) for _ in range(self.nlayers-1)])
        self.K = nn.ModuleList([nn.Linear(nhid,int(nhid)) for _ in range(self.nlayers-1)])
        self.att = nn.Parameter(torch.ones(self.nlayers))
        self.sm = nn.Softmax(dim=0)

    def forward(self,x):

        mask = self.sm(self.att)
        list_out = list()
        for i in range(self.nlayers):
            tmp_out = self.fc1[i](x)
            if self.adj_list[i] is not None:
                tmp_out_att=torch.mm(self.adj_att,tmp_out)
                Q=self.Q[i-1](tmp_out_att)
                K=self.K[i-1](tmp_out_att)
                attention=torch.mm(Q,K.T)*self.adj_list[i]
                tmp_out=torch.mm(attention,tmp_out)
            tmp_out = F.normalize(tmp_out,p=2,dim=1)
            tmp_out = torch.mul(mask[i],tmp_out)
            list_out.append(tmp_out)

        final_mat = torch.cat(list_out, dim=1)
        out = self.act_fn(final_mat)
        out = F.dropout(out,self.dropout,training=self.training)
        out = self.fc2(out)


        return F.log_softmax(out, dim=1)


