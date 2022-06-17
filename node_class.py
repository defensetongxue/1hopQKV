from __future__ import division
from __future__ import print_function
import time
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from process import *
from utils import *
from model import *
import uuid
import pickle

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.') #Default seed same as GCNII
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--layer', type=int, default=3, help='Number of layers.')
parser.add_argument('--hidden', type=int, default=64, help='hidden dimensions.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--data', default='cora', help='dateset')
parser.add_argument('--dev', type=int, default=0, help='device id')
parser.add_argument('--layer_norm',type=int, default=1, help='layer norm')
parser.add_argument('--w_att',type=float, default=0.0005, help='Weight decay scalar')
parser.add_argument('--w_fc2',type=float, default=0.0005, help='Weight decay layer-2')
parser.add_argument('--w_fc1',type=float, default=0.0005, help='Weight decay layer-1')
parser.add_argument('--lr_fc',type=float, default=0.02, help='Learning rate 2 fully connected layers')
parser.add_argument('--lr_att',type=float, default=0.02, help='Learning rate Scalar')
parser.add_argument('--feat_type',type=int, default=3, help='Type of features to be used,0:用adj_i^k 1:adj^k 3 都用')
parser.add_argument('--self_loop',type=int, default=0, help='Type of features to be used')
parser.add_argument('--w_t',type=int, default=1, help='weight_decay for Q,K times with fc1')
args = parser.parse_args()
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
num_layer = args.layer
feat_type = args.feat_type

layer_norm = bool(int(args.layer_norm))
print("==========================")
print(f"Dataset: {args.data}")
print(f"Dropout:{args.dropout}, layer_norm: {layer_norm}")
print(f"w_att:{args.w_att}, w_fc2:{args.w_fc2}, w_fc1:{args.w_fc1}, lr_fc:{args.lr_fc}, lr_att:{args.lr_att}")


cudaid = "cuda:"+str(args.dev)
device = torch.device(cudaid)
checkpt_file = 'pretrained/'+uuid.uuid4().hex+'.pt'


def train_step(model,optimizer,labels,features,idx_train):
    model.train()
    optimizer.zero_grad()
    output = model(features)
    acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    loss_train.backward()
    optimizer.step()
    return loss_train.item(),acc_train.item()


def validate_step(model,labels,features,idx_val):
    model.eval()
    with torch.no_grad():
        output = model(features)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val.item(),acc_val.item()

def test_step(model,labels,features,idx_test):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(features)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        #print(mask_val)
        return loss_test.item(),acc_test.item()


def train(datastr,splitstr):
    adj, adj_i, features, labels, idx_train, idx_val, idx_test, num_features, num_labels = full_load_data(datastr,splitstr)
    features = features.to(device)

    adj = adj.to(device) # adj是邻接矩阵，归一化的方法是，D^(-1/2)ADD^(-1/2),D是度数矩阵。adj和adj_i的区别是是否考虑自己是自己的邻居。
    adj_i = adj_i.to(device)
    adj_list = []
    E=torch.eye(adj.shape[0]).to(device)
    adj_list.append(None)
    no_loop_mat = E
    loop_mat = E

    for ii in range(args.layer):
        no_loop_mat = torch.spmm(adj, no_loop_mat)
        loop_mat = torch.spmm(adj_i, loop_mat)
        adj_list.append(no_loop_mat)
        adj_list.append(loop_mat)
    # Select X and self-looped features 
    if feat_type == 0:
        # 有的数据集，考虑自己，也就是khop邻居包括自己的情况效果更好，这一点具体为什么还没有一个合理的解释。
        # 绝大多数的数据集考虑adj的时候，有self-loop，也就是自己是自己的邻居会更好
        select_idx = [0] + [2*ll for ll in range(1,num_layer+1)]
        adj_list = [adj_list[ll] for ll in select_idx]
        raise
    #Select X and no-loop features
    elif feat_type == 1:
        select_idx = [0] + [2*ll-1 for ll in range(1,num_layer+1)]
        adj_list = [adj_list[ll] for ll in select_idx]

    if args.self_loop==1:
        adj_att=adj_i
    else:
        adj_att=adj
    #Otherwise all hop features are selected
    model = GAT(nfeat=num_features,
                nhid=args.hidden,
                nclass=num_labels,
                dropout=args.dropout,
                adj_list=adj_list,
                adj_att=adj_att).to(device)


    optimizer_sett = [
        {'params': model.fc2.parameters(), 'weight_decay': args.w_fc2, 'lr': args.lr_fc},
        {'params': model.fc1.parameters(), 'weight_decay': args.w_fc1, 'lr': args.lr_fc},
        {'params': model.att, 'weight_decay': args.w_att, 'lr': args.lr_att},
        {'params': model.Q.parameters(), 'weight_decay': args.w_fc1*3, 'lr': args.lr_fc},
        {'params': model.K.parameters(), 'weight_decay': args.w_fc1*3, 'lr': args.lr_fc},
        
    ]

    optimizer = optim.Adam(optimizer_sett)

    bad_counter = 0
    best = 999999999
    for epoch in range(args.epochs):
        loss_tra,acc_tra = train_step(model,optimizer,labels,features,idx_train)
        loss_val,acc_val = validate_step(model,labels,features,idx_val)
        #Uncomment following lines to see loss and accuracy values
        '''
        if(epoch+1)%1 == 0:

            print('Epoch:{:04d}'.format(epoch+1),
                'train',
                'loss:{:.3f}'.format(loss_tra),
                'acc:{:.2f}'.format(acc_tra*100),
                '| val',
                'loss:{:.3f}'.format(loss_val),
                'acc:{:.2f}'.format(acc_val*100))
        '''        

        if loss_val < best:
            best = loss_val
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break
    test_out = test_step(model,labels,features,idx_test)
    acc = test_out[1]


    return acc*100

t_total = time.time()
acc_list = []

for i in range(10):
    datastr = args.data
    splitstr = 'splits/'+args.data+'_split_0.6_0.2_'+str(i)+'.npz'
    accuracy_data = train(datastr,splitstr)
    acc_list.append(accuracy_data)


    ##print(i,": {:.2f}".format(acc_list[-1]))

print("Train cost: {:.4f}s".format(time.time() - t_total))
#print("Test acc.:{:.2f}".format(np.mean(acc_list)))
print(f"Test accuracy: {np.round(np.mean(acc_list),2)}, {np.round(np.std(acc_list),2)}")

