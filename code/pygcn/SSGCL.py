from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import assment_result
import os
from collections import Counter
import pandas as pd

from utils_SSGCL import *
from models import SSGCL
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='none')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--origin', action='store_true', default=True,
                    help='Keep the original implementation as the paper.')
parser.add_argument('--test_only', action="store_true", default=False,
                    help='Test on existing model')
parser.add_argument('--repeat', type=int, default=1,
                    help='number of experiments')
parser.add_argument('--verbose', action='store_true', default=False,
                    help='Show training process')
parser.add_argument('--split', type=str, default='random',
                    help='Data split method')
parser.add_argument('--rho', type=float, default=0.1,
                    help='Adj matrix corruption rate')
parser.add_argument('--corruption', type=str, default='node_shuffle',
                    help='Corruption method')

parser.add_argument('--gnnlayers', type=int, default=3, help="Number of gnn layers")

def get_data(node_num, time_matrix, time_node_feature):
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    print("prepared for loading data!")
    adj, features, graph = get_afldata(node_num, time_matrix, time_node_feature)
    print("Load have done!")
    idx_train, idx_val, idx_test = get_vttdata(node_num)

    if args.cuda:
        features = features.cuda()
        adj = adj.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
    return adj, features, idx_train, idx_val, idx_test, args, graph


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def train(num_epoch, time_step, last_embedding, patience=30, verbose=False):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_loss = float("inf")
    best_epoch = -1
    for epoch in range(num_epoch):
        t = time.time()
        optimizer.zero_grad()
        if time_step >= 1:
            outputs, labels = model(features, adj, last_embedding)
        else:
            outputs, labels = model(features, adj)
        if args.cuda:
            labels = labels.cuda()
        loss_train = F.binary_cross_entropy_with_logits(outputs, labels)
        loss = loss_train
        acc_train = binary_accuracy(outputs, labels)
        loss.backward()
        optimizer.step()

        loss_final = loss.item()
        accuracy = acc_train.item()
        if verbose:
            print('Epoch: {:04d}'.format(epoch + 1),
                  'loss_train: {:.4f}'.format(loss),
                  'acc_train: {:.4f}'.format(accuracy),
                  'time: {:.4f}s'.format(time.time() - t))

        # early stop
        if loss_final < best_loss:
            best_loss = loss_final
            best_epoch = epoch
        if epoch == best_epoch + patience:
            break


def test(verbose=False):
    with torch.set_grad_enabled(False):
        model.eval()
        last_embedding = 0
        outputs1, outputs2, weight1, weight2 = model(features, adj, last_embedding)
        outputs = (outputs1 + outputs2) / 2
        weight = (weight1 + weight2) / 2
        outputs_numpy = outputs.data.cpu().numpy()
    return outputs, weight, outputs_numpy

if __name__ == "__main__":
    dataset_list = ['mc3']
    node_num_list = [200]
    for data_num in range(1):

        node_num=node_num_list[data_num]
        dataset = dataset_list[data_num]
        print(dataset)
        dice = 0.0
        time_weight_list=[0]
        embedding_list = [0]
        NMI_list = []
        Q_list = []
        ARI_list = []
        NMI_t=[]
        ARI_t=[]
        Q_t=[]

        isone_hot = False 
        islabel = True 
        method = "SSGCL"
        base_data_path = "./data/"
        edges_base_path = "/edges"
        label_base_path = "/labels"
        edges_data_path = base_data_path + dataset + edges_base_path
        file_num = len(os.listdir(edges_data_path))
        print("file_num:{}".format(file_num))
        for t in range(file_num):
            print("The {}th snapshot".format(t))
            time_edges_path = edges_data_path + "/edge_" + str(t) + ".dat"
            time_matrix_ori = np.genfromtxt(f'./data/{dataset}/matrixs/matrix_{t}.dat')
            time_matrix = (time_matrix_ori + time_matrix_ori.T) / 2
            time_node_feature= np.genfromtxt(f'./data/{dataset}/node_features/feature_{t}.dat')
            adj, features, idx_train, idx_val, idx_test, args, graph = get_data(node_num, time_matrix, time_node_feature)
            print("Get data done!")
            print('Laplacian Smoothing...')
            adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
            adj.eliminate_zeros()
            adj_norm_s = preprocess_graph(adj, args.gnnlayers, norm='sym', renorm=True)
            sm_fea_s = sp.csr_matrix(features).toarray()
            for a in adj_norm_s:
                sm_fea_s = a.dot(sm_fea_s)
            sm_fea_s = torch.FloatTensor(sm_fea_s)
            features=sm_fea_s
            adj_1st = (adj + sp.eye(adj.shape[0])).toarray()
            adj_1st = torch.FloatTensor(adj_1st)
            adj=adj_1st

            if islabel:
                print('1111111111111111111111111')
                label_data_path = base_data_path + dataset + label_base_path + "/label_" + str(t) +".dat"
                original_cluster = np.loadtxt(label_data_path, dtype=int)
            sumARI = 0
            sumNMI = 0
            sumQ = 0
            for i in range(args.repeat):
                # model
                model = SSGCL(num_feat=features.shape[1],
                            num_hid=args.hidden,
                            time_step=t,
                            graph = graph,
                            time_weight=time_weight_list[-1],
                            dropout=args.dropout,
                            rho=args.rho,
                            corruption=args.corruption)

                print("----- %d / %d runs -----" % (i, args.repeat))
                # Train model
                t_total = time.time()
                if args.test_only:
                    model = torch.load("model")
                else:
                    train(args.epochs, t,embedding_list[-1], verbose=args.verbose)
                    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

                # Test
                outputs, weight, outputs_numpy = test(verbose=args.verbose)
                save_emb_path = "./code/embeddings/" + dataset + "/" + method + "/embeddingF_" + str(t) + ".dat"
                check_and_creat_dir(save_emb_path)
                np.savetxt(save_emb_path, outputs_numpy, fmt="%f")
                time_weight_list.append(weight)
                embedding_list.append(outputs)

                if islabel:
                    k = len(Counter(original_cluster))
                    print(k)
                    NMI, ARI,  Q = assment_result.eva(original_cluster, outputs_numpy, k, time_matrix)
                    print("NMI value is：{}".format(NMI))
                    NMI_list.append(NMI)
                    print("ARI value is：{}".format(ARI))
                    ARI_list.append(ARI)
                    print("Q value is：{}".format(Q))
                    Q_list.append(Q)
                    sumNMI+=NMI
                    sumARI+=ARI
                    sumQ+=Q

            t_ARI=sumARI/args.repeat
            t_NMI=sumNMI/args.repeat
            t_Q = sumQ / args.repeat
            ARI_t.append(t_ARI)
            NMI_t.append(t_NMI)
            Q_t.append(t_Q)
        if islabel:
            ave_NMI = np.mean(NMI_list)
            print('--------------------------------------------')
            print("The average NMI is:{}".format(ave_NMI))
            ave_ARI = np.mean(ARI_list)
            print('--------------------------------------------')
            print("The average ARI is:{}".format(ave_ARI))
            ave_Q = np.mean(Q_list)
            print('--------------------------------------------')
            print("The average Q is:{}".format(ave_Q))

