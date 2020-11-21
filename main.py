# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from time import time
import pandas
import pandas as pd

import torch.optim as optim

from collections import OrderedDict

import argparse
import logging

import os
import random

from sklearn.model_selection import KFold, StratifiedKFold

import torch.utils.data as Data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import copy

import torch as th

EMB_INIT_EPS = 2.0
gamma = 12.0


# --------------------------------------------initial param------------------------------------------------------------

def parse_SKGDDI_args():
    parser = argparse.ArgumentParser(description="Run SKGDDI.")

    parser.add_argument('--seed', type=int, default=2020,
                        help='Random seed.')

    parser.add_argument('--data_name', nargs='?', default='DRKG',
                        help='Choose a dataset from {DrugBank, DRKG}')
    parser.add_argument('--data_dir', nargs='?', default='data/',
                        help='Input data path.')

    parser.add_argument('--use_pretrain', type=int, default=1,
                        help='0: No pretrain, 1: Pretrain with the learned embeddings')
    parser.add_argument('--pretrain_embedding_dir', nargs='?', default='embedding_data/',
                        help='Path of learned embeddings.')
    parser.add_argument('--pretrain_model_path', nargs='?', default='trained_model/model.pth',
                        help='Path of stored model.')

    parser.add_argument('--DDI_batch_size', type=int, default=2048,
                        help='DDI batch size.')
    parser.add_argument('--kg_batch_size', type=int, default=2048,
                        help='KG batch size.')
    parser.add_argument('--DDI_evaluate_size', type=int, default=2500,
                        help='KG batch size.')
    parser.add_argument('-n', '--negative_sample_size', default=256, type=int)

    parser.add_argument('--entity_dim', type=int, default=100,
                        help='User / entity Embedding size.')
    parser.add_argument('--relation_dim', type=int, default=100,
                        help='Relation Embedding size.')

    parser.add_argument('--aggregation_type', nargs='?', default='sum',
                        help='Specify the type of the aggregation layer from {sum, concat, pna}.')
    parser.add_argument('--conv_dim_list', nargs='?', default='[64, 32, 16]',
                        help='Output sizes of every aggregation layer.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1, 0.1]',
                        help='Dropout probability w.r.t. message dropout for each deep layer. 0: no dropout.')

    parser.add_argument('--kg_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating KG l2 loss.')
    parser.add_argument('--DDI_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating DDI l2 loss.')

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--n_epoch', type=int, default=200,
                        help='Number of epoch.')
    parser.add_argument('--stopping_steps', type=int, default=10,
                        help='Number of epoch for early stopping')

    parser.add_argument('--ddi_print_every', type=int, default=1,
                        help='Iter interval of printing DDI loss.')

    parser.add_argument('--kg_print_every', type=int, default=1,
                        help='Iter interval of printing KG loss.')
    parser.add_argument('--evaluate_every', type=int, default=1,
                        help='Epoch interval of evaluating DDI.')

    parser.add_argument('--multi_type', nargs='?', default='False',
                        help='whether task is multi-class')
    parser.add_argument('--attention_type', nargs='?', default='concat',
                        help='sum or concat')
    parser.add_argument('--n_hidden_1', type=int, default=2048,
                        help='FC hidden 1 dim')
    parser.add_argument('--n_hidden_2', type=int, default=2048,
                        help='FC hidden 2 dim')
    parser.add_argument('--out_dim', type=int, default=1,
                        help='FC output dim: 86 or 1')
    parser.add_argument('--structure_dim', type=int, default=300,
                        help='structure_dim')
    parser.add_argument('--pre_entity_dim', type=int, default=400,
                        help='pre_entity_dim')
    parser.add_argument('--feature_fusion', nargs='?', default='init_double',
                        help='feature fusion type: concat / sum / none / cross/ double')

    parser.add_argument("--graph_split_size", type=float, default=0.5,
                        help="portion of edges used as positive sample")
    parser.add_argument("--negative_sample", type=int, default=10,
                        help="number of negative samples per positive sample")
    parser.add_argument("--edge_sampler", type=str, default="neighbor",
                        help="type of edge sampler: 'uniform' or 'neighbor'")
    parser.add_argument("--graph_batch_size", type=int, default=40000)

    # parser.add_argument('--negative_sample_size', default=256, type=int)
    parser.add_argument('--max_steps', default=128000, type=int)
    parser.add_argument('-adv', '--negative_adversarial_sampling', action='store_true')
    parser.add_argument('--uni_weight', action='store_true',
                        help='Otherwise use subsampling weighting like in word2vec')

    args = parser.parse_args()

    save_dir = 'trained_model/SKGDDI/epoch_200/{}/all_entitydim{}_relationdim{}_feature{}_{}_{}_lr{}_pretrain{}/'.format(
        args.data_name, args.entity_dim, args.relation_dim, args.feature_fusion, args.aggregation_type,
        '-'.join([str(i) for i in eval(args.conv_dim_list)]), args.lr, args.use_pretrain)
    args.save_dir = save_dir

    return args


# ----------------------------------------define log information--------------------------------------------------------

# create log information
def create_log_id(dir_path):
    log_count = 0
    file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    while os.path.exists(file_path):
        log_count += 1
        file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    return log_count


def logging_config(folder=None, name=None,
                   level=logging.DEBUG,
                   console_level=logging.DEBUG,
                   no_console=True):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []
    logpath = os.path.join(folder, name + ".txt")
    print("All logs will be saved to %s" % logpath)

    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)

    if not no_console:
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logging.root.addHandler(logconsole)
    return folder


def to_categorical(y, num_classes=None, dtype='float32'):
    """Converts a class vector (integers) to binary class matrix.
    E.g. for use with categorical_crossentropy.
    # Arguments
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
        dtype: The data type expected by the input, as a string
            (`float32`, `float64`, `int32`...)
    # Returns
        A binary matrix representation of the input. The classes axis
        is placed last.
    # Example
    ```python
    # Consider an array of 5 labels out of a set of 3 classes {0, 1, 2}:
    > labels
    array([0, 2, 1, 2, 0])
    # `to_categorical` converts this into a matrix with as many
    # columns as there are classes. The number of rows
    # stays the same.
    > to_categorical(labels)
    array([[ 1.,  0.,  0.],
           [ 0.,  0.,  1.],
           [ 0.,  1.,  0.],
           [ 0.,  0.,  1.],
           [ 1.,  0.,  0.]], dtype=float32)
    ```
    """
    # 将输入y向量转换为数组
    y = np.array(y, dtype='int')
    # 获取数组的行列大小
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    # y变为1维数组
    y = y.ravel()
    # 如果用户没有输入分类个数，则自行计算分类个数
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    # 生成全为0的n行num_classes列的值全为0的矩阵
    categorical = np.zeros((n, num_classes), dtype=dtype)
    # np.arange(n)得到每个行的位置值，y里边则是每个列的位置值
    categorical[np.arange(n), y] = 1
    # 进行reshape矫正
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


# -----------------------------------------loading KG data and DDI 5-fold data------------------------------------------

# loading data
class DataLoaderSKGDDI(object):

    def __init__(self, args, logging, multi_type=False):
        self.args = args
        self.data_name = args.data_name
        self.use_pretrain = args.use_pretrain
        self.pretrain_embedding_dir = args.pretrain_embedding_dir

        self.ddi_batch_size = args.DDI_batch_size
        self.kg_batch_size = args.kg_batch_size

        self.multi = multi_type

        self.entity_dim = args.entity_dim

        data_dir = os.path.join(args.data_dir, args.data_name)
        train_file = os.path.join(data_dir, 'DDI_pos_neg.txt')
        if args.data_name == 'DRKG':
            kg_file = os.path.join(data_dir, "train.tsv")
        else:
            kg_file = os.path.join(data_dir, "kg2id.txt")

        self.DDI_train_data_X, self.DDI_train_data_Y, self.DDI_test_data_X, self.DDI_test_data_Y = self.load_DDI_data(
            train_file)

        self.statistic_ddi_data()

        triples = self.read_triple(kg_file)
        self.construct_triples(triples)

        self.print_info(logging)

        self.train_graph = None
        if self.use_pretrain == 1:
            self.load_pretrained_data()

    def load_DDI_data(self, filename):

        train_X_data = []
        train_Y_data = []
        test_X_data = []
        test_Y_data = []

        traindf = pandas.read_csv(filename, delimiter='\t', header=None)
        data = traindf.values
        DDI = data[:, 0:2]
        # 1123100,2
        print(DDI.shape)
        Y = data[:, 2]
        label = np.array(list(map(int, Y)))

        print(DDI.shape)
        print(label.shape)

        kfold = KFold(n_splits=5, shuffle=True, random_state=3)

        for train, test in kfold.split(DDI, label):
            train_X_data.append(DDI[train])
            train_Y_data.append(label[train])
            test_X_data.append(DDI[test])
            test_Y_data.append(label[test])

        train_X = np.array(train_X_data)
        train_Y = np.array(train_Y_data)
        test_X = np.array(test_X_data)
        test_Y = np.array(test_Y_data)

        print('Loading DDI data down!')

        return train_X, train_Y, test_X, test_Y

    # 5-fold train data length
    def statistic_ddi_data(self):
        data = []
        for i in range(len(self.DDI_train_data_X)):
            data.append(len(self.DDI_train_data_X[i]))
        self.n_ddi_train = data

    def read_triple(self, path, mode='train', skip_first_line=False, format=[0, 1, 2]):
        heads = []
        tails = []
        rels = []
        print('Reading {} triples....'.format(mode))
        with open(path) as f:
            if skip_first_line:
                _ = f.readline()
            for line in f:
                triple = line.strip().split('\t')
                h, r, t = triple[format[0]], triple[format[1]], triple[format[2]]
                try:
                    heads.append(int(h))
                    tails.append(int(t))
                    rels.append(int(r))
                except ValueError:
                    print("For User Defined Dataset, both node ids and relation ids in the triplets should be int "
                          "other than {}\t{}\t{}".format(h, r, t))
                    raise
        heads = np.array(heads, dtype=np.int64)
        tails = np.array(tails, dtype=np.int64)
        rels = np.array(rels, dtype=np.int64)
        print('Finished. Read {} {} triples.'.format(len(heads), mode))

        return heads, rels, tails

    # load kg triple
    def load_kg(self, filename):
        kg_data = pd.read_csv(filename, sep='\t', names=['h', 'r', 't'], engine='python')
        kg_data = kg_data.drop_duplicates()
        return kg_data

    def construct_triples(self, kg_data):
        print("construct kg...")
        src, rel, dst = kg_data

        src, rel, dst = np.concatenate((src, dst)), np.concatenate((rel, rel)), np.concatenate((dst, src))
        self.kg_triple = np.array(sorted(zip(src, rel, dst)))
        # print(self.kg_triple.shape)

        self.n_relations = max(rel) + 1
        self.n_entities = max(max(src), max(dst)) + 1

        self.kg_train_data = self.kg_triple
        self.n_kg_train = len(self.kg_train_data)

        print('construct kg down!')

    def print_info(self, logging):

        logging.info('n_entities:         %d' % self.n_entities)
        logging.info('n_relations:        %d' % self.n_relations)
        logging.info('n_kg_train:         %d' % self.n_kg_train)
        logging.info('n_ddi_train:         %s' % self.n_ddi_train)

    def load_pretrained_data(self):

        if self.data_name == 'DrugBank':

            # load pretrained KG information

            transE_entity_path = 'embedding_data/entityVector_400.npz'
            transE_relation_path = 'embedding_data/relationVector_400.npz'
            transE_entity_data = np.load(transE_entity_path)
            transE_relation_data = np.load(transE_relation_path)
            transE_entity_data = transE_entity_data['embed']
            transE_relation_data = transE_relation_data['embed']

            # load pretrained Structure information

            masking_entity_path = 'embedding_data/gin_supervised_masking_embedding.npy'

            masking_entity_data = np.load(masking_entity_path)

        else:

            if self.entity_dim == 300:
                transE_entity_path = 'data/DRKG/TransE_l2_DRKG_0/DRKG_TransE_l2_entity_300.npy'
                transE_relation_path = 'data/DRKG/TransE_l2_DRKG_0/DRKG_TransE_l2_relation_300.npy'
            elif self.entity_dim == 256:
                transE_entity_path = 'ckpts/TransE_l2_DRKG_19/DRKG_TransE_l2_entity.npy'
                transE_relation_path = 'ckpts/TransE_l2_DRKG_19/DRKG_TransE_l2_relation.npy'
            elif self.entity_dim == 100:
                # 128 negative sample
                transE_entity_path = 'data/DRKG/DRKG_TransE_l2_entity.npy'
                transE_relation_path = 'data/DRKG/DRKG_TransE_l2_relation.npy'
            elif self.entity_dim == 128:
                transE_entity_path = 'data/DRKG/DRKG_TransE_l2_entity_128.npy'
                transE_relation_path = 'data/DRKG/DRKG_TransE_l2_relation_128.npy'
            elif self.entity_dim == 32:
                transE_entity_path = 'data/DRKG/32/TransE_l2_DRKG_0/DRKG_TransE_l2_entity.npy'
                transE_relation_path = 'data/DRKG/32/TransE_l2_DRKG_0/DRKG_TransE_l2_relation.npy'
            elif self.entity_dim == 64:
                transE_entity_path = 'data/DRKG/64/TransE_l2_DRKG_0/DRKG_TransE_l2_entity.npy'
                transE_relation_path = 'data/DRKG/64/TransE_l2_DRKG_0/DRKG_TransE_l2_relation.npy'

            transE_entity_data = np.load(transE_entity_path)
            transE_relation_data = np.load(transE_relation_path)

            masking_entity_path = 'data/DRKG/gin_supervised_masking_embedding.npy'
            masking_entity_data = np.load(masking_entity_path)

        # apply pretrained data

        self.entity_pre_embed = transE_entity_data
        self.relation_pre_embed = transE_relation_data

        self.structure_pre_embed = masking_entity_data

        self.n_approved_drug = self.structure_pre_embed.shape[0]

        print('loading pretrain data down!')


def chunkIt(seq, num):
    data = []
    for i in range(0, len(seq), num):
        if i + num > len(seq):
            data.append(seq[i:])
        else:
            data.append(seq[i:i + num])

    return data


def early_stopping(recall_list, stopping_steps):
    best_recall = max(recall_list)
    best_step = recall_list.index(best_recall)
    if len(recall_list) - best_step - 1 >= stopping_steps:
        should_stop = True
    else:
        should_stop = False
    return best_recall, should_stop


def save_model(all_embed, model, model_dir, current_epoch, last_best_epoch=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(current_epoch))
    torch.save({'model_state_dict': model.state_dict(), 'epoch': current_epoch}, model_state_file)

    file_name = os.path.join(model_dir, 'drug_embed{}.npy'.format(current_epoch))
    np.save(file_name, all_embed.cpu().detach().numpy())

    data = np.load(file_name)
    print(data.shape)

    if last_best_epoch is not None and current_epoch != last_best_epoch:
        old_model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(last_best_epoch))
        old_embedding_file = os.path.join(model_dir, 'drug_embed{}.npy'.format(last_best_epoch))
        if os.path.exists(old_model_state_file):
            os.system('rm {}'.format(old_model_state_file))
        if os.path.exists(old_embedding_file):
            os.system('rm {}'.format(old_embedding_file))


def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))

    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError:
        state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            k_ = k[7:]  # remove 'module.' of DistributedDataParallel instance
            state_dict[k_] = v
        model.load_state_dict(state_dict)

    model.eval()
    return model


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


def get_device(args):
    return th.device('cpu') if args.gpu[0] < 0 else th.device('cuda:' + str(args.gpu[0]))


# ----------------------------------------------  Main model part  -----------------------------------------------------

class GCNModel(nn.Module):

    def __init__(self, args, n_entities, n_relations, entity_pre_embed=None, relation_pre_embed=None,
                 structure_pre_embed=None):

        super(GCNModel, self).__init__()
        self.use_pretrain = args.use_pretrain

        self.n_entities = n_entities
        self.n_relations = n_relations

        self.entity_dim = args.entity_dim
        self.relation_dim = args.relation_dim

        self.structure_dim = args.structure_dim
        self.pre_entity_dim = args.pre_entity_dim

        self.aggregation_type = args.aggregation_type
        self.attention_type = args.attention_type
        self.fusion_type = args.feature_fusion
        self.multi_type = args.multi_type

        self.conv_dim_list = [args.entity_dim] + eval(args.conv_dim_list)
        self.mess_dropout = eval(args.mess_dropout)
        self.n_layers = len(eval(args.conv_dim_list))

        self.kg_l2loss_lambda = args.kg_l2loss_lambda
        self.ddi_l2loss_lambda = args.DDI_l2loss_lambda

        self.hidden_dim = args.entity_dim
        self.eps = EMB_INIT_EPS
        self.emb_init = (gamma + self.eps) / self.hidden_dim

        # fusion type
        if self.fusion_type == 'concat':

            self.layer1_f = nn.Sequential(nn.Linear(self.structure_dim + self.entity_dim, self.entity_dim),
                                          nn.BatchNorm1d(self.entity_dim),
                                          nn.LeakyReLU(True))
            self.layer2_f = nn.Sequential(nn.Linear(self.entity_dim, self.entity_dim), nn.BatchNorm1d(self.entity_dim),
                                          nn.LeakyReLU(True))
            self.layer3_f = nn.Sequential(nn.Linear(self.entity_dim, self.entity_dim), nn.BatchNorm1d(self.entity_dim),
                                          nn.LeakyReLU(True))

        elif self.fusion_type == 'sum':

            self.W_s = nn.Linear(self.structure_dim, self.entity_dim)
            self.W_e = nn.Linear(self.entity_dim, self.entity_dim)

        elif self.fusion_type == 'init_double':

            self.druglayer_structure = nn.Linear(self.structure_dim, self.entity_dim)
            self.druglayer_KG = nn.Linear(self.entity_dim, self.entity_dim)

            self.add_drug = nn.Sequential(nn.Linear(self.entity_dim, self.entity_dim))
            self.cross_add_drug = nn.Sequential(nn.Linear(self.entity_dim, self.entity_dim))
            self.multi_drug = nn.Sequential(nn.Linear(self.entity_dim, self.entity_dim))
            self.activate = nn.ReLU()

            self.conv1 = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 5)),
                nn.BatchNorm2d(8), nn.MaxPool2d((2, 2)), nn.ReLU())
            self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(5, 5)),
                nn.BatchNorm2d(8), nn.MaxPool2d((2, 2)), nn.ReLU())
            if self.entity_dim == 300:
                self.fc1 = nn.Sequential(nn.Linear(72 * 72 * 8, self.entity_dim), nn.BatchNorm1d(self.entity_dim),
                                         nn.ReLU(True))
            elif self.entity_dim == 256:
                self.fc1 = nn.Sequential(nn.Linear(61 * 61 * 8, self.entity_dim), nn.BatchNorm1d(self.entity_dim),
                                         nn.ReLU(True))
            elif self.entity_dim == 128:
                self.fc1 = nn.Sequential(nn.Linear(29 * 29 * 8, self.entity_dim), nn.BatchNorm1d(self.entity_dim),
                                         nn.ReLU(True))
            elif self.entity_dim == 100:
                self.fc1 = nn.Sequential(nn.Linear(22 * 22 * 8, self.entity_dim), nn.BatchNorm1d(self.entity_dim),
                                         nn.ReLU(True))
            elif self.entity_dim == 32:
                self.fc1 = nn.Sequential(nn.Linear(5 * 5 * 8, self.entity_dim), nn.BatchNorm1d(self.entity_dim),
                                         nn.ReLU(True))
            elif self.entity_dim == 64:
                self.fc1 = nn.Sequential(nn.Linear(13 * 13 * 8, self.entity_dim), nn.BatchNorm1d(self.entity_dim),
                                         nn.ReLU(True))

            self.fc2_global = nn.Sequential(
                nn.Linear(self.entity_dim * self.entity_dim + self.entity_dim, self.entity_dim),
                nn.ReLU(True))
            self.fc2_global_reverse = nn.Sequential(
                nn.Linear(self.entity_dim * self.entity_dim + self.entity_dim, self.entity_dim),
                nn.ReLU(True))
            self.fc2_cross = nn.Sequential(
                nn.Linear(self.entity_dim * 4, self.entity_dim),
                nn.ReLU(True))

        if (self.use_pretrain == 1) and (structure_pre_embed is not None):

            self.n_approved_drug = structure_pre_embed.shape[0]
            self.structure_pre_embed = structure_pre_embed

            if self.fusion_type in ['init_double', 'sum', 'concat']:
                self.pre_entity_embed = entity_pre_embed

        if self.fusion_type in ['double', 'init_double']:
            self.all_embedding_dim = (self.entity_dim * 3 + self.structure_dim + self.entity_dim) * 2

        elif self.fusion_type in ['sum', 'concat']:
            self.all_embedding_dim = self.entity_dim * 2
        self.layer1 = nn.Sequential(nn.Linear(self.all_embedding_dim, args.n_hidden_1), nn.BatchNorm1d(args.n_hidden_1),
                                    nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(args.n_hidden_1, args.n_hidden_2), nn.BatchNorm1d(args.n_hidden_2),
                                    nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(args.n_hidden_2, args.out_dim))

    def generate_fusion_feature(self, embedding_pre, embedding_after, batch_data, epoch):
        # we focus on approved drug
        global embedding_data
        global embedding_data_reverse

        entity_embed_pre = self.pre_entity_embed[:self.n_approved_drug, :]

        if self.fusion_type == 'concat':

            x = torch.cat([self.structure_pre_embed, entity_embed_pre], dim=1)
            x = self.layer1_f(x)
            x = self.layer2_f(x)
            x = self.layer3_f(x)

            return x

        elif self.fusion_type == 'sum':

            structure = self.W_s(self.structure_pre_embed)
            entity = self.W_e(entity_embed_pre)
            add_structure_entity = structure + entity

            return add_structure_entity

        elif self.fusion_type == 'init_double':

            structure = self.druglayer_structure(self.structure_pre_embed)

            entity = self.druglayer_KG(entity_embed_pre)

            structure_embed_reshape = structure.unsqueeze(-1)  # batch_size * embed_dim * 1
            entity_embed_reshape = entity.unsqueeze(-1)  # batch_size * embed_dim * 1

            entity_matrix = structure_embed_reshape * entity_embed_reshape.permute(
                (0, 2, 1))  # batch_size * embed_dim * embed_dim

            entity_matrix_reverse = entity_embed_reshape * structure_embed_reshape.permute(
                (0, 2, 1))  # batch_size * embed_dim * embed_dim

            entity_global = entity_matrix.view(entity_matrix.size(0), -1)

            entity_global_reverse = entity_matrix_reverse.view(entity_matrix.size(0), -1)

            entity_matrix_reshape = entity_matrix.unsqueeze(1)

            for i, data in enumerate(batch_data):

                entity_matrix_reshape = entity_matrix_reshape.to('cuda')
                entity_data = entity_matrix_reshape.index_select(0, data[0].to('cuda'))

                out = self.conv1(entity_data)
                out = self.conv2(out)
                out = out.view(out.size(0), -1)
                out = self.fc1(out)

                if i == 0:
                    embedding_data = out
                else:
                    embedding_data = torch.cat((embedding_data, out), 0)

            global_local_before = torch.cat((embedding_data, entity_global), 1)
            cross_embedding_pre = self.fc2_global(global_local_before)

            # another reverse part

            entity_matrix_reshape_reverse = entity_matrix_reverse.unsqueeze(1)

            for i, data in enumerate(batch_data):

                entity_matrix_reshape_reverse = entity_matrix_reshape_reverse.to('cuda')
                entity_reverse = entity_matrix_reshape_reverse.index_select(0, data[0].to('cuda'))

                out = self.conv1(entity_reverse)

                out = self.conv2(out)

                out = out.view(out.size(0), -1)

                out = self.fc1(out)

                if i == 0:
                    embedding_data_reverse = out
                else:
                    embedding_data_reverse = torch.cat((embedding_data_reverse, out), 0)

            global_local_before_reverse = torch.cat((embedding_data_reverse, entity_global_reverse), 1)
            cross_embedding_pre_reverse = self.fc2_global_reverse(global_local_before_reverse)

            out3 = self.activate(self.multi_drug(structure * entity))

            out_concat = torch.cat(
                (self.structure_pre_embed, entity_embed_pre, cross_embedding_pre, cross_embedding_pre_reverse, out3), 1)

            return out_concat

    def train_DDI_data(self, mode, g, train_data, embedding_pre, embedding_after, batch_data, epoch):

        all_embed = self.generate_fusion_feature(embedding_pre, embedding_after, batch_data, epoch)

        drug1_embed = all_embed[train_data[:, 0]]
        drug2_embed = all_embed[train_data[:, 1]]
        drug_data = torch.cat((drug1_embed, drug2_embed), 1)

        x = self.layer1(drug_data)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def test_DDI_data(self, mode, g, test_data, embedding_pre, embedding_after, batch_data, epoch):

        all_embed = self.generate_fusion_feature(embedding_pre, embedding_after, batch_data, epoch)
        drug1_embed = all_embed[test_data[:, 0]]
        drug2_embed = all_embed[test_data[:, 1]]
        drug_data = torch.cat((drug1_embed, drug2_embed), 1)

        x = self.layer1(drug_data)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.multi_type != 'False':
            pred = F.softmax(x, dim=1)
        else:
            pred = torch.sigmoid(x)
        return pred, all_embed

    def forward(self, mode, *input):
        if mode == 'calc_att':
            return self.compute_attention(*input)
        if mode == 'calc_ddi_loss':
            return self.train_DDI_data(mode, *input)
        if mode == 'predict':
            return self.test_DDI_data(mode, *input)
        if mode == 'calc_kg_loss':
            return self.kg_train(*input)
        if mode == 'feature_fusion':
            return self.generate_fusion_feature(*input)


# -------------------------------------- metrics and evaluation define -------------------------------------------------

def calc_metrics(y_true, y_pred, pred_score, multi_type):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    # auc = roc_auc_score(y_one_hot, pred_score, average='micro')
    if multi_type != 'False':
        y_one_hot = to_categorical(y_true, num_classes=86)
        auc = roc_auc_score(y_one_hot, pred_score, average='micro')
    else:
        auc = roc_auc_score(y_true.cuda().data.cpu().numpy(), pred_score.cuda().data.cpu().numpy())
    print(acc, precision, recall, f1, auc)

    return acc, precision, recall, f1, auc


def evaluate(args, model, train_graph, loader_test, embedding_pre, embedding_after, loader_idx, epoch):
    model.eval()

    precision_list = []
    recall_list = []
    f1_list = []
    acc_list = []
    auc_list = []

    with torch.no_grad():
        for data in loader_test:
            test_x, test_y = data
            out, all_embedding = model('predict', train_graph, test_x, embedding_pre, embedding_after, loader_idx,
                                       epoch)
            if args.multi_type == 'False':
                out = out.squeeze(-1)
                prediction = copy.deepcopy(out)
                prediction[prediction >= 0.5] = 1
                prediction[prediction < 0.5] = 0

            else:
                prediction = torch.max(out, 1)[1]
            prediction = prediction.cuda().data.cpu().numpy()
            acc, precision, recall, f1, auc = calc_metrics(test_y, prediction, out, args.multi_type)

            acc_list.append(acc)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            auc_list.append(auc)

    precision = np.mean(precision_list)
    recall = np.mean(recall_list)
    f1 = np.mean(f1_list)
    acc = np.mean(acc_list)
    auc = np.mean(auc_list)

    return precision, recall, f1, acc, auc, all_embedding


# -----------------------------------   train model  -------------------------------------------------------------------

def train(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # set log file
    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # GPU / CPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # initialize data
    data = DataLoaderSKGDDI(args, logging)

    n_approved_drug = data.n_approved_drug
    n_entities = data.n_entities

    # define pretrain embedding information
    if args.use_pretrain == 1:

        if args.feature_fusion in ['sum', 'concat', 'init_double']:

            structure_pre_embed = torch.tensor(data.structure_pre_embed).to(device)
            entity_pre_embed = torch.tensor(data.entity_pre_embed).to(device).float()
            relation_pre_embed = torch.tensor(data.relation_pre_embed).to(device).float()
            embedding_pre = torch.LongTensor(range(data.n_approved_drug)).to(device)
            embedding_after = torch.LongTensor(range(data.n_approved_drug, data.n_entities)).to(device)

        else:
            entity_pre_embed, relation_pre_embed = None, None
            structure_pre_embed = torch.tensor(data.structure_pre_embed)

    else:
        entity_pre_embed, relation_pre_embed, structure_pre_embed = None, None, None


    train_graph = None

    all_acc_list = []
    all_precision_list = []
    all_recall_list = []
    all_f1_list = []
    all_auc_list = []

    # train model
    # use 5-fold cross validation
    for i in range(5):

        # construct model & optimizer
        model = GCNModel(args, data.n_entities, data.n_relations, entity_pre_embed, relation_pre_embed,
                         structure_pre_embed)
        if args.use_pretrain == 2:
            # 加载模型
            model = load_model(model, args.pretrain_model_path)

        model.to(device)

        logging.info(model)

        # define optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        if args.multi_type != 'False':
            print('Yes')
            loss_func = torch.nn.CrossEntropyLoss()
        else:
            print('No')
            loss_func = torch.nn.BCEWithLogitsLoss()

        # Data.TensorDataset()里的两个输入是tensor类型
        train_x = torch.from_numpy(data.DDI_train_data_X[i])
        train_y = torch.from_numpy(data.DDI_train_data_Y[i])
        test_x = torch.from_numpy(data.DDI_test_data_X[i])
        test_y = torch.from_numpy(data.DDI_test_data_Y[i])
        torch_dataset_train = Data.TensorDataset(train_x, train_y)
        torch_dataset_test = Data.TensorDataset(test_x, test_y)

        loader_train = Data.DataLoader(
            dataset=torch_dataset_train,
            batch_size=data.ddi_batch_size,
            shuffle=True
        )
        loader_test = Data.DataLoader(
            dataset=torch_dataset_test,
            batch_size=args.DDI_evaluate_size,
            shuffle=True
        )

        data_idx = Data.TensorDataset(torch.LongTensor(range(n_approved_drug)))
        loader_idx = Data.DataLoader(
            dataset=data_idx,
            batch_size=128,
            shuffle=False
        )
        best_epoch = -1
        epoch_list = []
        acc_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        auc_list = []
        init_step = 0

        for epoch in range(1, args.n_epoch + 1):
            time0 = time()
            model.train()

            time1 = time()
            ddi_total_loss = 0
            n_ddi_batch = data.n_ddi_train[i] // data.ddi_batch_size + 1

            for step, (batch_x, batch_y) in enumerate(loader_train):
                iter = step + 1
                time2 = time()

                if use_cuda:
                    batch_x = batch_x.to(device)
                    batch_y = batch_y.to(device)

                out = model('calc_ddi_loss', train_graph, batch_x, embedding_pre, embedding_after, loader_idx, epoch)

                if args.multi_type == 'False':
                    out = out.squeeze(-1)

                loss = loss_func(out, batch_y.float())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                ddi_total_loss += loss.item()

                if (iter % args.ddi_print_every) == 0:
                    logging.info(
                        'DDI Training: Epoch {:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean '
                        'Loss {:.4f}'.format(
                            epoch, iter, n_ddi_batch, time() - time2, loss.item(), ddi_total_loss / iter))
            logging.info(
                'DDI Training: Epoch {:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(
                    epoch,
                    n_ddi_batch,
                    time() - time1,
                    ddi_total_loss / n_ddi_batch))

            logging.info('DDI + KG Training: Epoch {:04d} | Total Time {:.1f}s'.format(epoch, time() - time0))

            # evaluate cf
            if (epoch % args.evaluate_every) == 0:
                time1 = time()
                precision, recall, f1, acc, auc, all_embed = evaluate(args, model, train_graph, loader_test,
                                                                      embedding_pre,
                                                                      embedding_after, loader_idx, epoch)
                logging.info(
                    'DDI Evaluation: Epoch {:04d} | Total Time {:.1f}s | Precision {:.4f} Recall {:.4f} F1 {:.4f} ACC '
                    '{:.4f} AUC {:.4f}'.format(
                        epoch, time() - time1, precision, recall, f1, acc, auc))

                epoch_list.append(epoch)
                precision_list.append(precision)
                recall_list.append(recall)
                f1_list.append(f1)
                acc_list.append(acc)
                auc_list.append(auc)
                best_auc, should_stop = early_stopping(auc_list, args.stopping_steps)

                if should_stop:
                    index = auc_list.index(best_auc)
                    all_acc_list.append(acc_list[index])
                    all_auc_list.append(auc_list[index])
                    all_precision_list.append(precision_list[index])
                    all_recall_list.append(recall_list[index])
                    all_f1_list.append(f1_list[index])
                    logging.info('Final DDI Evaluation: Precision {:.4f} Recall {:.4f} F1 {:.4f} ACC '
                                 '{:.4f} AUC {:.4f}'.format(precision, recall, f1, acc, auc))
                    break

                if auc_list.index(best_auc) == len(auc_list) - 1:
                    save_model(all_embed, model, args.save_dir, epoch, best_epoch)
                    logging.info('Save model on epoch {:04d}!'.format(epoch))
                    best_epoch = epoch

                if epoch == args.n_epoch:
                    index = auc_list.index(best_auc)
                    all_acc_list.append(acc_list[index])
                    all_auc_list.append(auc_list[index])
                    all_precision_list.append(precision_list[index])
                    all_recall_list.append(recall_list[index])
                    all_f1_list.append(f1_list[index])
                    logging.info('Final DDI Evaluation: Precision {:.4f} Recall {:.4f} F1 {:.4f} ACC '
                                 '{:.4f} AUC {:.4f}'.format(precision, recall, f1, acc, auc))

    print(all_acc_list)
    print(all_precision_list)
    print(all_recall_list)
    print(all_f1_list)
    print(all_auc_list)
    mean_acc = np.mean(all_acc_list)
    mean_precision = np.mean(all_precision_list)
    mean_recall = np.mean(all_recall_list)
    mean_f1 = np.mean(all_f1_list)
    mean_auc = np.mean(all_auc_list)
    logging.info('5-fold cross validation DDI Mean Evaluation: Precision {:.4f} Recall {:.4f} F1 {:.4f} ACC '
                 '{:.4f} AUC {:.4f}'.format(mean_precision, mean_recall, mean_f1, mean_acc, mean_auc))


if __name__ == '__main__':
    args = parse_SKGDDI_args()
    train(args)
