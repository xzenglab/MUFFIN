# -*- coding: utf-8 -*-
#
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import dgl
import errno
import numpy as np
import os
import torch

from dgl.nn.pytorch.glob import AvgPooling
from dgllife.model import load_pretrained
from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
from rdkit import Chem
from torch.utils.data import DataLoader

import pandas as pd

from argparse import ArgumentParser
from dgllife.utils import load_smiles_from_txt


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def graph_construction_and_featurization(smiles):
    """Construct graphs from SMILES and featurize them
    Parameters
    ----------
    smiles : list of str
        SMILES of molecules for embedding computation
    Returns
    -------
    list of DGLGraph
        List of graphs constructed and featurized
    list of bool
        Indicators for whether the SMILES string can be
        parsed by RDKit
    """
    # print(len(smiles))
    graphs = []
    success = []
    for smi in smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            # print(mol is None)
            if mol is None:
                success.append(False)
                continue
            # # print(mol)
            # print('it can go there.')
            g = mol_to_bigraph(mol, add_self_loop=True,
                               node_featurizer=PretrainAtomFeaturizer(),
                               edge_featurizer=PretrainBondFeaturizer(),
                               canonical_atom_order=False)
            # print('it can also go there.')
            graphs.append(g)
            success.append(True)
            print(len(graphs))
        except:
            success.append(False)
    # print(len(graphs))

    return graphs, success


def collate(graphs):
    return dgl.batch(graphs)


def main(args, dataset, name):
    data_loader = DataLoader(dataset, batch_size=args['batch_size'],
                             collate_fn=collate, shuffle=False)
    model = load_pretrained(args['model']).to(args['device'])
    model.eval()
    readout = AvgPooling()

    mol_emb = []
    for batch_id, bg in enumerate(data_loader):
        print('Processing batch {:d}/{:d}'.format(batch_id + 1, len(data_loader)))
        nfeats = [bg.ndata.pop('atomic_number').to(args['device']),
                  bg.ndata.pop('chirality_type').to(args['device'])]
        efeats = [bg.edata.pop('bond_type').to(args['device']),
                  bg.edata.pop('bond_direction_type').to(args['device'])]
        with torch.no_grad():
            node_repr = model(bg, nfeats, efeats)
        mol_emb.append(readout(bg, node_repr))
    print(len(mol_emb))
    mol_emb = torch.cat(mol_emb, dim=0).detach().cpu().numpy()
    np.save(args['out_dir'] + '/' + name + '.npy', mol_emb)


if __name__ == '__main__':

    parser = ArgumentParser("Molecule Embedding Computation with Pre-trained Models")
    parser.add_argument('-fi', '--file', type=str,
                        help="Path to the file of SMILES")
    parser.add_argument('-fo', '--format', choices=['txt', 'csv'], default='csv',
                        help="Format for the file of SMILES (default: 'txt')")
    parser.add_argument('-sc', '--smiles-column', type=str,
                        help="Column for SMILES in the CSV file.")
    parser.add_argument('-m', '--model', choices=['gin_supervised_contextpred',
                                                  'gin_supervised_infomax',
                                                  'gin_supervised_edgepred',
                                                  'gin_supervised_masking'],
                        help='Pre-trained model to use for computing molecule embeddings')
    parser.add_argument('-b', '--batch-size', type=int, default=256,
                        help='Batch size for embedding computation')
    parser.add_argument('-o', '--out-dir', type=str, default='./data/DRKG/SARS_Embedding_Jure_300',
                        help='Path to the computation results')
    args = parser.parse_args().__dict__
    mkdir_p(args['out_dir'])

    if torch.cuda.is_available():
        args['device'] = torch.device('cuda:0')
    else:
        args['device'] = torch.device('cpu')

    if args['format'] == 'txt':
        smiles = load_smiles_from_txt(args['file'])
    else:
        df = pd.read_csv(args['file'])
        # smiles = df[args['smiles_column']].tolist()
        # 可以根据你的csv文件中"SMILES"所在列的名字来修改此处对应的"smiles"
        smiles = df['smiles'].tolist()
    print(len(smiles))
    # 文件命名格式以数据集文件名字命名
    name = args['file'].split('/')[-1].split('.')[0]
    dataset, success = graph_construction_and_featurization(smiles)
    # np.save(args['out_dir'] + '/mol_parsed.npy', np.array(success))
    main(args, dataset, name)
