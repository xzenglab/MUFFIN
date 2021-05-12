# MUFFIN

## Overview

This repository contains source code for our paper "MUFFIN: Multi-Scale Feature Fusion for Drug–Drug Interaction Prediction".

## Dataset Preparation

You need to provide datasets defined as below:

##### 1. DDI dataset file:

'DDI_pos_neg.txt': store the DDI dataset, the form is "drug1 \t drug2 \t type". For binary data: type is in {0,1}, for the multi-class DDI dataset, type ranges from 0 to 80, and for the multi-label dataset, it is in [0,200).

For the TWOSIDES dataset, you can obtained from http://tatonettilab.org/offsides/

you can also use following command to get the total multi-label DDI dataset.

> wget http://tatonettilab.org/resources/nsides/TWOSIDES.csv.xz

For the DrugBank dataset, you can obtained from https://go.drugbank.com/releases/latest

##### 2. Knowledge Graph file:

DRKG : 'train.tsv' which is defined as "h \t r \t t" id form

you can get DRKG dataset from https://github.com/gnn4dr/DRKG or just download files using command below:

> wget https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz

##### 3. Customize your dataset:

you can use your dataset follow the form as ours.

you can define your KG dataset file name as "kg2id.txt" and DDI dataset name as "DDI_pos_neg.txt".

## Pretrain your data Preparation

##### 1. Graph-based drug embedding:

you can use "pretrain_smiles_embedding.py" file to generate your drug embedding, which is at last shown at ".npy" form in the data/DRKG directory.

try this code:

> python pretrained_smiles_embedding.py -fi ./data/DRKG/your_smiles_file.csv -m gin_supervised_masking -fo csv -sc smiles

-fi : your smiles file position

-fo : your file's type: txt or csv

-sc : the smiles "column name" in your smiles file. 

when you run this code, you can then get the final Graph-based drug embedding.

For convenient, the drug smiles order(in the "npy" file) is consistent with the KG entity, which means if you have 2322 drugs, and the grph-based embedding [ID：0-2321] is the same as the former 2322 KG entity [ID:0-2321]

## Environment Setting 

This code is based on Pytorch 3.6.5. You need prepare your virtual enviroment early.

## Running the code

You can run the following command to re-implement our work:

> python main.py

## Contact

If you have any question, just contact us.  
