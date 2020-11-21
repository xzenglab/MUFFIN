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

## Environment Setting 

This code is based on Pytorch 3.6.5. You need prepare your virtual enviroment early.

## Running the code

You can run the following command to re-implement our work:

> python main.py

## Contact

If you have any question, just contact us.  
