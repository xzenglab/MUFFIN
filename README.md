# MUFFIN

## Overview

This repository contains source code for our paper "MUFFIN: Multi-Scale Feature Fusion for Drug–Drug Interaction Prediction".

## Dataset Preparation

You need to provide datasets defined as below:

#### 1. Approved Drug SMILES file:

ALl we want to consider is the FDA-approved drug, the form is look like:

> Compound::DB00119	CC(=O)C(O)=O

In our work, we have 2322 drugs. Just 2322 lines in this file.

#### 2. DDI dataset file:

'DDI_pos_neg.txt': store the DDI dataset, the form is "drug1 \t drug2 \t type". For binary data: type is in {0,1}, for the multi-class DDI dataset "multi_ddi_sift.txt", type ranges from 0 to 80, and for the multi-label dataset, it is in {0-200}.

For the TWOSIDES dataset, you can obtained from http://tatonettilab.org/offsides/

you can also use following command to get the total multi-label DDI dataset.

> wget http://tatonettilab.org/resources/nsides/TWOSIDES.csv.xz

For the DrugBank dataset, you can obtained from https://go.drugbank.com/releases/latest

#### 3. Knowledge Graph file:

DRKG : 'train.tsv' which is defined as "h \t r \t t" id form

you can get DRKG dataset from https://github.com/gnn4dr/DRKG or just download files using command below:

> wget https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz

Now, you get "drkg.tsv" file, put it into the "./data/DRKG" directory. Next you need to change the entity ID inconsitent with your drug smiles file (in first step, you get it). 

just run the code below:

> python process_raw_DRKG.py

After that, you can get "entities.tsv","relation.tsv" and "train.tsv".

#### 4. Customize your dataset:

you can use your dataset follow the form as ours.

you can define your KG dataset file name as "kg2id.txt" and DDI dataset name as "DDI_pos_neg.txt".

## Pretrain your data Preparation

#### 1. Graph-based drug embedding:

you can use "pretrain_smiles_embedding.py" file to generate your drug embedding, which is at last shown at ".npy" form in the data/DRKG directory.

try this code:

> python pretrained_smiles_embedding.py -fi ./data/DRKG/your_smiles_file.csv -m gin_supervised_masking -fo csv -sc smiles

-fi : your smiles file position

-fo : your file's type: txt or csv

-sc : the smiles "column name" in your smiles file. 

when you run this code, you can then get the final Graph-based drug embedding.

tips: For convenient, the drug smiles order(in the "npy" file) is consistent with the KG entity, which means if you have 2322 drugs, and the graph-based embedding \[ID：0-2321\] is the same as the 2322 former KG entity embedding \[ID:0-2321\]

#### 2. KG-based drug embedding:

We use DGL-KG tools to train our KG entities. if you want to generate KG-embedding, just run the following code according to your needs:

> python pretrain_kg_embedding.py --model_name TransE_l2 --dataset DRKG --data_path data/DRKG/ --data_files entities.tsv relations.tsv train.tsv --format udd_hrt --batch_size 2048 --neg_sample_size 128 --hidden_dim 100 --gamma 12.0 --lr 0.1 --max_step 100000 --log_interval 1000 --batch_size_eval 16 -adv --regularization_coef 1.00E-07 --test --num_thread 1 --gpu 1 2 --num_proc 2 --neg_sample_size_eval 10000 --async_update

Now, you can get the KG-based embedding named "DRKG_TransE_l2_entity.npy", just moved it into your dataset directory!

## Environment Setting 

This code is based on Pytorch 3.6.5. You need prepare your virtual enviroment early.

## Running the code

You can run the following command to re-implement our work:

> python main.py

what you need to do according to your customized file:
1. change “—graph_embedding_file” name with “your smiles file position”
2. change “—entity_embedding_file” name with “your entity file position”
3. change “—relation_embedding_file” name with “your relation file position”
4. change “—out_dim” name with “1 or 81”, which is depends on your task “binary or multi-class”

## Contact

If you have any question, just contact us.  
