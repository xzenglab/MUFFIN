# coding=utf-8
import pandas as pd
import numpy as np
import sys
import re

import csv
import re
import random


def extract_drugbank_from_drkg(file):
    entity_map = {}
    rel_map = {}
    train = []
    df = pd.read_csv(file, sep="\t", header=None)
    triples = df.values.tolist()
    # print(len(triples))

    for i in range(len(triples)):
        drug_1, relation, drug_2 = triples[i][0], triples[i][1], triples[i][2]
        result = re.match(r'DRUGBANK::', relation)
        if result and relation not in ['DRUGBANK::ddi-interactor-in::Compound:Compound']:
            l1 = "{}{}{}{}{}\n".format(drug_1, '\t', relation, '\t', drug_2)
            print(l1)
            train.append(l1)

    with open("./data/DRKG/drugbank_train_raw.tsv", "w+") as f:
        f.writelines(train)
    n_kg = len(train)
    # 45519个三元组
    print(n_kg)


def extract_hetionet_from_drkg(file):
    entity_map = {}
    rel_map = {}
    train = []
    df = pd.read_csv(file, sep="\t", header=None)
    triples = df.values.tolist()
    # print(len(triples))

    for i in range(len(triples)):
        drug_1, relation, drug_2 = triples[i][0], triples[i][1], triples[i][2]
        result = re.match(r'Hetionet::', relation)
        if result and relation not in ['Hetionet::CrC::Compound:Compound']:
            l1 = "{}{}{}{}{}\n".format(drug_1, '\t', relation, '\t', drug_2)
            print(l1)
            train.append(l1)

    with open("./data/DRKG/hetionet_train_raw.tsv", "w+") as f:
        f.writelines(train)
    n_kg = len(train)
    # 45519个三元组
    print(n_kg)


def delete_drugbank_hetionet_ddi_from_drkg(infile, outfile):
    train = []
    df = pd.read_csv(infile, sep="\t", header=None)
    triples = df.values.tolist()
    # print(len(triples))

    for i in range(len(triples)):
        drug_1, relation, drug_2 = triples[i][0], triples[i][1], triples[i][2]
        result = re.match(r'DRUGBANK::', relation)
        result2 = re.match(r'Hetionet::', relation)
        if relation not in ['DRUGBANK::ddi-interactor-in::Compound:Compound',
                                                    'Hetionet::CrC::Compound:Compound']:
            l1 = "{}{}{}{}{}\n".format(drug_1, '\t', relation, '\t', drug_2)
            print(l1)
            train.append(l1)

    with open(outfile, "w+") as f:
        f.writelines(train)
    n_kg = len(train)
    # 45519个三元组
    print(n_kg)
    print('generate triple files without DDI pairs successfully!')


def _get_id(dict, key):
    id = dict.get(key, None)
    if id is None:
        id = len(dict)
        dict[key] = id
    return id


def generate_entity_relation_id_file(delimiter, smilesfile, file, file2, entity_file, relation_file, triple_file,
                                     ddi_name_file, ddi_id_file):
    entity_map = {}
    rel_map = {}
    train = []

    infile = open(smilesfile, 'r')
    approved_drug_list = set()
    for line in infile:
        drug = line.replace('\n', '').replace('\r', '').split('\t')
        # approved_drug_list.add('Compound::' + drug[0])
        # drug_id = _get_id(entity_map, 'Compound::' + drug[0])
        approved_drug_list.add(drug[0])
        drug_id = _get_id(entity_map, drug[0])
        print(drug, drug_id)

    df = pd.read_csv(file, sep="\t", header=None)
    triples = df.values.tolist()

    for i in range(len(triples)):
        src, rel, dst = triples[i][0], triples[i][1], triples[i][2]
        src_id = _get_id(entity_map, src)
        dst_id = _get_id(entity_map, dst)
        rel_id = _get_id(rel_map, rel)
        train_id = "{}{}{}{}{}\n".format(src_id, delimiter, rel_id, delimiter, dst_id)
        print(train_id)
        train.append(train_id)

    entities = ["{}{}{}\n".format(val, delimiter, key) for key, val in sorted(entity_map.items(), key=lambda x: x[1])]
    with open(entity_file, "w+") as f:
        f.writelines(entities)
    n_entities = len(entities)

    relations = ["{}{}{}\n".format(val, delimiter, key) for key, val in sorted(rel_map.items(), key=lambda x: x[1])]
    with open(relation_file, "w+") as f:
        f.writelines(relations)
    n_relations = len(relations)

    with open(triple_file, "w+") as f:
        f.writelines(train)
    n_kg = len(train)

    # the code down from here is just extract DDI pairs from DRKG and transfer it into id form.
    df_2 = pd.read_csv(file2, sep="\t", header=None)
    triples2 = df_2.values.tolist()

    ddi_name = []
    ddi = []
    ddi_name_list = set()
    for i in range(len(triples2)):
        src, rel, dst = triples2[i][0], triples2[i][1], triples2[i][2]
        if rel in ['DRUGBANK::ddi-interactor-in::Compound:Compound', 'Hetionet::CrC::Compound:Compound']:
            # 存储有SMILES的DDI信息
            if src in approved_drug_list and dst in approved_drug_list:
                ddi_pair_single = "{}{}{}\n".format(src, '\t', dst)
                ddi_pair_single_reverse = "{}{}{}\n".format(dst, '\t', src)
                # print(ddi_pair_single)
                
                if ddi_pair_single not in ddi_name_list:
                    ddi_name_list.add(ddi_pair_single)
                    ddi_name.append(ddi_pair_single)
                    drug_id_1 = _get_id(entity_map, src)
                    drug_id_2 = _get_id(entity_map, dst)
                    ddi_id = "{}{}{}\n".format(drug_id_1, delimiter, drug_id_2)
                    ddi.append(ddi_id)
                    
                if ddi_pair_single_reverse not in ddi_name_list:
                    ddi_name_list.add(ddi_pair_single_reverse)
                    ddi_name.append(ddi_pair_single_reverse)
                    drug_id_1 = _get_id(entity_map, src)
                    drug_id_2 = _get_id(entity_map, dst)
                    ddi_id_reverse = "{}{}{}\n".format(drug_id_2, delimiter, drug_id_1)
                    ddi.append(ddi_id_reverse)
                


    with open(ddi_name_file, "w+") as f:
        f.writelines(ddi_name)

    with open(ddi_id_file, 'w+') as f:
        f.writelines(ddi)

    n_ddi = len(ddi)

    print(n_entities)
    print(n_relations)
    print(n_kg)
    print(n_ddi)
    print('You have done it successfully!')


def judge_drug_in_kg():
    infile = open('./data/DRKG/drugbank_smiles.txt', 'r')
    approved_drug_list = []
    for line in infile:
        drug = line.replace('\n', '').replace('\r', '').split('\t')
        approved_drug_list.append(('Compound::' + drug[0], drug[1]))
    # 8807
    print(len(approved_drug_list))

    file = './data/DRKG/drugbank_hetionet_train_raw.tsv'
    entity = []
    df = pd.read_csv(file, sep="\t", header=None)
    triples = df.values
    entity.extend(triples[:, 0])
    entity.extend(triples[:, 2])
    print(len(entity))
    entity = list(set(entity))
    print(len(triples))
    print(len(entity))

    sift = []
    for (d, s) in approved_drug_list:
        if d in entity:
            l1 = "{}{}{}\n".format(d, '\t', s)
            sift.append(l1)
    print(len(sift))
    with open("./data/DRKG/drugbank_smiles_sift.txt", 'w+') as f:
        f.writelines(sift)


def generate_positive_pairs(DDI_positive_file):
    druglist_left = []
    druglist_right = []
    DDI = {}
    DDI_pos_num = 0
    with open(DDI_positive_file, 'r') as csvfile:
        for row in csvfile:
            DDI_pos_num += 1
            lines = row.replace('\n', '').replace('\r', '').split('\t')
            drug_1 = lines[0]
            drug_2 = lines[1]
            if drug_1 not in DDI:
                DDI[drug_1] = []
            DDI[drug_1] += [drug_2]
            druglist_left.append(drug_1)
            druglist_right.append(drug_2)
        druglist_left = list(set(druglist_left))
        druglist_right = list(set(druglist_right))
    print('generate_positive_pairs.')
    return druglist_left, druglist_right, DDI, DDI_pos_num


def generate_negative_pairs(DDI_pos, druglist_right):
    DDI_neg = {}
    druglist_neg = druglist_right[:]
    for k, v in DDI_pos.items():
        if k not in DDI_neg:
            DDI_neg[k] = []
        if k in druglist_neg:
            druglist_neg.remove(k)
        for i in v:
            if i in druglist_neg:
                druglist_neg.remove(i)
        DDI_neg[k] = druglist_neg
        druglist_neg = druglist_right[:]
    print('generate_negative_pairs.')
    return DDI_neg


def generate_neg_data(DDI_pos_num, DDI_neg, output_negative):
    DDI_pos_neg = dict()
    drug1_name = 'Drug_1'
    drug2_name = 'Drug_2'
    drug_index = 0
    c = 0

    for drug_1, v in DDI_neg.items():
        if drug_1 not in DDI_pos_neg:
            DDI_pos_neg[drug_index] = dict()
        for drug_2 in v:
            DDI_pos_neg[drug_index] = dict()
            DDI_pos_neg[drug_index][drug1_name] = drug_1
            DDI_pos_neg[drug_index][drug2_name] = drug_2
            drug_index += 1

    resultList = random.sample(range(0, drug_index), DDI_pos_num)
    for i in resultList:
        drug_1_id = DDI_pos_neg[i][drug1_name]
        drug_2_id = DDI_pos_neg[i][drug2_name]
        c += 1
        outline = drug_1_id + '\t' + drug_2_id + '\t' + str(0) + '\n'
        output_negative.write(outline)
    output_negative.close()
    print(c)
    print('Yep! Finish generate_neg_data_file.')


def concate_pos_neg_data(infile_1, infile_2, outputfile):
    c = 0
    for line in infile_1:
        c += 1
        lines = line.replace('\n', '').replace('\r', '').split('\t')
        drug_1 = lines[0]
        drug_2 = lines[1]
        drug_state = 1
        outline = "{}\t{}\t{}\n".format(drug_1, drug_2, drug_state)
        # outline = drug_1 + '\t' + drug_2 + '\t' + drug_state + '\n'
        outputfile.write(outline)
    print('pos data finish!')
    for line in infile_2:
        c += 1
        lines = line.replace('\n', '').replace('\r', '').split('\t')
        drug_1 = lines[0]
        drug_2 = lines[1]
        drug_state = int(lines[2])
        outline = "{}\t{}\t{}\n".format(drug_1, drug_2, drug_state)
        # outline = drug_1 + '\t' + drug_2 + '\t' + drug_state + '\n'
        outputfile.write(outline)
    print('neg data finish!')
    outputfile.close()
    print(c)
    print('DDI_pos_neg.txt done!')


if __name__ == '__main__':
    # 根据下载的DRKG文件，生成新的实体、关系和三元组文件
    # 同时从DRKG中提取DDI，用于二分类任务，这里的药物都是approved，确保该药物有Graph-based embedding信息
    drkg_file = './data/DRKG/drkg.tsv'
    combine_file = './data/DRKG/train_without_ddi_raw.tsv'
    entity_file = './data/DRKG/entities.tsv'
    relation_file = './data/DRKG/relations.tsv'
    triple_file = './data/DRKG/train.tsv'
    ddi_pos_file = './data/DRKG/ddi_facts_pos.txt'
    ddi_pos_id_file = './data/DRKG/ddi_facts_pos_id.txt'
    # this file looks like "Compound::DB00119	CC(=O)C(O)=O"
    # if your drug name not this form, just map your name into the DRKG'name form
    # maybe you just need to append "Compound::" in the "DB..."(DrugBank form) name
    smiles_file = './data/DRKG/drugname_smiles.txt'

    # extract_drugbank_from_drkg(drkg_file)
    # extract_hetionet_from_drkg(drkg_file)

    delete_drugbank_hetionet_ddi_from_drkg(drkg_file, combine_file)
    generate_entity_relation_id_file('\t', smiles_file, combine_file, drkg_file, entity_file, relation_file,
                                     triple_file, ddi_pos_file, ddi_pos_id_file)

    # judge_drug_in_kg()

    """
    # generate negative DDI pairs this data is written before，the original code is in another server, sadly, 
    # that sever can not work again.. so just waiting for me(server), maybe you can just use your own code to 
    # generate the negative DDI pairs. 
    
    DDI_negfile = 'DDI_neg.txt'
    DDI_outfile = 'DDI_pos_neg.txt'
    drug_list_left, drug_list_right, DDI_positive, DDI_positive_num = generate_positive_pairs(ddi_pos_id_file)
    DDI_negative = generate_negative_pairs(DDI_positive, drug_list_right)
    generate_neg_data(DDI_positive_num, DDI_negative, open(DDI_negfile, 'w'))
    concate_pos_neg_data(open(ddi_pos_id_file, 'r'), open(DDI_negfile, 'r'),
                         open(DDI_outfile, 'w'))
    """
