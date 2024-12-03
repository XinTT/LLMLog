# import openai
import json
import os
import pandas as pd
from tqdm import tqdm
# from openai.embeddings_utils import get_embedding
import argparse
import torch
import re

import torch.nn.functional as F
from llm2vec import LLM2Vec

import torch
import time


def processLog(logs,idx,mode='default'):
    if mode == 'basic':
        return logs[idx]['parsing']
    else:
        return logs[idx]['parsing_parsing_parameters']
l2v = LLM2Vec.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
)

parser = argparse.ArgumentParser(description="Runs batchtest.py with a datasetname.")
input_dir = "../../data/loghub_2k"
output_dir = "embeddings/"
# output_dir = "drain_embeddings/"
log_list = ['Thunderbird','BGL','Linux','Zookeeper']

time1 = time.time()
for logs in log_list:
    # logs = 'webank'
    embedding = dict()
    print("Embedding " + logs + "...")
    i = pd.read_csv(input_dir + '/' + logs + '/' + logs+ "_2k.log_structured.csv")
    templates = i['EventTemplate']
    contents = i['Content']
    with open('processedLogs/parsing_'+logs+'.json','r') as f:
        dic = json.load(f)
    for i,log in tqdm(enumerate(contents)):
        
        
        plog = processLog(dic,i,'basic')
        for word in plog:
            if word not in embedding:
                q_reps = l2v.encode([word])
                q_reps_norm = torch.nn.functional.normalize(q_reps, p=2, dim=1).squeeze()
                embedding[word] = q_reps_norm.tolist()
    with open(logs+'_words_embedding.json','w') as f:
        json.dump(embedding,f)

    similar_words = {}
    words = {}
    duplicate = set()
    for item in embedding:
        similar_words[item] = set()
        words[item] = {}
    for item in embedding:
        for item1 in embedding:
            if item == item1:
                continue
            dist =  float(F.cosine_similarity(torch.Tensor(embedding[item]).to('cuda').view(1,-1), torch.Tensor(embedding[item1]).to('cuda').view(1,-1))[0])
            if dist >= 0.8:
                similar_words[item].add(item1)
            words[item][item1] = dist
            duplicate.add((item,item1))
    for k in similar_words:
        similar_words[k] = list(similar_words[k])
    # for k in words:
    #     words[k] = list(words[k])
    with open(logs+'_similar_words.json','w') as f:
        json.dump(similar_words,f)
    with open(logs+'_words_similarity.json','w') as f:
        json.dump(words,f)
    time2 = time.time()
    print(time2-time1)
