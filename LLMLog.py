import json
import os
import math
import numpy as np
import pandas as pd
import re
import time
# import openai
# import tiktoken as tt
from tqdm import tqdm
from random import sample
from collections import Counter
import transformers
import torch
import torch.nn.functional as F
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import statistics
from openai import AzureOpenAI

# from openai.embeddings_utils import cosine_similarity
# the code is written based on DivLog from https://github.com/logpai/logparser

def contains_only_digits_and_non_letters(s):
  pattern = r'^[0-9\W]*$'
  return bool(re.match(pattern, s))

def dpp(kernel_matrix, max_length, epsilon=1E-10):
    item_size = kernel_matrix.shape[0]
    cis = np.zeros((max_length, item_size))
    di2s = np.copy(np.diag(kernel_matrix))
    selected_items = list()
    selected_item = np.argmax(di2s)
    # print(selected_item)
    selected_items.append(selected_item)
    while len(selected_items) < max_length:
        k = len(selected_items) - 1
        ci_optimal = cis[:k, selected_item]
        di_optimal = math.sqrt(di2s[selected_item])
        elements = kernel_matrix[selected_item, :]
        eis = (elements - np.dot(ci_optimal, cis[:k, :])) / di_optimal
        cis[k, :] = eis
        di2s -= np.square(eis)
        selected_item = np.argmax(di2s)
        selected_items.append(selected_item)
    return selected_items

def getDppIndex(log_emb_list, 
                item_size,    # log dataset size
                split_ratio):

    max_length = int(item_size * split_ratio)
    feature_vectors = np.array(log_emb_list) 

    # standarization no need for log embeddings
    feature_vectors /= np.linalg.norm(feature_vectors, axis=1, keepdims=True)

    # calculate similarity matrix of log embeddings
    similarities = np.dot(feature_vectors, feature_vectors.T) 

    t = time.time()
    result = dpp(similarities, max_length)
    result.sort()
    print('DPP algorithm running time: ' + '\t' + "{0:.4e}".format(time.time() - t))
    return result


def DPPsplit(log_list, groundtruth_template, candidate_idx):
    cand_logs = [log_list[idx] for idx in candidate_idx]
    cand_templates = [groundtruth_template[idx] for idx in candidate_idx]
    test_idx = []
    for i in range(len(log_list)):
      if i not in candidate_idx: test_idx.append(i)
    # print(candidate_idx)
    test_idx.sort()
    test_logs = [log_list[idx] for idx in test_idx]
    test_templates = [groundtruth_template[idx] for idx in test_idx]
    return test_logs, cand_logs, test_templates, cand_templates

# calculate parsing accuracy
def evaluatePA(groundtruth, result):
    # len(predicted_list) may smaller than len(groundtruth)
    length = len(result['template'])
    if length == 0: return 0
    correct = 0
    for i in range(length):
        if type(result['template'][i]) == float:
           continue
        if result['template'][i] == groundtruth.loc[groundtruth['Content'] == result['log'][i]]['EventTemplate'].values[0] or groundtruth.loc[groundtruth['Content'] == result['log'][i]]['EventTemplate'].values[0].replace(' ','') == result['template'][i].replace(' ',''):
            correct += 1
    return correct/length

# correctly identified templates over total num of identified template
def evaluatePTA(groundtruth, result):
    # generate a "template: log indexes list" mapping for groundtruth
    oracle_tem_dict = {}
    for idx in range(len(result['template'])):
        if groundtruth['EventTemplate'][idx].replace(' ','') not in oracle_tem_dict:
          oracle_tem_dict[groundtruth['EventTemplate'][idx].replace(' ','')] = [groundtruth['Content'][idx]]
        else: oracle_tem_dict[groundtruth['EventTemplate'][idx].replace(' ','')].append(groundtruth['Content'][idx])

    # generate mapping for identified template
    result_tem_dict = {}
    for idx in range(len(result['template'])):
        if type(result['template'][idx]) == float:
           continue
        if result['template'][idx].replace(' ','') not in result_tem_dict:
          result_tem_dict[result['template'][idx].replace(' ','')] = [result['log'][idx]]
        else: result_tem_dict[result['template'][idx].replace(' ','')].append(result['log'][idx])

    correct_num = 0
    for key in result_tem_dict.keys():
        if key not in oracle_tem_dict: continue
        else:
          if Counter(oracle_tem_dict[key]) == Counter(result_tem_dict[key]): correct_num += 1
    
    return correct_num/len(result_tem_dict)

# correctly identified templates over total num of oracle template
def evaluateRTA(groundtruth, result):
    # generate a "template: log indexes list" mapping for groundtruth
    oracle_tem_dict = {}
    for idx in range(len(result['template'])):
        if groundtruth['EventTemplate'][idx].replace(' ','') not in oracle_tem_dict:
          oracle_tem_dict[groundtruth['EventTemplate'][idx].replace(' ','')] = [groundtruth['Content'][idx]]
        else: oracle_tem_dict[groundtruth['EventTemplate'][idx].replace(' ','')].append(groundtruth['Content'][idx])

    # generate mapping for identified template
    result_tem_dict = {}
    for idx in range(len(result['template'])):
        if type(result['template'][idx]) == float:
           continue
        if result['template'][idx].replace(' ','') not in result_tem_dict:
          result_tem_dict[result['template'][idx].replace(' ','')] = [result['log'][idx]]
        else: result_tem_dict[result['template'][idx].replace(' ','')].append(result['log'][idx])

    correct_num = 0
    for key in oracle_tem_dict.keys():
        if key not in result_tem_dict: continue
        else:
          if Counter(oracle_tem_dict[key]) == Counter(result_tem_dict[key]): correct_num += 1
    
    return correct_num/len(oracle_tem_dict)


def word_dist(a, b,dic):
  if a == b:
    return 1
  if a not in dic:
    return 0
  if b not in dic[a]:
    return 0
  
  return dic[a][b]

def edit_distance_on_real_sequences(seq1, seq2, threshold=0.8,dic={}):
    len1 = len(seq1)
    len2 = len(seq2)
    
    dp = np.zeros((len1 + 1, len2 + 1))

    for i in range(len1 + 1):
        dp[i][0] = i  
    for j in range(len2 + 1):
        dp[0][j] = j  
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            dist = word_dist(seq1[i - 1][0], seq2[j - 1][0],dic)
            cost = 0 if dist < threshold else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,     
                dp[i][j - 1] + 1,     
                dp[i - 1][j - 1] + cost  
            )

    return dp[len1][len2]
  


def processLog(logs,idx,mode='default',keywords=set()):
    # print(1)
  if mode == 'basic':
    words = logs[idx]['parsing']
  else:
    words = logs[idx]['parsing_parsing_parameters']
  temp = []
  for word in words:
     if word not in keywords:
        temp.append(word)
  return temp

def find_replaced_characters(a, b, special_string='<*>'):
    replaced_indices = []  # 用于存储被替换的字符索引
    if '<*>:<*>:<*><*>:<*>' in b:
        b = b.replace('<*>:<*>:<*><*>:<*>','<*>:<*>:<*>:<*>:<*>:<*>')
    elif '<*><*><*>' in b:
        b = b.replace('<*><*><*>','<*>')
    elif '<*><*>' in b:
        b = b.replace('<*><*>','<*>')
    special_length = len(special_string)
    indices,length,keywords = find_special_indices(b, special_string)
    i = 0
    j = 0
    k = 0
    # print(keywords)
    matched = []
    cnt = 0
    while i!= len(a):
        # print(f'{b.startswith(special_string,j)} {j}')
        if b.startswith(special_string,j):
            # temp = a[i,i+length[cnt]]
            temp = ''
            if keywords[cnt] != '':
                # print(i)
                f = 0
                for idx in range(i,len(a)):
                    if keywords[cnt] in temp:
                        
                        # print(cnt)
                        matched.append(temp.replace(keywords[cnt],''))
                        j += len(special_string)+len(keywords[cnt])
                        cnt += 1
                        # print(temp)
                        f = 1
                        break
                    temp += a[idx]
                    # print(temp)
                # print(f == 0)
                # print(keywords[cnt] in temp)
                if f == 0 and keywords[cnt] in temp:
                    matched.append(temp.replace(keywords[cnt],''))
            else:
                
                matched.append(a[i:len(a)])
                cnt += 1
            # print(f'{i} {len(a)} {temp} {cnt}')
            if cnt == len(keywords):
                break
            # print(f"{i}_{a[i]} {j}_{b[j]}")
            i += len(temp)
            
        else:
            i += 1 
            j += 1
    return matched,indices
                
def find_special_indices(b, special_string='<*>'):
    indices = []
    index = b.find(special_string)
    start_indices = []
    while index != -1:
        start_indices.append(index+len(special_string))
        indices.append(index)
        index = b.find(special_string, index + len(special_string))  
    # print(indices)
    # print(start_indices)
    temp = []
    for i in range(len(indices)):
        if i + 1 == len(indices):
            temp.append(len(b) - (indices[i]+len(special_string)))
        else:
            temp.append(indices[i+1] - (indices[i]+len(special_string)))
    keywords = []
    for i in range(len(temp)):
        keywords.append(b[start_indices[i]:start_indices[i]+temp[i]])
    # print(keywords)
    return indices,temp,keywords

class ModelParser():
  def __init__(self, 
        log_path, 
        result_path, 
        map_path, 
        dataset,
        emb_path,
        cand_ratio,
        split_method, # random or DPP
        order_method, # random or KNN
        permutation,
        warmup, # warmup or not
        subname, # subname of the files
        evaluate, # evaluate or not
        single_annotation,
        hard_ratio,
        token_dist,
        budget,
        round,
        self_check,
        selection,
        lamb,
        threshold,
        prompt,
        dist_metric,
        model,
        key,
        budget_strategy
    ):
    # make_print_to_file(path='icl_log')
    self.log_path = log_path + "/{}/{}_2k.log_structured.csv".format(dataset,dataset)
    self.result_path = result_path
    self.cand_ratio = cand_ratio
    self.cand_num = int(2000*self.cand_ratio)
    self.budget = budget
    self.reselect_num = self.budget - self.cand_num
    
    self.dataset = dataset
    self.emb_path = emb_path + "/{}.json".format(dataset)
    self.hard_ratio = hard_ratio
    
    self.split_method = split_method
    self.order_method = order_method
    self.permutation = permutation
    self.warmup = warmup
    self.subname = subname
    self.evaluate = evaluate
    self.single_annotation = single_annotation
    
    self.token_dist = token_dist
    self.round = round
    self.self_check = '_selfcheck' if self_check == 'yes' else ''
    self.selection = selection
    self.threshold = threshold
    self.lamb = lamb
    self.prompt = prompt
    self.dist_metric = dist_metric
    self.budget_strategy = budget_strategy
    print(self.single_annotation)
    # split candidate set
    log_list = self.extractCsvContent(self.log_path)
    groundtruth_template = self.extractCsvTemplate(self.log_path)
    print(f'{self.selection} {self.token_dist} {self.permutation}')
    if self.selection == 'max_cover':
      
      self.map_path = map_path + "/{}_{}_{}_{}_{}_des_lookupmap{}_m{}.json".format(self.cand_num, self.budget, cand_ratio,dataset,self.hard_ratio,self.self_check,self.round)
     
    else:
      
      self.map_path = map_path + "/{}_{}_{}_{}_{}_des_lookupmap_{}_{}mod2{}_m{}_word_{}.json".format(self.cand_num, self.budget, cand_ratio,dataset,self.token_dist,self.prompt,self.dist_metric,self.self_check,self.round,self.lamb)
    self.log_list = log_list
    if self.cand_ratio > 0:
      self.log_test, self.log_cand, self.gt_test, self.gt_cand,selected_indices = self.splitCandidates(log_list, groundtruth_template, self.cand_ratio, self.split_method)
      # log_cand_unselected, log_cand_selected, gt_cand_unselected, gt_cand_selected, selected_cand_indices = self.splitCandidates(log_cand, gt_cand, self.label_ratio/self.cand_ratio, self.split_method)
    else:
      self.log_test = log_list
      self.gt_test = groundtruth_template
      selected_indices = []
      self.gt_cand = []
      self.log_cand = []
    # build lookup map
    self.selected_indices = selected_indices
    
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    self.tokenizer = AutoTokenizer.from_pretrained(model_id)
    if 'set_cover' in self.prompt:
      with open(dataset+'_similar_words.json','r') as f:
        self.similar_words = json.load(f)
      with open(dataset+'_words_similarity.json','r') as f:
        self.sim_dic = json.load(f)
    else:
      self.similar_words = {}
      self.sim_dic = {}
    if self.token_dist == 'prob':
      if self.single_annotation == '':
        
        self.confidentLLM = None
      else:
        self.confidentLLM = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')
    elif self.token_dist == 'weighted':
      self.confidentLLM = AutoModelForCausalLM.from_pretrained(model_id, device_map='auto')
    
    
        
    self.pipeline = AzureOpenAI(
        api_key=key,
        api_version='2023-05-15',
        azure_endpoint='https://hkust.azure-api.net'
    )
    
    self.model_name = "gpt-4o"
    with open('processedLogs/parsing_'+self.dataset,'r') as f:
      self.word_file = json.load(f)
    epoch = 0
    print(f'{len(self.selected_indices) } {budget}')
    if self.selection == 'max_cover':
      
      cand_path = 'cands/'+str(self.budget)+'_'+str(self.hard_ratio)+'_'+self.dataset+'_des_dpp_cands'+self.self_check+'_m'+str(self.round)+'.json'
      
    else:
     
      cand_path = 'cands/'+str(self.budget)+'_'+str(self.hard_ratio)+'_'+self.dataset+'_des_dpp_'+self.prompt+'_'+self.dist_metric+'_cands'+self.self_check+'_m'+str(self.round)+'_word_parsing_'+str(self.lamb)+'.json'
        
    print(f'{os.path.exists(self.map_path)} {self.map_path} {os.path.exists(cand_path)} {cand_path}')
    self.gts = pd.read_csv('/export/data/kb_group_shares/logparser/logparser-main/data/loghub_2k/'+self.dataset+'/'+self.dataset+'_2k.log_structured.csv')
    if os.path.exists(self.map_path) and os.path.exists(cand_path):
      self.lookUpMap = self.buildLookupMap(self.map_path)
      with open(cand_path,'r') as f:

        self.log_cand = json.load(f)
      self.log_test = log_list
      self.gt_test = groundtruth_template
      self.gt_cand = []
      for log in self.log_cand:
        gt = self.gts.loc[self.gts['Content'] == log]['EventTemplate'].values[0]
        self.gt_cand.append(gt)
    
    else:
      if self.selection == 'max_cover':
        if len(self.selected_indices) == self.budget:
          self.lookUpMap = self.buildLookupMap(self.map_path)
        else:
          while len(self.selected_indices) < self.budget:
            print(f'{epoch} {len(self.selected_indices)}')
            if epoch == 0:
              cand_size = self.annotation(init=True,iter_num=epoch,cover=[])
              prev_cand = cand_size
            else:
              cand_size = self.annotation(init=False,iter_num=epoch,cover=[])
            if cand_size == 0 and prev_cand == 0 and self.token_dist == 'dist':
              break
            prev_cand = cand_size

            
            epoch += 1
            if self.single_annotation == '':
              break
        if len(self.gt_cand) < self.budget:
          self.log_test, temp_log_cand, self.gt_test, temp_gt_cand,temp_selected_indices = self.splitCandidates(self.log_test, self.gt_test, (self.budget-len(self.gt_cand))/(len(log_list)-len(self.gt_cand)), 'random')
          self.log_cand = self.log_cand + temp_log_cand
          self.gt_cand = self.gt_cand + temp_gt_cand
        self.log_test = log_list
        self.gt_test = groundtruth_template
        self.lookUpMap = self.generateMap(self.map_path)
      else:
        cover = set()
        self.lookUpMap = self.generateMap(self.map_path)
        keywords = set()
        for log in self.log_cand:
          words = set(processLog(self.word_file,log.strip()))
          keywords.update(words)
        print(len(cover))
        while len(self.selected_indices) < self.budget:
          print(f'{epoch} {len(self.selected_indices)} {len(cover)}')
          if epoch == 0:
            cover,keywords = self.annotation(init=False,iter_num=epoch,cover=set(),keywords=set())
          else:
            cover,keywords = self.annotation(init=False,iter_num=epoch,cover=set(),keywords=keywords)
          epoch += 1
        self.log_test = log_list
        self.gt_test = groundtruth_template
        
        self.lookUpMap = self.generateMap(self.map_path)
      
      
      if self.selection == 'max_cover':
        
        with open('cands/'+str(self.budget)+'_'+str(self.hard_ratio)+'_'+self.dataset+'_des_dpp_cands'+self.self_check+'_m'+str(self.round)+'.json','w') as f:
          json.dump(self.log_cand,f)
       
      else:
        
        with open('cands/'+str(self.budget)+'_'+str(self.hard_ratio)+'_'+self.dataset+'_des_dpp_'+self.prompt+'_'+self.dist_metric+'_cands'+self.self_check+'_m'+str(self.round)+'_word_parsing_'+str(self.lamb)+'.json','w') as f:
          json.dump(self.log_cand,f)
          
    self.confidentLLM = None
    import gc
    gc.collect()

  def buildLookupMap(self, map_path):
    if (os.path.exists(map_path)): 
      print("Loading look up map of {} ...".format(self.dataset))
      with open(map_path, "r") as file:
            return json.load(file)
    else:
      
      return self.generateMap(map_path)

  def extractCsvContent(self, groundtruth_path):
      dataframe = pd.read_csv(groundtruth_path)
      content_list = dataframe['Content'].values.tolist()
      return content_list

  def extractCsvTemplate(self, groundtruth_path):
      dataframe = pd.read_csv(groundtruth_path)
      template_list = dataframe['EventTemplate'].values.tolist()
      return template_list

  def splitCandidates(self, log_list, groundtruth_template, cand_ratio, method="random"):
      
      
      file = open(self.emb_path, "r")
      emb_map = json.load(file)
      file.close()
      log_embs = []
      for log in log_list:
        log_embs.append(emb_map[log])
      print(f"length of log embs is {len(log_embs)}")
      candidate_idx = getDppIndex(log_embs, 2000, cand_ratio)
      log_test, log_cand, gt_test, gt_cand = DPPsplit(log_list, groundtruth_template, candidate_idx)
      # log_test = log_test + log_cand
      # gt_test = gt_test + gt_cand
      log_test = log_test
      # print(len(log_test))
      gt_test = gt_test 
      return log_test, log_cand, gt_test, gt_cand,candidate_idx

  
  def generateMap(self, look_up_map_path):
      # get embeddings from embedding json file
      print('Generating lookup map for {} ...'.format(self.dataset))
      

      lookUpMap = {}
      
      if self.prompt == 'set_cover':
        
        
        candidates, sorted_list, covered,covered_var,prev = self.prompt_set_cover(self.log_test[test_idx])
        lookUpMap[self.log_test[test_idx]] = {'logs':sorted_list,'keywords':list(covered),'var':list(covered_var),'prev':prev}
      else:
        dis_dict = {}
        with open(self.emb_path, "r") as file:
          emb_map = json.load(file)

        test_embs = [emb_map[log] for log in self.log_test]
        cand_embs = [emb_map[log] for log in self.log_cand]
        # print(len(self.log_test))
        # print(len(self.log_cand))
        for test_idx in tqdm(range(len(self.log_test))):
          dis_dict = {}
          for cand_idx in range(len(self.log_cand)):
            dis_dict[F.cosine_similarity(torch.Tensor(test_embs[test_idx]).to('cuda').view(1,-1), torch.Tensor(cand_embs[cand_idx]).to('cuda').view(1,-1))[0]] = cand_idx
          sorted_list = []
          for key in sorted(dis_dict, reverse=True): 
            sorted_list.append(dis_dict[key])
          lookUpMap[self.log_test[test_idx]] = {'logs':sorted_list,'keywords':[]}
      # write the map into a json file
      with open(look_up_map_path, 'w') as file:
        file.write(json.dumps(lookUpMap))
      return lookUpMap

  def getNearest(self, log, N=5):
      # if self.selection = 'max_cover':
      cand_list = self.lookUpMap[log]['logs']
      # else:
      #   cand_list = self.lookUpMap[log]
      if self.order_method == 'random':
        return sample(cand_list, N)
      # return the idexes of most similar N log candidates
      elif self.order_method == 'KNN':
        shift = 0
        if len(cand_list) == 0:
          return []
        if type(cand_list[0]) == tuple or type(cand_list[0]) == list:
          result = []
          for idx in range(len(cand_list)):
            result.append(cand_list[idx][0])
        else:
          result = cand_list[0:N]
        while log in result:
          shift += 1
        # print(shift)
        #   result = cand_list[shift:N+shift]
        if self.permutation == 'ascend':
          return result
        elif self.permutation == 'descend':
          result.reverse()
          return result
        elif self.permutation == 'random':
          result = sample(result, N)
          return result
  # generate a prompt in str for a specific log message
  def generatePrompt(self, log, nearest_num=5):
      # if self.prompt == 'max_cover':
      #   nearest_num = 
      idxes = self.getNearest(log, nearest_num)
      # print(idxes)
      prev = False
      if self.selection != 'max_cover' and self.prompt != 'max_cover':
        # print(self.lookUpMap[log])
        identified_keywords = self.lookUpMap[log]['keywords']
        identified_var = self.lookUpMap[log]['var']
        if 'prev' in self.lookUpMap[log]:
          prev = self.lookUpMap[log]['prev']
      else:
        identified_keywords = ''
        identified_var = ''
      prompt = ""
    
      
      gt_cand = self.gt_cand
      log_cand = self.log_cand
  
      for i in range(len(idxes)-1,-1,-1):
        if 'set_cover' in self.prompt:
          keywords = processLog(self.word_file,log_cand[idxes[i]].strip())
          # print(keywords)
          varwords,var_indices = find_replaced_characters(log_cand[idxes[i]].strip(), gt_cand[idxes[i]].strip(), special_string='<*>')
        if self.selection != 'max_cover':
          prompt = prompt + "<prompt>:" + log_cand[idxes[i]].strip() + \
                '\n<keywords>: '+','.join(keywords)+'\n<variable>: '+','.join(varwords)+'\n<extraction>: <START> ' +gt_cand[idxes[i]].strip() + ' <END>\n\n' 
        else:
          prompt = prompt + "<prompt>:" +log_cand[idxes[i]].strip() + '\n<extraction>: <START> ' + gt_cand[idxes[i]].strip() + ' <END>\n\n' 
      if self.prompt == 'set_cover':
        return prompt, '', identified_keywords, identified_var
      elif 'set_cover' not in self.prompt:
        return prompt, '', [],[]
         

  def writeResult(self, result, path, limit):
      output = pd.DataFrame(data={"log": self.log_test[:limit], "template": result})
      output.to_csv(path, index=False)

  def writecheckResult(self, result, check_template,path, limit):
      output = pd.DataFrame(data={"log": self.log_test[:limit], "template": result,"check_template":check_template})
      output.to_csv(path, index=False)

  # extract result from model's response
  def extractResultTemplate(self, text):
      # this pattern is for ChatGPT
      # pattern = re.compile('<START> <Event\d> (.+) <END>')
      pattern = re.compile('<START>(.+)<END>')
      # findall return a list
      result = pattern.findall(text)
      if (len(result)): return result[0]
      else: return ""


  def BatchParse(self, limit, N=5):
      # list to store the model's parsing on each log message
      # enc = tt.encoding_for_model(model)
    
      
      print("Result file does not exist, generating result ...")
      overall_results = []
      no_cnt = {}
      time1 = time.time()
      for line_idx in tqdm(range(len(self.log_test[:limit]))):
        re_id = 0
        temperature = 0.6
        parsing_result = ["" for k in range(3)]
        parsing_result[0] = self.log_test[line_idx]
        if line_idx >= limit: break
        line = self.log_test[line_idx]
        prompt = ''
        if self.log_test[line_idx] in self.log_cand and 'set_cover' in self.prompt:
          result = self.gt_cand[self.log_cand.index(self.log_test[line_idx])]
          no_cnt[line_idx] = {'log':self.log_test[line_idx],'result':result}
        else:

          prompt, similarist_gt,keywords,var = self.generatePrompt(self.log_test[line_idx], nearest_num=5)
          instruction = "For each log after <prompt> tag, extract one log template according to keywords\
    (construct the template based on keywords, identify the remaining tokens are keywords or not, reserve the keywords and substitute non keywords as <*>)\
    and put the template after <extraction> tag and between <START> and <END> tags."
          
          
          try:
            response = self.pipeline.chat.completions.create(
                    model=self.model_name,  
                    messages=[{"role": "system", "content": "You need to extract template from logs."},
                                {"role": "user", "content":instruction+'\n'+prompt + "<prompt>:" + line.strip() + "\n<extraction>: "}]
                )
            result = self.extractResultTemplate(response.choices[0].message.content)
            # result = error_logs[self.log_test[line_idx]]
            # result = ""
          except Exception as e: # if exception occurs
            print(e)
            result = ''
          # result = ''
          
          if self.dataset == 'Mac'and self.prompt == 'set_cover':
            no_cnt[line_idx] = {'log':self.log_test[line_idx],'keywords':list(keywords),'variable':list(var),'prompt':prompt,'result':result}
          else:
            no_cnt[line_idx] = {'log':self.log_test[line_idx],'keywords':list(keywords),'variable':list(var),'prompt':prompt,'result':result}
          overall_results.append(parsing_result) 
      # print(no_cnt)
      time2 = time.time()
      with open(self.dataset+'_'+self.selection+'_'+self.token_dist+'_'+self.prompt+'_'+str(self.round)+'_'+str(self.lamb)+'_time.json','w') as f:
        json.dump({'time':time2-time1},f)
      print("Result file generated.")
      if self.threshold != 0.29:
        
        with open('results/'+self.dataset+'_'+self.selection+'_'+self.token_dist+'_'+self.prompt+'_'+str(self.round)+'_threshold'+str(self.threshold)+'.json','w') as f:
          json.dump(no_cnt,f,indent=2)
      else:
        if self.lamb!=0.5:
          with open('results/'+self.dataset+'_'+self.selection+'_'+self.token_dist+'_'+self.prompt+'_'+str(self.round)+'_'+str(self.lamb)+'.json','w') as f:
            json.dump(no_cnt,f,indent=2)
        else:
          if self.budget_strategy != 'avg':
            with open('results/'+self.dataset+'_'+self.selection+'_'+self.token_dist+'_'+self.prompt+'_'+str(self.round)+'_'+self.budget_strategy+'.json','w') as f:
              json.dump(no_cnt,f,indent=2)
          else:
            with open('results/'+self.dataset+'_'+self.selection+'_'+self.token_dist+'_'+self.prompt+'_'+str(self.round)+'_'+str(self.lamb)+'_error.json','w') as f:
              json.dump(no_cnt,f,indent=2)
      


    
  

  def annotation(self,init,iter_num,cover=set(),keywords=set()):
    log_list = self.extractCsvContent(self.log_path)
    groundtruth_template = self.extractCsvTemplate(self.log_path)
    unselected_indices = []
    # selected_indices = []
    for i in range(len(log_list)):
      if i not in self.selected_indices:
        unselected_indices.append(i)
    print(len(unselected_indices))
    file = open(self.emb_path, "r")
    emb_map = json.load(file)
    file.close()
    log_embs = []
    for log_idx, log in enumerate(log_list):
      # if log_idx in unselected_indices:
      log_embs.append(emb_map[log])
    # print(len(log_embs))
    log_embs = np.array(log_embs)
    # print(log_embs.shape)
    self.uncertain = {}
    
    # else:
    
    if init and self.token_dist == 'prob':
      hard_indices = unselected_indices
      easy_indices = []
      hard_logs = {}
      easy_logs = {}
    else:
      if init:
        self.lookUpMap = self.generateMap(self.map_path)
      hard_indices = []
      easy_indices = []
      hard_logs = {}
      easy_logs = {}
      dist_score = {}
      prob_score = {}
      dist_hard_idx = []
      dist_easy_idx = []
      for line_idx in tqdm(unselected_indices):
        prompt = ''
        
      
        prompt = ''
        prompt, similarist_gt,keywords,var = self.generatePrompt(log_list[line_idx], nearest_num=5)
        instruction = "For each log after <prompt> tag, extract one log template according to keywords\
  (construct the template based on keywords, identify the remaining tokens are keywords or not, reserve the keywords and substitute non keywords as <*>)\
  and put the template after <extraction> tag and between <START> and <END> tags."
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt + "<prompt>:" + log_list[line_idx].strip() + "\n<keywords>:" + ','.join(keywords)+ "\n<variables>:" + ','.join(var)+"\n<extraction>: "},
        ]
        
        model_inputs = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
        input_length = model_inputs.shape[1]
        # attention_mask = torch.ones
        outputs = self.confidentLLM.generate(model_inputs,return_dict_in_generate=True, output_scores=True, max_length=input_length+20,pad_token_id=self.tokenizer.eos_token_id)
        gen_sequences = outputs.sequences[:, input_length:]
        probs = torch.stack(outputs.scores, dim=1).softmax(-1)
        gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)
        
        self.uncertain[line_idx] = float(gen_probs[:, 0][0])
        
      
        response = self.pipeline.chat.completions.create(
              model=self.model_name,  
              messages=[{"role": "system", "content": "You need to extract template from logs."},
                          {"role": "user", "content":instruction+'\n'+prompt + "<prompt>:" +  log_list[line_idx].strip() + "\n<keywords>:" + ','.join(keywords)+ "\n<variables>:" + ','.join(var)+"\n<extraction>: "}]
          )
        result = self.extractResultTemplate(response.choices[0].message.content)
        token_log = log_list[line_idx].split(' ')
        token_pred = result.replace('<*>','').split(' ')
        # print(token_pred)
        for item in token_pred:
            f = 0
            for log_item in token_log:
              if item in log_item:
                f = 1
                break
            if f == 0:
              dist_hard_idx.append(line_idx)
              dist_score[line_idx] = 1

              break
        if line_idx not in dist_hard_idx:
          dist_easy_idx.append(line_idx)
          dist_score[line_idx] = 0
        
    
      
    
    temp = sorted(self.uncertain,key=self.uncertain.get)
    for idx,indice in enumerate(temp):
      self.uncertain[indice] = (len(temp) - idx) / len(temp)
    for k in self.uncertain:
      if self.dataset == 'Mac':
        self.uncertain[k] = 0.7*self.uncertain[k] + 0.3 *dist_score[k]
      else:
        self.uncertain[k] = 0.3*self.uncertain[k] + 0.7 *dist_score[k]
    weights = self.uncertain
  # else:
   
    if hard_indices == [] and self.selection == 'max_cover':
      print('no instances')
      self.lookUpMap = self.generateMap(self.map_path)
      return 0
    print(self.reselect_num)
    print(f'{len(set(hard_indices))} {len(set(hard_indices))}')
    if self.selection == 'max_cover':
      selected_indices_new = self.density_max_coverage(embeddings=log_embs, 
                                                            hard_idx = hard_indices, 
                                                            easy_idx = easy_indices,
                                                            selected_indices=self.selected_indices, 
                                                            select_num= self.reselect_num,
                                                            k=5,
                                                            vote_file=None,
                                                            weighted = False,
                                                            two_hop=False,
                                                            thres_graph=False,
                                                            mc_selection='hard')
    
      if len(self.selected_indices) + len(selected_indices_new) == self.budget and iter_num < self.round:
        selected_indices_new = selected_indices_new[:int((self.budget-2000*self.cand_ratio)//self.round)]
    else:
      if self.budget_strategy == 'avg':
        next_round_num = int((self.budget-2000*self.cand_ratio)//self.round)
      elif self.budget_strategy == 'inc':
        next_round_num = int((self.budget-2000*self.cand_ratio)//self.round+(iter_num-1)*10)
        print(f'next round: {next_round_num}')
      elif self.budget_strategy == 'dec':
        next_round_num = int((self.budget-2000*self.cand_ratio)//self.round-(iter_num-1)*10)
        print(f'next round: {next_round_num}')
      selected_indices_new = self.max_selection(weights,keywords,threshold=self.threshold,log_list=log_list,lamb=self.lamb,
                                                            selected_indices=self.selected_indices, 
                                                            select_num= next_round_num
                                                            )
      
    temp_files = []
    new_cand_log = []
    new_cand_gt = []
    if self.selection != 'max_cover':
      keywords = set(keywords)
    for index_new in selected_indices_new:
      new_cand_log.append(log_list[index_new])
      new_cand_gt.append(groundtruth_template[index_new])
      temp_files.append({'id':index_new,'log':log_list[index_new],'template':groundtruth_template[index_new]})
      if self.selection != 'max_cover':
        
        words = set(processLog(self.word_file,log_list[index_new].strip()))
        keywords.update(words)
    
    self.gt_cand = self.gt_cand + new_cand_gt
    self.log_cand = self.log_cand + new_cand_log

    self.log_test = log_list
    self.gt_test = groundtruth_template

    self.selected_indices  = self.selected_indices + selected_indices_new
    self.reselect_num = self.reselect_num - len(selected_indices_new)
    
    print(f'log_test: {len(self.log_test)}')
    print(f'log_cand: {len(self.log_cand)}')
    print(f'gt_test: {len(self.gt_test)}')
    if self.selection == 'max_cover':
      if len(self.selected_indices) == self.budget:
        return len(hard_indices)
      else:
        self.lookUpMap = self.generateMap(self.map_path)
      return len(hard_indices)
    else:
      
      self.lookUpMap = self.generateMap(self.map_path)
      return cover,keywords

  def density_max_coverage(self,embeddings,hard_idx, easy_idx, selected_indices,select_num,k,vote_file=None, weighted=False, two_hop = True, thres_graph=False, mc_selection="hard"):
    """
    MaxCover porblem formulation and solution.
    The annotation method for AdaICL is from https://github.com/amazon-science/adaptive-in-context-learning 
    Args:
        embeddings 
        hard_idx: indices the model is uncertain about
        easy_idx: indices the model is confident about
        selected_indices: already annotated indices
        select_num: new budget
        k: graph hyperparameter for k-NN
        vote_file (optional): for saving results. Defaults to None.
        weighted (bool, optional): AdaICL or AdaICL+. Defaults to False.
        two_hop (bool, optional): one-hop or two-hop graph. Defaults to True.
        thres_graph (bool, optional): kNN or threshold graph. Defaults to False.
        mc_selection (str, optional): selecting hard (vs. easy vs. both) examples. Defaults to "hard".

    Returns:
        list: New annotated data
    """
    
    if mc_selection=="hard":
        selected = easy_idx.copy() + selected_indices.copy()
    elif mc_selection=="hard_easy":
        selected = selected_indices.copy()
    elif mc_selection=="easy":
        selected = hard_idx.copy() + selected_indices.copy()
    #selected_indices = easy_idx.copy() + selected_indices.copy()
    n = len(embeddings)
    # print("2hop graph: ", two_hop)
    
    bar = tqdm(range(n),desc=f'voting')
    vote_stat = defaultdict(list)
    if not thres_graph:
        for i in range(n):
            if i in selected_indices:
              continue
            cur_emb = embeddings[i].reshape(1, -1)
            cur_scores = np.sum(cosine_similarity(embeddings, cur_emb), axis=1)
            sorted_indices = np.argsort(cur_scores).tolist()
            cnt = 0
            for idx in sorted_indices[::-1]:
                if idx!=i and idx not in selected_indices:
                    vote_stat[idx].append(i)
                    cnt += 1
                if cnt >= k:
                  break
            bar.update(1)
        
    else:
        print("Threshold graph")
        thresholds = []
        for i in range(n):
            cur_emb = embeddings[i].reshape(1, -1)
            cur_scores = np.sum(cosine_similarity(embeddings, cur_emb), axis=1)
            thres_idx = np.argsort(cur_scores).tolist()[-k-1]
            thresholds.append(cur_scores[thres_idx])
        thresholds.sort()
        mean_thres = statistics.median(thresholds) #sum(thresholds) / len(thresholds)

        for i in range(n):
            cur_emb = embeddings[i].reshape(1, -1)
            cur_scores = np.sum(cosine_similarity(embeddings, cur_emb), axis=1)
            sorted_indices = np.argsort(cur_scores).tolist()
            for idx in sorted_indices:
                if idx!=i and cur_scores[idx] > mean_thres: # and idx in hard_idx:
                    vote_stat[idx].append(i)
            bar.update(1)

    if vote_file is not None:
      # if os.exist(vote_file)
      with open(vote_file,'w') as f:
          json.dump(vote_stat,f)

    votes = sorted(vote_stat.items(),key=lambda x:len(x[1]),reverse=True)
    new_selected_indices = []
    
    selected_times = defaultdict(int)
    egonet = defaultdict(list)

    #Create egonets
    for idx,candidates in votes:
        for idx_support in candidates:
            if (idx_support in hard_idx) and (idx_support not in egonet[idx]):
                egonet[idx].append(idx_support)
                selected_times[idx] += 1
                if two_hop:
                    neigh_2hop = vote_stat[idx_support]
                    for node in neigh_2hop:
                        if (node in hard_idx) and (node != idx) and (node not in egonet[idx]):
                            egonet[idx].append(node)
                            selected_times[idx] += 1

    

    # print("Distribution of Sets: ", selected_times)
    print("Weighted sum:", weighted)

    egonet_greedy = sorted(egonet.items(),key=lambda x:len(x[1]),reverse=True)

    selected_weight = defaultdict(int)

    #print("Egonets:", egonet_greedy)
    while len(new_selected_indices)<select_num:
        cur_scores = defaultdict(int)
        for idx,candidates in egonet_greedy:
            if idx in selected+new_selected_indices:
                cur_scores[idx] = -100 #sanity check
                continue
            for idx_support in candidates:
                if idx_support in hard_idx: #sanity check
                    if weighted:
                        cur_scores[idx] += 10 ** (-selected_weight[idx_support])
                    else:
                        cur_scores[idx] += 1

        cur_selected_idx = max(cur_scores.items(),key=lambda x:x[1])[0]
        new_selected_indices.append(int(cur_selected_idx))

        for idx_support in egonet[cur_selected_idx]:
            selected_weight[idx_support] += 1
            if (not weighted) and (idx_support in hard_idx):
                hard_idx.remove(idx_support)
                
            
        if len(hard_idx) == 0: #only true for weighted=False
            print("All hard examples covered, annotation size:", len(new_selected_indices) )
            break

    return list(set(new_selected_indices))

  def max_selection(self,weights,keywords,threshold,log_list,lamb, selected_indices,select_num):
   
    
    bar = tqdm(range(len(log_list)),desc=f'voting')
    vote_stat = defaultdict(set)
    
    for i in range(len(log_list)):
        if i in selected_indices:
          continue
        log = log_list[i]
        p_log = [[word,idx] for idx, word in enumerate(processLog(self.word_file,log.strip(),keywords=keywords))]
        dist_dic = {}
        
        for j,cand_log in enumerate(log_list):
          if j in selected_indices or j == i:
            continue
          p_cand_log = [[word,idx] for idx, word in enumerate(processLog(self.word_file,cand_log.strip(),keywords=keywords))]
          
          dist = 0.5*(edit_distance_on_real_sequences(p_log, p_cand_log, 0.8,self.sim_dic)+edit_distance_on_real_sequences( p_cand_log, p_log,0.8,self.sim_dic))
          
          if dist <= threshold:
            dist_dic[j] = dist
            vote_stat[i].add(j)
          else:
            if i not in vote_stat:
              vote_stat[i] = set()
        
        bar.update(1)

    new_selected_indices = []
    
    covered = set()
    while len(new_selected_indices)<select_num:
        best_weight = 0
        best_log = -1
        best_activated = set()
        for k in vote_stat:
          if k not in list(new_selected_indices)+selected_indices:
            activated = vote_stat[k].union({k}) - covered
            weight = lamb*weights[k] + (1-lamb)*len(activated)/2000
            if weight > best_weight:
              best_weight = weight
              best_log = k
              best_activated = activated
        

        if best_log == -1:
          break
                
        
        covered.update(best_activated)
        new_selected_indices.append(best_log)

    return list(set(new_selected_indices))

  def max_cover(self,embeddings,hard_idx, easy_idx, selected_indices,select_num,k,cover,vote_file=None, weighted=False, two_hop = True, thres_graph=False, mc_selection="hard"):
    """
    MaxCover porblem formulation and solution.

    Args:
        embeddings 
        hard_idx: indices the model is uncertain about
        easy_idx: indices the model is confident about
        selected_indices: already annotated indices
        select_num: new budget
        k: graph hyperparameter for k-NN
        vote_file (optional): for saving results. Defaults to None.
        weighted (bool, optional): AdaICL or AdaICL+. Defaults to False.
        two_hop (bool, optional): one-hop or two-hop graph. Defaults to True.
        thres_graph (bool, optional): kNN or threshold graph. Defaults to False.
        mc_selection (str, optional): selecting hard (vs. easy vs. both) examples. Defaults to "hard".

    Returns:
        list: New annotated data
    """
    
    if mc_selection=="hard":
        selected = easy_idx.copy() + selected_indices.copy()
    elif mc_selection=="hard_easy":
        selected = selected_indices.copy()
    elif mc_selection=="easy":
        selected = hard_idx.copy() + selected_indices.copy()
    #selected_indices = easy_idx.copy() + selected_indices.copy()
    n = len(embeddings)
    # print("2hop graph: ", two_hop)
    
    bar = tqdm(range(n),desc=f'voting')
    vote_stat = defaultdict(list)
    if not thres_graph:
        for i in range(n):
            if i in selected_indices:
              continue
            cur_emb = embeddings[i].reshape(1, -1)
            cur_scores = np.sum(cosine_similarity(embeddings, cur_emb), axis=1)
            sorted_indices = np.argsort(cur_scores).tolist()
            cnt = 0
            for idx in sorted_indices[::-1]:
                if idx!=i:# and idx not in selected_indices:
                    vote_stat[i].append(idx)
                    cnt += 1
                if cnt >= k:
                  break
            bar.update(1)
        
    else:
        print("Threshold graph")
        thresholds = []
        for i in range(n):
            cur_emb = embeddings[i].reshape(1, -1)
            cur_scores = np.sum(cosine_similarity(embeddings, cur_emb), axis=1)
            thres_idx = np.argsort(cur_scores).tolist()[-k-1]
            thresholds.append(cur_scores[thres_idx])
        thresholds.sort()
        mean_thres = statistics.median(thresholds) #sum(thresholds) / len(thresholds)

        for i in range(n):
            cur_emb = embeddings[i].reshape(1, -1)
            cur_scores = np.sum(cosine_similarity(embeddings, cur_emb), axis=1)
            sorted_indices = np.argsort(cur_scores).tolist()
            for idx in sorted_indices:
                if idx!=i and cur_scores[idx] > mean_thres: # and idx in hard_idx:
                    vote_stat[idx].append(i)
            bar.update(1)

    if vote_file is not None:
      # if os.exist(vote_file)
      with open(vote_file,'w') as f:
          json.dump(vote_stat,f)

    votes = sorted(vote_stat.items(),key=lambda x:len(x[1]),reverse=True)
    new_selected_indices = []
    
    selected_times = defaultdict(int)
    egonet = defaultdict(set)

    #Create egonets
    for idx,candidates in votes:
        for idx_support in candidates:
            if (idx_support in hard_idx) and (idx_support not in egonet[idx]):
                egonet[idx].add(idx_support)
                selected_times[idx] += 1
                

    

    # print("Distribution of Sets: ", selected_times)
    print("Weighted sum:", weighted)

    egonet_greedy = sorted(egonet.items(),key=lambda x:len(x[1]),reverse=True)

    selected_weight = defaultdict(int)
    covered = set()
    universe = set(hard_idx)
    new_selected = set()
    # cover = []
    #print("Egonets:", egonet_greedy)
    while covered != universe:
        best_subset = -1
        best_covered = set()
        for idx,candidates in egonet_greedy:
          if idx not in list(new_selected)+selected:
            covered_elements = candidates - covered
            if len(covered_elements) > len(best_covered):
                best_covered = covered_elements
                best_subset = idx

        if best_subset == -1:
          break
                
        for item in vote_stat[best_subset]:
          cover.add(item)
        cover.update(best_covered)
        cover.add(best_subset)
        covered.update(best_covered)
        covered.add(best_subset)

        # 从子集列表中移除已选择的子集
        new_selected.add(best_subset)

    return list(new_selected), cover

  def prompt_set_cover(self,log):
    universe = set()
    temp_log = log
    new_selected_indices = []

    candidate_word_list = {}
    candidate_var_list = {}
    var_set = []
    key_set = []
    log_var = set()
    for idx in range(len(self.gt_cand)):
      
      varwords,var_indices = find_replaced_characters(self.log_cand[idx].strip(), self.gt_cand[idx].strip())
      
      words = processLog(self.word_file,self.log_cand[idx].strip())
      key_set.extend(words)
      candidate_word_list[idx] = set(words)

    key_set = set(key_set)
    
    prev = False
    log_words = processLog(self.word_file,log.strip())
    log_keywords= processLog(self.word_file,log.strip(),mode='basic')
    for w in log_keywords:
      if w not in log_words:
        log_var.add(w)
      
    
    for k in key_set:
      if k in log_words:
        universe.add(k)
    p_log = temp_log
    p_log = [[word,idx] for idx, word in enumerate(processLog(self.word_file,log.strip(),self.gts))]
    
    
    covered = set()
    covered_variable = set()
    new_selected = set()
    sort_log = {}
    while covered != universe:
        best_subset = -1
        best_covered = set()
        best_keyword = set()
        best_var = set()
        for idx in candidate_word_list:
          if idx not in list(new_selected):
            words = candidate_word_list[idx]
            
            covered_elements = words.intersection((universe))
            if len(covered_elements) > len(best_covered):
                best_covered = covered_elements
                best_keyword = covered_elements
                best_subset = idx

        if best_subset == -1:
          break
                
        if len(new_selected) == 7:
          break
        covered.update(best_keyword)
        covered_variable.update(best_var)
        new_selected.add(best_subset)
    
    for item in new_selected:
      
      p_cand_log = [[word,idx] for idx, word in enumerate(processLog(self.word_file,self.log_cand[item].strip()))]
      
      dist = 0.5*(edit_distance_on_real_sequences(p_log, p_cand_log, 0.8,self.sim_dic)+edit_distance_on_real_sequences( p_cand_log, p_log,0.8,self.sim_dic))
      
      sort_log[item] = dist
    temp = []
    sort_log = sorted(sort_log.items(),key=lambda x:x[1])
    for rank,(idx,dist) in enumerate(sort_log):
      if rank > 3 and dist > self.threshold:
        continue
      temp.append((idx,dist))
    
    return list(new_selected), temp,covered,log_var,prev 
 