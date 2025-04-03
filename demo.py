import os
# import openai
import argparse
import pandas as pd
from DivLog_api import ModelParser, evaluatePA, evaluatePTA, evaluateRTA
import time,json

def main(args):
    # get a tester object with data
    # openai.api_key = args.key
    print("Parsing " + args.dataset + " ...")
    # print(args.single)
    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

    if not os.path.exists(args.map_path):
        os.mkdir(args.map_path)

    if not os.path.exists(args.log_path):
        print("Log path does not exist. Please check the path.")
        exit()

    if not os.path.exists(args.emb_path):
        print("Embedding path does not exist. Please check the path.")
        exit()

    parser = ModelParser(
                log_path = args.log_path,        # .log_structured_csv
                result_path=args.result_path,    # .result_csv
                map_path=args.map_path,          # .map_json
                dataset = args.dataset,             # 16 datasets
                emb_path = args.emb_path,           # embedding
                cand_ratio = args.cand_ratio,       # ratio of candidate set
                split_method = args.split_method,   # random or DPP
                order_method = args.order_method,   # random or KNN
                permutation = args.permutation,     # permutation
                warmup = args.warmup,               # warmup or not
                subname = args.subname,  
                evaluate = args.evaluate,           # evaluate or not
                single_annotation = args.single,
                hard_ratio= args.hard_ratio,
                confidence = args.confidence,
                budget = args.budget,
                round = args.max_round,
                self_check=args.self_check,
                selection = args.selection,
                lamb=args.lamb,
                threshold=args.threshold,
                prompt=args.prompt,
                dist_metric=args.dist,
                model= args.model,
                key = args.key,
                budget_strategy= args.budget_strategy,
                ke = args.ke,
                cos_sim=args.cos_sim
                )
    time1 = time.time()
    parser.BatchParse(
                        cost=args.cost_test,
                        limit = args.limit,         # number of logs for testing
                        N = args.N,
                                                            # number of examples in the prompt
                        )
    time2 = time.time()
    t = {'time':time2-time1}
    with open('llmlog_gpt_time_'+args.dataset+'_'+args.selection+'_'+args.prompt+'_'+args.budget_strategy+'.json','w') as f:
        json.dump(t,f,indent=2)
    # parser.evaluation()

def evaluation(log_path,dataset,result_path):
    log_path = log_path + "/{}/{}_2k.log_structured.csv".format(dataset,dataset)
    docs = os.listdir(result_path)
   
    docs = os.listdir(result_path)
    for doc in docs:
     
        df_groundtruth = pd.read_csv(log_path)
        print(doc)
        df_parsedlog = pd.read_csv(result_path+'/'+doc)
        PA = evaluatePA(df_groundtruth, df_parsedlog)
        PTA = evaluatePTA(df_groundtruth, df_parsedlog)
        RTA = evaluateRTA(df_groundtruth, df_parsedlog)
        print("{}:\t PA:\t{:.6f}\tPTA:\t{:.6f}\tRTA:\t{:.6f}\tGA:\t{:.6f}".format(dataset, PA, PTA, RTA))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', type=str, default="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", help='openai key')
    parser.add_argument('--log_path', type=str, default='../../data/loghub_2k', help='log path')
    parser.add_argument('--result_path', type=str, default='webank_results', help='result path')
    parser.add_argument('--map_path', type=str, default='webank_maps', help='map path')
    parser.add_argument('--dataset', type=str, default='Hadoop', help='dataset name')
    parser.add_argument('--emb_path', type=str, default='webank_embeddings', help='embedding path')
    parser.add_argument('--cand_ratio', type=float, default=0, help='ratio of candidate set')
    # parser.add_argument('--label_ratio', type=float, default=0.05, help='ratio of candidate set')
    parser.add_argument('--split_method', type=str, default='DPP', help='random or DPP')
    parser.add_argument('--order_method', type=str, default='KNN', help='random or KNN')
    parser.add_argument('--permutation', type=str, default='ascend', help='ascend, descend, or random')
    parser.add_argument('--warmup', type=bool, default=False, help='warmup or not')
    parser.add_argument('--model', type=str, default='gpt-4o', help='model name')
    parser.add_argument('--model_name', type=str, default='gptC', help='model name')
    parser.add_argument('--limit', type=int, default=2000, help='number of logs for testing')
    parser.add_argument('--N', type=int, default=5, help='number of examples in the prompt')
    parser.add_argument('--subname', type=str, default='', help='subname of the files')
    parser.add_argument('--evaluate', type=bool, default=False, help='evaluate or not')
    parser.add_argument('--single', type=str, default='yes', help='single annotation or not')
    parser.add_argument('--sim', type=str, default='cos', help='cos or idf or drain')
    parser.add_argument('--hard_ratio', type=float, default=0.8, help='ratio of hard samples')
    parser.add_argument('--confidence', type=str, default='prob', help='prob or dist')
    parser.add_argument('--budget', type=int, default=200, help='budget number')
    parser.add_argument('--max_round', type=int, default=2, help='round')
    parser.add_argument('--self_check', type=str, default='', help='round')
    parser.add_argument('--selection', type=str, default='max_cover', help='round')
    parser.add_argument('--lamb', type=float, default=0.3, help='round')
    parser.add_argument('--threshold', type=float, default=0.29, help='round')
    parser.add_argument('--prompt', type=str, default='set_cover', help='round')
    parser.add_argument('--dist', type=str, default='edr', help='round')
    parser.add_argument('--budget_strategy', type=str, default='avg', help='determine budget')
    parser.add_argument('--cost_test', type=str, default='no', help='yes or no')
    parser.add_argument('--ke', type=str, default='no', help='yes or no')
    parser.add_argument('--cos_sim', type=float, default=0.8, help='threshold of similar word')
    args = parser.parse_args()
    main(args)
    # evaluation(args.log_path,args.dataset,args.result_path)