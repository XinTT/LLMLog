import os
# import openai
import argparse
import pandas as pd
from LLMLog import ModelParser
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
                token_dist = args.confidence,
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
                budget_strategy= args.budget_strategy
                )
    
    parser.BatchParse(model = args.model, 
                        model_name = args.model_name, 
                        limit = args.limit,         # number of logs for testing
                        N = args.N,                  # number of examples in the prompt
                        )
    
    # parser.evaluation()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', type=str, default="xxxxxxxxxxxxxxxxxxxxxxx", help='openai key')
    parser.add_argument('--log_path', type=str, default='../../data/loghub_2k', help='log path')
    parser.add_argument('--result_path', type=str, default='results', help='result path')
    parser.add_argument('--map_path', type=str, default='maps', help='map path')
    parser.add_argument('--dataset', type=str, default='Hadoop', help='dataset name')
    parser.add_argument('--emb_path', type=str, default='embeddings', help='embedding path')
    parser.add_argument('--cand_ratio', type=float, default=0.05, help='ratio of candidate set')
    # parser.add_argument('--label_ratio', type=float, default=0.05, help='ratio of candidate set')
    parser.add_argument('--split_method', type=str, default='DPP', help='random or DPP')
    parser.add_argument('--permutation', type=str, default='ascend', help='ascend, descend, or random')
    parser.add_argument('--model', type=str, default='gpt', help='model name')
    parser.add_argument('--model_name', type=str, default='gptC', help='model name')
    parser.add_argument('--limit', type=int, default=2000, help='number of logs for testing')
    parser.add_argument('--N', type=int, default=5, help='number of examples in the prompt')
    parser.add_argument('--budget', type=int, default=200, help='budget number')
    parser.add_argument('--max_round', type=int, default=2, help='round')
    parser.add_argument('--selection', type=str, default='max_cover', help='round')
    parser.add_argument('--lamb', type=float, default=0.3, help='round')
    parser.add_argument('--threshold', type=float, default=0.29, help='round')
    parser.add_argument('--prompt', type=str, default='set_cover', help='round')
    parser.add_argument('--budget_strategy', type=str, default='avg', help='determine budget')
    args = parser.parse_args()
    main(args)
    # evaluation(args.log_path,args.dataset,args.result_path)