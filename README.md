# LLMLog
The official code repository of the paper "LLMLog: Advanced Log Template Generation via LLM-driven Multi-Round Annotation"

Install all packages via:

```
pip install -r requirements.txt
```
The log-pai dataset can be found [here](https://github.com/logpai/logparser)

The word embedding and log embedding can be generated via:
```
  python embedding.py
```

Then run experiments via: 

```
  python demo.py --dataset Mac --permutation descend --selection max_influence --prompt adaptive --max_round 2 --lambda 0.5 --threshold 0.3 --budget_strategy avg --key xxxxxxxxxxxx
```

--permutation: similarity permuation for demontration logs to input log, available: ascend, descend, random

--dataset: The dataset experimented on;

--selection: Annotation criteria, max_influence: for proposed criteria, max_cover: for DivLog+AdaICL

--prompt: adaptive or fixed

--max_round: Annotation round

--lambda: weight to control LLM prediction confidence and representativeness in selection;

--threshold: similarity threshold for representativeness;

--budget_strategy: strategy to control budget in different annotation rounds, available: inc, dec, and avg;

--key: api_key for GPT.
