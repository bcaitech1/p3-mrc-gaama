"""
Open-Domain Question Answering 을 수행하는 inference 코드 입니다.

대부분의 로직은 train.py 와 비슷하나 retrieval, predict
"""
from pororo import Pororo
from konlpy.tag import Mecab
import argparse
import logging
import os
import sys
from datasets import load_metric, load_from_disk, Sequence, Value, Features, Dataset, DatasetDict
from subprocess import Popen, PIPE, STDOUT
from elasticsearch import Elasticsearch
from transformers import AutoConfig, AutoModelForQuestionAnswering, AutoTokenizer
import time
from collections import defaultdict
from tqdm import tqdm
import json
import numpy as np

from transformers import (
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)

from utils_qa import postprocess_qa_predictions, check_no_error, tokenize
from trainer_qa import QuestionAnsweringTrainer
from retrieval import SparseRetrieval

from arguments import (
    ModelArguments,
    DataTrainingArguments,
)

logger = logging.getLogger(__name__)


def main():
    # 가능한 arguments 들은 ./arguments.py 나 transformer package 안의 src/transformers/training_args.py 에서 확인 가능합니다.
    # --help flag 를 실행시켜서 확인할 수 도 있습니다.
    # mrc = Pororo(task="mrc", lang="ko")

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.do_train = True

    print(f"model is from {model_args.model_name_or_path}")
    print(f"data is from {data_args.dataset_name}")

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed before initializing model.
    set_seed(training_args.seed)

    datasets = load_from_disk(data_args.dataset_name)
    print(datasets)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name
        if model_args.config_name
        else model_args.model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name
        if model_args.tokenizer_name
        else model_args.model_name_or_path,
        use_fast=True,
    )
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
    )

    # run passage retrieval if true
    if data_args.eval_retrieval:
        datasets = run_sparse_retrieval(datasets, training_args, data_args)
        print("============check==============")

    # eval or predict mrc model
    if training_args.do_eval or training_args.do_predict:
        run_mrc(data_args, training_args, model_args, datasets, tokenizer, model)


def run_sparse_retrieval(datasets, training_args, data_args):
    #### retreival process ####
    retriever = SparseRetrieval(tokenize_fn=tokenize,
                                data_path="/opt/ml/input/data",
                                context_path="wikipedia_documents.json",
                                args = data_args
                                )

    if data_args.embedding_mode != 'bm25_new' and data_args.embedding_mode != 'elastic':
        retriever.get_sparse_embedding()

    if data_args.embedding_mode == 'elastic':
        es = setting_elastic()
        df = retriever.retrieve_elastic(datasets['validation'], topk=data_args.topk, what='val', es = es)
    else:
        df = retriever.retrieve(datasets['validation'], topk=data_args.topk, what='val')

    if training_args.do_predict: # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
        f = Features({'context': Value(dtype='string', id=None),
                      'score': Value(dtype='string', id=None),
                      'id': Value(dtype='string', id=None),
                      'question': Value(dtype='string', id=None)})

    elif training_args.do_eval: # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
        f = Features({'answers': Sequence(feature={'text': Value(dtype='string', id=None),
                                                   'answer_start': Value(dtype='int32', id=None)},
                                          length=-1, id=None),
                      'context': Value(dtype='string', id=None),
                      'id': Value(dtype='string', id=None),
                      'question': Value(dtype='string', id=None)})

    datasets = DatasetDict({'validation': Dataset.from_pandas(df, features=f)})
    print(datasets)
    return datasets

def write_json_file(filename, data):
    with open(filename, "w+") as writer:
        try:
            writer.write(json.dumps(data, indent=4, ensure_ascii=False) + "\n")
        except ValueError as e:
            print('Writing failed! Error: {}'.format(e))
            return None

def run_mrc(data_args, training_args, model_args, datasets, tokenizer, model):
    # only for eval or predict
    column_names = datasets["validation"].column_names
    
    question_column_name = "question" if "question" in column_names else column_names[0]
    context_column_name = "context" if "context" in column_names else column_names[1]
    answer_column_name = "answers" if "answers" in column_names else column_names[2]
    
    json_dump_data = defaultdict(set)

    mrc = Pororo(task="mrc", lang="ko")
    for idx, element in tqdm(enumerate(datasets['validation'])):
        contexts = element['context'].split('[SEP]')
        scores_softmax = element['score'].split('[SEP]')
        answers = []

        # topk passage 검사
        assert len(contexts) == data_args.topk
        
        for i, context in enumerate(contexts):
            # print(element['question'], context)
            answer = mrc(element['question'], context)
            for ii in range(len(answer)):
                answer[ii][2] = float(answer[ii][2]) * float(scores_softmax[i])
            answers.append(answer)
            
        # answers = mrc(element['question'], element['context'])
        answers = [y for x in answers for y in x]

        # exit()
        # sorting
        
        answers.sort(key=lambda t: t[2], reverse=True)
        json_dump_data[element['id']] = answers[0][0]

    json_write_dir = os.path.join('/opt/ml/code/outputs/test_dataset/koelectra-pororo-top10', 'predictions_pp.json')
    write_json_file(json_write_dir, json_dump_data)

def setting_elastic():
    es_server = Popen(['/opt/ml/elasticsearch-7.9.2/bin/elasticsearch'],
                    stdout=PIPE, stderr=STDOUT,
                    preexec_fn=lambda: os.setuid(1)
                    )
                    
    # time.sleep(30) 

    es = Elasticsearch('localhost:9200')

    return es

if __name__ == "__main__":
    main()
