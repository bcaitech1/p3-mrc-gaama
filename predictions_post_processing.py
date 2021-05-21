import enum
import os
import pathlib
import json
from tqdm import tqdm
from konlpy.tag import Mecab
from collections import defaultdict
from pororo import Pororo

ner = Pororo(task="ner", lang="ko")
mecab = Mecab()

def tokenize(text):
    # return text.split(" ")
    return mecab.pos(text)

def open_json_file(filename):
    with open(filename, encoding='UTF8') as file:
        try:
            return json.load(file)
        except ValueError as e:
            print('Parsing failed! Error: {}'.format(e))
            return None


def write_json_file(filename, data):
    with open(filename, "w") as writer:
        try:
            writer.write(json.dumps(data, indent=4, ensure_ascii=False) + "\n")
        except ValueError as e:
            print('Writing failed! Error: {}'.format(e))
            return None


def post_processing(answer):
    answer_split = answer.split(' ')
    processed_answer = []
    for i, element in enumerate(answer_split):
        tokenized_answer = tokenize(element)
        temp = ""
        for idx, pos in enumerate(tokenized_answer):
            if not pos[1].startswith('J'):
                temp += pos[0]
            else:
                processed_answer.append(temp)
                return ' '.join(processed_answer)

        processed_answer.append(temp)

    return ' '.join(processed_answer)


def post_processing_merge(answer):

    
    answer_split = answer.split(' ')
    processed_answer = []
    for i, element in enumerate(answer_split):
        tokenized_answer = tokenize(element)
        temp = ""
        for idx, pos in enumerate(tokenized_answer):
            if not pos[1].startswith('J'):
                temp += pos[0]
            else:
                processed_answer.append(temp)
                return ' '.join(processed_answer)

        processed_answer.append(temp)

    return ' '.join(processed_answer)


if __name__ == "__main__":
    # cwd = pathlib.Path.cwd()
    json_dir = os.path.join('/opt/ml/code/outputs/train_dataset/koelectra-korquad', 'nbest_predictions.json')
    print(json_dir)
    json_data = open_json_file(json_dir)
    # print(json_data)
    # exit()
    json_dump_data = defaultdict(set)

    temp = defaultdict(set)
    for idx, element in tqdm(enumerate(json_data)):
        answer = json_data[element]
        pp_answer = post_processing(answer)
        if answer != pp_answer:
            print(answer, " / " , pp_answer)

        print(post_processing(answer))
        json_dump_data[element] = post_processing(answer)
        print(idx, element, answer)
        exit()

    json_write_dir = os.path.join('/opt/ml/code/outputs/test_dataset/koelectra-new', 'predictions_pp.json')
    write_json_file(json_write_dir, json_dump_data)
    print("===== Success =====")
