import enum
import os
import pathlib
import json
from tqdm import tqdm
from konlpy.tag import Mecab
from collections import defaultdict

mecab = Mecab()

def main():
    json_dir = os.path.join('/opt/ml/code/outputs/test_dataset/koelectra-new', 'nbest_predictions.json')
    print(json_dir)
    json_data = open_json_file(json_dir)
    json_dump_data = defaultdict(set)
    topk = 3

    temp = defaultdict(dict)
    count = 0
    for idx, element in tqdm(enumerate(json_data)):
        if count == topk:
            # print(temp)
            json_dump_data[title.split('/')[0]] = post_processing(temp)
            temp.clear()
            count = 0

        title = element
        for data in json_data[element]:
           temp[data['text']] = data['probability']
        count+=1

    json_write_dir = os.path.join('/opt/ml/code/outputs/test_dataset/koelectra-new', 'predictions_pp.json')
    write_json_file(json_write_dir, json_dump_data)
    print("===== Success =====")


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


def post_processing(dict):
    dict = sorted(dict.items(), reverse=True, key=lambda item: item[1])

    for k, v in dict:
        t = tokenize(k)
        # print(k, t)
        # exit()
        # flag = False
        if len(t) == 1 and t[0][1] == 'MAG':
            continue
        # for t_element in t:
        #     # if '+' in t_element[1]:
        #     #     t_element[1].split('+')
        #     # else:
        #     if t_element[1] in candidate:
        #         flag = True
        #         break

        # if not flag:
        return k


if __name__ == "__main__":
    # candidate =  ['NNG', 'NNP', 'SN', 'NNBC','SY','JKG','XSN','SL','SSC','SSO','JKB','EC','VV']
    candidate =  ['MAG']
    # candidate = ['VX', 'NP', 'NR', 'XPN', 'SY', 'XSA', 'XR', 'VCP', 'JKG', 'UNKNOWN', 'NNG', 'SSC', 'JKS', 'ETN', 'SC', 'EC', 'EP', 'SF', 'XSN', 'SL', 'XSV', 'JKQ', 'JX', 'ETM', 'VV', 'JC', 'MAG', 'SH', 'NNBC', 'JKB', 'JKO', 'VA', 'MM', 'SSO', 'NNB', 'JKV', 'NNP', 'MAJ', 'IC', 'SN', 'EF']

    # print(tokenize("이따금"))
    main()
