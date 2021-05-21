import faiss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from tqdm.auto import tqdm
import pandas as pd
import scipy
import pickle
import json
import os
import numpy as np
from collections import Counter
import re

from datasets import (
    Dataset,
    load_from_disk,
    concatenate_datasets,
)
from konlpy.tag import Mecab

import numpy as np
from tqdm import tqdm, trange
import argparse
import random
import torch
import os
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel, AdamW, TrainingArguments, get_linear_schedule_with_warmup, AutoTokenizer
import pickle
from rank_bm25 import BM25Okapi, BM25Plus, BM25L, BM25
import time
from contextlib import contextmanager

import math
from multiprocessing import Pool, cpu_count

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] done in {time.time() - t0:.3f} s')
ban_words=("이따금","아마","절대로","무조건","한때","대략","오직",
          "오로지","감히","최소","아예","반드시","꼭","때때로","이미"
          ,"종종","졸곧","약간","기꺼이", "비록","꾸준히","일부러","어쩔", "문득", "어쨌든", "순전히", "필수")

mecab = Mecab()

class ES(BM25):
    def __init__(self, corpus, tokenizer=None, k1=1.2, b=0.75, delta=0):
        # Algorithm specific parameters
        self.k1 = k1
        self.b = b
        self.delta = delta
        super().__init__(corpus, tokenizer)

    def _calc_idf(self, nd):
        for word, freq in nd.items():
            idf = math.log(1 + ((self.corpus_size + 0.5 - freq) / (freq+0.5)))
            self.idf[word] = idf

    def get_scores(self, query):
        score = np.zeros(self.corpus_size)
        doc_len = np.array(self.doc_len)
        for q in query:
            q_freq = np.array([(doc.get(q) or 0) for doc in self.doc_freqs])
            score += (self.idf.get(q) or 0) * (self.delta + (q_freq) /
                                               (self.k1 * (1 - self.b + self.b * doc_len / self.avgdl) + q_freq))
        return score


def remove_q(query):
    stop = "|".join(
        "어느 무엇인가요 무엇 누가 누구인가요 누구인가 누구 어디에서 어디에 어디서 어디인가요 어디를 어디 언제 어떤 어떠한 몇 얼마 얼마나 뭐 어떻게 무슨 \?".split(
            " "
        )
    )
    rm = re.sub(stop, "", query).strip()
    return rm

class BertEncoder(BertPreTrainedModel):
  def __init__(self, config):
    super(BertEncoder, self).__init__(config)

    self.bert = BertModel(config)
    self.init_weights()
      
  def forward(self, input_ids, attention_mask=None, token_type_ids=None):   
      outputs = self.bert(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
      
      pooled_output = outputs[1]

      return pooled_output
'''
tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model_checkpoint="bert-base-multilingual-cased"

# load pre-trained model on cuda (if available)
p_encoder = BertEncoder.from_pretrained(model_checkpoint).cuda()
q_encoder = BertEncoder.from_pretrained(model_checkpoint).cuda()

p_encoder.load_state_dict(torch.load("./p_encoder_fin.pth"))
q_encoder.load_state_dict(torch.load("./q_encoder_fin.pth"))

'''

def to_cuda(batch):
  return tuple(t.cuda() for t in batch)
    
class DenseRetrieval:
    def __init__(self, tokenize_fn, data_path="./data/", context_path="wikipedia_documents.json"):
        self.data_path = data_path
        self.wiki_embs = None
        with open(os.path.join(data_path, context_path), "r") as f:
            wiki = json.load(f)
        self.contexts = list(dict.fromkeys([v['text'] for v in wiki.values()]))
        pickle_name = f"./data/dense_embedding.bin"
        if os.path.isfile(pickle_name):
            with open(pickle_name,"rb") as file:
                self.wiki_embs=pickle.load(file)
                print("Pre")
        else:
            with torch.no_grad():
              self.wiki_embs = []
              for text in self.contexts:
                p = tokenizer(text, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
                wiki_emb = p_encoder(**p).to('cpu').numpy()
                self.wiki_embs.append(wiki_emb)
            self.wiki_embs = torch.Tensor(self.wiki_embs).squeeze()  # (num_passage, emb_dim)
            with open(pickle_name,"wb") as file:
                pickle.dump(self.wiki_embs,file)
        self.f=open("./track.txt","w")
                
    def retrieve(self, query_or_dataset, topk=1):
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            return doc_scores, doc_indices 

        elif isinstance(query_or_dataset, Dataset):
            # make retrieved result as dataframe
            total = []
            super_count=0
            doc_scores, doc_indices = self.get_relevant_doc_bulk(query_or_dataset['question'], k=5)
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Dense retrieval: ")):
                tmp = {
                    "question": example["question"],
                    "id": example['id'],
                    #"context_id": doc_indices[idx][0],  # retrieved id
                    "context": " ".join(self.contexts[doc_indices[idx][i]] for i in range(5))  # retrieved doument
                }
                if 'context' in example.keys() and 'answers' in example.keys():
                    tmp["original_context"] = example['context']  # original document
                    tmp["answers"] = example['answers']           # original answer
                total.append(tmp)
            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query, k=3):
        with torch.no_grad():
          q_seqs_val = tokenizer([query], padding="max_length", truncation=True, return_tensors='pt').to('cuda')
          q_emb = q_encoder(**q_seqs_val).to('cpu')  #(num_query, emb_dim)

        result = torch.matmul(q_emb, torch.transpose(self.wiki_embs, 0, 1))
        rank = torch.argsort(result, dim=1, descending=True).squeeze()
        self.f.write("=============\n")
        self.f.write("Query "+query+"\n")
        for idx,i in enumerate(range(k)):
            self.f.write(str(idx)+self.contexts[rank[i]][:75]+"\n")
        print(result.squeeze()[rank].tolist()[:k], rank.tolist()[:k])
        return result.squeeze()[rank].tolist()[:k], rank.tolist()[:k]
        

    def get_relevant_doc_bulk(self, queries, k=1):
        doc_scores = []
        doc_indices = []
        for query in queries:
            ret0,ret1=self.get_relevant_doc(query,k)
            doc_scores.append(ret0)
            doc_indices.append(ret1)
        self.f.close()
        return doc_scores, doc_indices
    

class SparseRetrieval:
    def __init__(self, tokenize_fn, data_path="./data/", context_path="wikipedia_documents.json"):
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v['text'] for v in wiki.values()])) # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # Transform by vectorizer
        self.tfidfv = TfidfVectorizer(
            tokenizer=tokenize_fn,
            ngram_range=(1, 2),
            max_features=50000,
        )

        # should run get_sparse_embedding() or build_faiss() first.
        self.p_embedding = None
        self.indexer = None

    def get_sparse_embedding(self):
        # Pickle save.
        pickle_name = f"sparse_embedding.bin"
        tfidfv_name = f"tfidv.bin"
        emd_path = os.path.join(self.data_path, pickle_name)
        tfidfv_path = os.path.join(self.data_path, tfidfv_name)
        if os.path.isfile(emd_path) and os.path.isfile(tfidfv_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            with open(tfidfv_path, "rb") as file:
                self.tfidfv = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            self.p_embedding = self.tfidfv.fit_transform(self.contexts)
            print(self.p_embedding.shape)
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            with open(tfidfv_path, "wb") as file:
                pickle.dump(self.tfidfv, file)
            print("Embedding pickle saved.")
            
    def retrieve(self, query_or_dataset, topk=3):
        assert self.p_embedding is not None, "You must build faiss by self.get_sparse_embedding() before you run self.retrieve()."
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            return doc_scores, doc_indices 

        elif isinstance(query_or_dataset, Dataset):
            # make retrieved result as dataframe
            total = []
            doc_scores, doc_indices = self.get_relevant_doc_bulk(query_or_dataset['question'], k=20)
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
                tmp = {
                    "question": example["question"],
                    "id": example['id'],
                    #"context_id": doc_indices[idx][0],  # retrieved id
                    "context": " ".join(self.contexts[doc_indices[idx][i]] for i in range(20))  # retrieved doument
                }
                if 'context' in example.keys() and 'answers' in example.keys():
                    tmp["original_context"] = example['context']  # original document
                    tmp["answers"] = example['answers']           # original answer
                total.append(tmp)
            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query, k=1):
        """
        참고: vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        with timer("transform"):
            query_vec = self.tfidfv.transform([query])
        assert (
                np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        with timer("query ex search"):
            result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        sorted_result = np.argsort(result.squeeze())[::-1]
        return result.squeeze()[sorted_result].tolist()[:k], sorted_result.tolist()[:k]

    def get_relevant_doc_bulk(self, queries, k=1):
        query_vec = self.tfidfv.transform(queries)
        assert (
                np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices

    
    
class BM25:
    def __init__(self, tokenize_fn, data_path="./data/", context_path="wikipedia_documents.json", k1=1.2, b=0.25):
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v['text'] for v in wiki.values()])) # set 은 매번 순서가 바뀌므로
        self.l=[]
        for v in self.contexts:
            self.l.append(len(mecab.morphs(v)))
        self.l=np.array(self.l)
        print("Avg",np.average(self.l))
        self.l=self.l/np.average(self.l)
        self.l=(k1*(1-b+b*self.l)).astype(np.float32)
        print("L SHAP",self.l.shape)
        
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # Transform by vectorizer
        self.tfidfv = CountVectorizer(
            tokenizer=tokenize_fn,
            ngram_range=(1, 2),
            max_features=50000,
        )
        
        # should run get_sparse_embedding() or build_faiss() first.
        self.p_embedding = None
        self.a=None
        

    def get_sparse_embedding(self):
        # Pickle save.
        pickle_name = f"bm_embedding.bin"
        tfidfv_name = f"bm.bin"
        emd_path = os.path.join(self.data_path, pickle_name)
        tfidfv_path = os.path.join(self.data_path, tfidfv_name)
        if os.path.isfile(emd_path) and os.path.isfile(tfidfv_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            with open(tfidfv_path, "rb") as file:
                self.tfidfv = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            self.p_embedding = self.tfidfv.fit_transform(self.contexts)
            print(self.p_embedding.shape)
            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
            with open(tfidfv_path, "wb") as file:
                pickle.dump(self.tfidfv, file)
            print("Embedding pickle saved.")
        self.a=self.p_embedding
        z1=(len(self.contexts)-self.a.count_nonzero()+0.5)
        z2=(self.a.count_nonzero()+0.5)
        #z1=(len(self.contexts)-np.count_nonzero(self.a>0,axis=0)+0.5)
        #z2=(np.count_nonzero(self.a>0,axis=0)+0.5)
        z1=np.log(1+z1/z2)
        del(z2)
        self.a=(z1*((self.a)/(self.a+self.l[:,np.newaxis])))
        print(self.a)

    def retrieve(self, query_or_dataset, topk=1):
        assert self.p_embedding is not None, "You must build faiss by self.get_sparse_embedding() before you run self.retrieve()."
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])
            return doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)]

        elif isinstance(query_or_dataset, Dataset):
            # make retrieved result as dataframe
            total = []
            with timer("query exhaustive search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk(query_or_dataset['question'], k=1)
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
                # relev_doc_ids = [el for i, el in enumerate(self.ids) if i in doc_indices[idx]]
                tmp = {
                    "question": example["question"],
                    "id": example['id'],
                    "context_id": doc_indices[idx][0],  # retrieved id
                    "context": self.contexts[doc_indices[idx][0]]  # retrieved doument
                }
                if 'context' in example.keys() and 'answers' in example.keys():
                    tmp["original_context"] = example['context']  # original document
                    tmp["answers"] = example['answers']           # original answer
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query, k=1):
        query_vec = self.tfidfv.transform([query])
        print("Query_vec",query_vec.toarray())
        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        sorted_result = np.argsort(result.squeeze())[::-1]
        return result.squeeze()[sorted_result].tolist()[:k], sorted_result.tolist()[:k]

    def get_relevant_doc_bulk(self, queries, k=1):
        query_vec = self.tfidfv.transform(queries)
        result = query_vec * self.p_embedding.T
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        doc_scores = []
        doc_indices = []
        for i in range(result.shape[0]):
            sorted_result = np.argsort(result[i, :])[::-1]
            doc_scores.append(result[i, :][sorted_result].tolist()[:k])
            doc_indices.append(sorted_result.tolist()[:k])
        return doc_scores, doc_indices

    
class BM25Arti:
    def __init__(self, tokenize_fn, data_path="./data/", context_path="wikipedia_documents.json"):
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v['text'] for v in wiki.values()])) # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # Transform by vectorizer
        pickle_name=f"token_corpus"
        emd_path = os.path.join(self.data_path, pickle_name)
        
        self.tokenized_corpus = None
        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.tokenized_corpus = pickle.load(file)
        else:
            self.tokenized_corpus = [tokenize_fn(doc) for doc in self.contexts]
            with open(emd_path, "wb") as file:
                pickle.dump(self.tokenized_corpus, file)
        
        self.bm25 = None
        self.sCounter=Counter()
        
        pickle_name = f"bm25.bin"
        emd_path = os.path.join(self.data_path, pickle_name)
        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.bm25 = pickle.load(file)
        else:
            self.bm25 = ES(self.tokenized_corpus)
            with open(emd_path, "wb") as file:
                pickle.dump(self.bm25, file)


    def retrieve(self, query_or_dataset, topk=20):
        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc(query_or_dataset, k=topk)
            return doc_scores, doc_indices 

        elif isinstance(query_or_dataset, Dataset):
            # make retrieved result as dataframe
            f=open("./context.txt","w")
            total = []
            doc_scores, doc_indices = self.get_relevant_doc_bulk(query_or_dataset['question'], k=35)
            for idx, example in enumerate(tqdm(query_or_dataset, desc="Sparse retrieval: ")):
                standard=doc_scores[idx][0]
                self.sCounter+=Counter(doc_indices[idx])
                print("S-score",standard)
                standard_idx=0
                for idx2, doc_score in enumerate(doc_scores[idx]):
                    if doc_score>=standard*0.7:
                        standard_idx=idx2
                print("SIDX",standard_idx)
                print("DID",doc_indices[idx])
                temp_doc=" ".join(self.contexts[doc_indices[idx][i]] for i in range(standard_idx+1))
                f.write("\n------------\n")
                f.write(str(doc_indices[idx]))
                f.write("\n")
                f.write(example["question"]+"\n")
                for idid in range(standard_idx+1):
                    f.write("Doc id: ")
                    f.write(str(doc_indices[idx][idid]))
                    f.write("\n")
                    f.write(self.contexts[doc_indices[idx][idid]])
                    
                print(temp_doc[:75])
                tmp = {
                    "question": example["question"],
                    "id": example['id'],
                    #"context_id": doc_indices[idx][0],  # retrieved id
                    "context":temp_doc   # retrieved doument
                }
                if 'context' in example.keys() and 'answers' in example.keys():
                    tmp["original_context"] = example['context']  # original document
                    tmp["answers"] = example['answers']           # original answer
                total.append(tmp)
            cqas = pd.DataFrame(total)
            f.write(str(self.sCounter.most_common(10)))
            f.close()
            return cqas

    def get_relevant_doc(self, query, k=3):
        
        result=self.bm25.get_scores(mecab.morphs(query))
        if not isinstance(result, np.ndarray):
            result = result.toarray()
        sorted_result = np.argsort(result.squeeze())[::-1]
        #print("Query "+query)
        #print(result.squeeze()[sorted_result].tolist()[:k], sorted_result.tolist()[:k])
        return result.squeeze()[sorted_result].tolist()[:k], sorted_result.tolist()[:k]

    def get_relevant_doc_bulk(self, queries, k=3):
        
        doc_scores = []
        doc_indices = []
        for query in queries:
            ret0,ret1=self.get_relevant_doc(query,k)
            doc_scores.append(ret0)
            doc_indices.append(ret1)
        
        return doc_scores, doc_indices
 
if __name__ == "__main__":
    # Test sparse
    org_dataset = load_from_disk("data/train_dataset")
    full_ds = concatenate_datasets(
        [
            org_dataset["train"].flatten_indices(),
            org_dataset["validation"].flatten_indices(),
        ]
    ) # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    print("*"*40, "query dataset", "*"*40)
    print(full_ds)

    mecab = Mecab()
    def tokenize(text):
        # return text.split(" ")
        return mecab.morphs(text)


    wiki_path = "wikipedia_documents.json"
    retriever = BM25Arti(
        # tokenize_fn=tokenizer.tokenize,
        tokenize_fn=tokenize,
        data_path="data",
        context_path=wiki_path)
    query=" 1994년 FIFA 월드컵 당시 대한민국 축구 대표팀의 감독은 누구인가?"
    print(query)
    print(retriever.retrieve(query))