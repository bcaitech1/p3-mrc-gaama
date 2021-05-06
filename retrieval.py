import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm.auto import tqdm
import pandas as pd
import pickle
import json
import os
import numpy as np
import torch

from transformers import BertModel, BertPreTrainedModel, BertConfig, AutoTokenizer

from datasets import Dataset, load_from_disk, concatenate_datasets
from konlpy.tag import Mecab
from rank_bm25 import BM25Okapi

from utils_qa import set_seed

import time
from contextlib import contextmanager


@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f"[{name}] done in {time.time() - t0:.3f} s")


class BertEncoder(BertPreTrainedModel):
    def __init__(self, config):
        super(BertEncoder, self).__init__(config)

        self.bert = BertModel(config)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

        pooled_output = outputs[1]
        return pooled_output


class DenseRetrieval:
    def __init__(
        self,
        data_path="./data/",
        context_path="wikipedia_documents.json",
        model_name="bert-base-multilingual-cased",
        model_path="/opt/ml/dense_encoder/encoder.pth",
    ):
        # Load wikipedia data
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r") as f:
            wiki = json.load(f)

        self.contexts = list(dict.fromkeys([v["text"] for v in wiki.values()]))

        # Create passage and question encoders
        self.p_encoder = BertEncoder.from_pretrained(
            "/opt/ml/p3-mrc-gaama/dense_encoder/p_encoder"
        ).cuda()
        self.q_encoder = BertEncoder.from_pretrained(
            "/opt/ml/p3-mrc-gaama/dense_encoder/q_encoder"
        ).cuda()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.p_embedding = None

    def get_dense_embedding(self):
        pickle_name = f"dense_embedding.bin"
        emd_path = os.path.join(self.data_path, pickle_name)

        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.p_embedding = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            with torch.no_grad():
                self.p_encoder.eval()

                self.p_embedding = []
                for p in tqdm(self.contexts):
                    p = self.tokenizer(
                        p, padding="max_length", truncation=True, return_tensors="pt"
                    ).to("cuda")
                    p_emb = self.p_encoder(**p).to("cpu").numpy()
                    self.p_embedding.append(p_emb)

            self.p_embedding = torch.Tensor(
                self.p_embedding
            ).squeeze()  # (num_passage, emb_dim)

            with open(emd_path, "wb") as file:
                pickle.dump(self.p_embedding, file)
                print("Embedding pickle saved.")

        print(self.p_embedding.shape)

    def retrieve(self, query_or_dataset, topk=1):
        assert (
            self.p_embedding is not None
        ), "You must build faiss by self.get_dense_embedding() before you run self.retrieve()."
        print(f">>>> Called retrieve with type {type(query_or_dataset)}")
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
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=1
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Dense retrieval: ")
            ):
                # relev_doc_ids = [el for i, el in enumerate(self.ids) if i in doc_indices[idx]]
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context_id": doc_indices[idx][0],  # retrieved id
                    "context": self.contexts[doc_indices[idx][0]],  # retrieved doument
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]  # original document
                    tmp["answers"] = example["answers"]  # original answer

                # print("Top-1 passage with score %.4f" % (doc_scores[idx][0]))
                # print(self.contexts[doc_indices[idx][0]])
                # print("-" * 40)
                # print(tmp["original_context"])
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query, k=1):
        with torch.no_grad():
            self.q_encoder.eval()

            # 효율성을 위해서는 지금처럼 한번에 한 query + get_relevant_doc 여러번 부르기보다,
            # tokenizer 입력으로 [query, query2] 이런식으로 넣어주는 것.
            # 혹은 아예 함수 입력을 query_list 로 받은 후 bracket 없이 tokenizer 에 넣어주는 것
            q_seqs_val = self.tokenizer(
                [query], padding="max_length", truncation=True, return_tensors="pt"
            ).to("cuda")
            q_emb = self.q_encoder(**q_seqs_val).to("cpu")  # (num_query, emb_dims)

        dot_prod_scores = torch.matmul(q_emb, torch.transpose(self.p_embedding, 0, 1))
        rank = torch.argsort(dot_prod_scores, descending=True).squeeze()

        return dot_prod_scores.squeeze()[rank][:k], rank[:k]

    def get_relevant_doc_bulk(self, queries, k=1):
        with torch.no_grad():
            self.q_encoder.eval()

            q_seqs_val = self.tokenizer(
                queries, padding="max_length", truncation=True, return_tensors="pt"
            ).to("cuda")
            q_embs = self.q_encoder(**q_seqs_val).to("cpu")  # (num_query, emb_dims)

        result = torch.matmul(q_embs, torch.transpose(self.p_embedding, 0, 1))
        doc_scores = []
        doc_indices = []
        for i in range(q_embs.shape[0]):
            # sorted_result = np.argsort(result[i, :])[::-1]
            rank = torch.argsort(result[i, :], descending=True).squeeze()
            doc_scores.append(result[i, :][rank][:k])
            doc_indices.append(rank[:k])

        return doc_scores, doc_indices


class SparseBM25Retrieval:
    def __init__(
        self, tokenize_fn, data_path="./data/", context_path="wikipedia_documents.json"
    ):
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
        print(f"Lengths of unique contexts : {len(self.contexts)}")
        self.ids = list(range(len(self.contexts)))

        # should run get_sparse_embedding() first.
        self.bm25 = None
        self.tokenize_fn = tokenize_fn

    def get_sparse_embedding(self):
        # Pickle save.
        pickle_name = f"sparse_bm25_embedding.bin"
        emd_path = os.path.join(self.data_path, pickle_name)
        if os.path.isfile(emd_path):
            with open(emd_path, "rb") as file:
                self.bm25 = pickle.load(file)
            print("Embedding pickle load.")
        else:
            print("Build passage embedding")
            tokenized_corpus = list(map(self.tokenize_fn, self.contexts))
            print(len(tokenized_corpus))
            self.bm25 = BM25Okapi(tokenized_corpus)
            with open(emd_path, "wb") as file:
                pickle.dump(self.bm25, file)
            print("Embedding pickle saved.")

    def retrieve(self, query_or_dataset, topk=1):
        assert (
            self.bm25 is not None
        ), "You must build faiss by self.get_sparse_embedding() before you run self.retrieve()."
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
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=1
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context_id": doc_indices[idx][0],  # retrieved id
                    "context": self.contexts[doc_indices[idx][0]],  # retrieved doument
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]  # original document
                    tmp["answers"] = example["answers"]  # original answer
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc(self, query, k=1):
        with timer("transform"):
            query_vec = self.tokenize_fn(query)
        with timer("query ex search"):
            doc_scores = self.bm25.get_scores(query_vec)
        print(doc_scores.shape)
        sorted_result = np.argsort(doc_scores)[::-1]
        return doc_scores[sorted_result][:k], sorted_result[:k]

    def get_relevant_doc_bulk(self, queries, k=1):
        query_vec = list(map(self.tokenize_fn, queries))

        doc_scores = []
        doc_indices = []
        for i in tqdm(range(len(query_vec))):
            result = self.bm25.get_scores(query_vec[i])
            sorted_result = np.argsort(result)[::-1]
            doc_scores.append(result[sorted_result][:k])
            doc_indices.append(sorted_result[:k])
        return doc_scores, doc_indices


class SparseRetrieval:
    def __init__(
        self, tokenize_fn, data_path="./data/", context_path="wikipedia_documents.json"
    ):
        self.data_path = data_path
        with open(os.path.join(data_path, context_path), "r") as f:
            wiki = json.load(f)

        self.contexts = list(
            dict.fromkeys([v["text"] for v in wiki.values()])
        )  # set 은 매번 순서가 바뀌므로
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

    def build_faiss(self):
        # FAISS build
        num_clusters = 16
        niter = 5

        # 1. Clustering
        p_emb = self.p_embedding.toarray().astype(np.float32)
        emb_dim = p_emb.shape[-1]
        index_flat = faiss.IndexFlatL2(emb_dim)

        clus = faiss.Clustering(emb_dim, num_clusters)
        clus.verbose = True
        clus.niter = niter
        clus.train(p_emb, index_flat)

        centroids = faiss.vector_float_to_array(clus.centroids)
        centroids = centroids.reshape(num_clusters, emb_dim)

        quantizer = faiss.IndexFlatL2(emb_dim)
        quantizer.add(centroids)

        # 2. SQ8 + IVF indexer (IndexIVFScalarQuantizer)
        self.indexer = faiss.IndexIVFScalarQuantizer(
            quantizer, quantizer.d, quantizer.ntotal, faiss.METRIC_L2
        )
        self.indexer.train(p_emb)
        self.indexer.add(p_emb)

    def retrieve(self, query_or_dataset, topk=1):
        assert (
            self.p_embedding is not None
        ), "You must build faiss by self.get_sparse_embedding() before you run self.retrieve()."
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
                doc_scores, doc_indices = self.get_relevant_doc_bulk(
                    query_or_dataset["question"], k=1
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                # relev_doc_ids = [el for i, el in enumerate(self.ids) if i in doc_indices[idx]]
                tmp = {
                    "question": example["question"],
                    "id": example["id"],
                    "context_id": doc_indices[idx][0],  # retrieved id
                    "context": self.contexts[doc_indices[idx][0]],  # retrieved doument
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]  # original document
                    tmp["answers"] = example["answers"]  # original answer
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

    def retrieve_faiss(self, query_or_dataset, topk=1):
        assert (
            self.indexer is not None
        ), "You must build faiss by self.build_faiss() before you run self.retrieve_faiss()."

        if isinstance(query_or_dataset, str):
            doc_scores, doc_indices = self.get_relevant_doc_faiss(
                query_or_dataset, k=topk
            )
            print("[Search query]\n", query_or_dataset, "\n")

            for i in range(topk):
                print("Top-%d passage with score %.4f" % (i + 1, doc_scores[i]))
                print(self.contexts[doc_indices[i]])
            return doc_scores, [self.contexts[doc_indices[i]] for i in range(topk)]

        elif isinstance(query_or_dataset, Dataset):
            queries = query_or_dataset["question"]
            # make retrieved result as dataframe
            total = []
            with timer("query faiss search"):
                doc_scores, doc_indices = self.get_relevant_doc_bulk_faiss(
                    queries, k=topk
                )
            for idx, example in enumerate(
                tqdm(query_or_dataset, desc="Sparse retrieval: ")
            ):
                # relev_doc_ids = [el for i, el in enumerate(self.ids) if i in doc_indices[idx]]

                tmp = {
                    "question": example["question"],
                    "id": example["id"],  # original id
                    "context_id": doc_indices[idx][0],  # retrieved id
                    "context": self.contexts[doc_indices[idx][0]],  # retrieved doument
                }
                if "context" in example.keys() and "answers" in example.keys():
                    tmp["original_context"] = example["context"]  # original document
                    tmp["answers"] = example["answers"]  # original answer
                total.append(tmp)

            cqas = pd.DataFrame(total)
            return cqas

    def get_relevant_doc_faiss(self, query, k=1):
        """
        참고: vocab 에 없는 이상한 단어로 query 하는 경우 assertion 발생 (예) 뙣뙇?
        """
        query_vec = self.tfidfv.transform([query])
        assert (
            np.sum(query_vec) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."

        q_emb = query_vec.toarray().astype(np.float32)
        with timer("query faiss search"):
            D, I = self.indexer.search(q_emb, k)
        return D.tolist()[0], I.tolist()[0]

    def get_relevant_doc_bulk_faiss(self, queries, k=1):
        query_vecs = self.tfidfv.transform(queries)

        assert (
            np.sum(query_vecs) != 0
        ), "오류가 발생했습니다. 이 오류는 보통 query에 vectorizer의 vocab에 없는 단어만 존재하는 경우 발생합니다."
        q_embs = query_vecs.toarray().astype(np.float32)
        D, I = self.indexer.search(q_embs, k)
        return D.tolist(), I.tolist()


if __name__ == "__main__":
    set_seed(42)

    # Test sparse
    org_dataset = load_from_disk("/opt/ml/input/data/train_dataset")

    # full_ds = concatenate_datasets(
    #     [
    #         org_dataset["train"].flatten_indices(),
    #         org_dataset["validation"].flatten_indices(),
    #     ]
    # )  # train dev 를 합친 4192 개 질문에 대해 모두 테스트
    # print("*" * 40, "query dataset", "*" * 40)
    # print(full_ds)

    small_ds = concatenate_datasets(
        [
            org_dataset["train"].select(range(500)).flatten_indices(),
            org_dataset["validation"].select(range(100)).flatten_indices(),
        ]
    )
    print("*" * 40, "query dataset", "*" * 40)
    print(small_ds)

    ### Mecab 이 가장 높은 성능을 보였기에 mecab 으로 선택 했습니다 ###
    mecab = Mecab()

    def tokenize(text):
        # return text.split(" ")
        return mecab.morphs(text)

    # from transformers import AutoTokenizer
    #
    # tokenizer = AutoTokenizer.from_pretrained(
    #     "bert-base-multilingual-cased",
    #     use_fast=True,
    # )
    ###############################################################

    wiki_path = "wikipedia_documents.json"
    # retriever_s = SparseBM25Retrieval(
    #     # tokenize_fn=tokenizer.tokenize,
    #     tokenize_fn=tokenize,
    #     data_path="/opt/ml/input/data",
    #     context_path=wiki_path,
    # )
    # retriever_s.get_sparse_embedding()

    retriever_d = DenseRetrieval(
        data_path="/opt/ml/input/data",
        context_path=wiki_path,
    )
    retriever_d.get_dense_embedding()

    # retriever.build_faiss()

    # test single query
    query = "대통령을 포함한 미국의 행정부 견제권을 갖는 국가 기관은?"
    # query = org_dataset["validation"][2]["question"]

    with timer("single query by exhaustive search"):
        scores, indices = retriever_d.retrieve(query, topk=1)
    # with timer("single query by faiss"):
    #     scores, indices = retriever.retrieve_faiss(query)

    # test bulk
    with timer("bulk query by exhaustive search"):
        df = retriever_d.retrieve(small_ds)
        df["correct"] = df["original_context"] == df["context"]
        print(
            "correct retrieval result by exhaustive search",
            df["correct"].sum() / len(df),
        )

    # with timer("bulk query by exhaustive search"):
    #     df = retriever.retrieve_faiss(full_ds)
    #     df["correct"] = df["original_context"] == df["context"]
    #     print("correct retrieval result by faiss", df["correct"].sum() / len(df))
