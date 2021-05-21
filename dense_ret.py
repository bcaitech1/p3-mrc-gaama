import json
from transformers import AutoTokenizer
import numpy as np
from tqdm import tqdm, trange
import argparse
import random
import torch
import os
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel, AdamW, TrainingArguments, get_linear_schedule_with_warmup
import pickle

class BertEncoder(BertPreTrainedModel):
  def __init__(self, config):
    super(BertEncoder, self).__init__(config)

    self.bert = BertModel(config)
    self.init_weights()
      
  def forward(self, input_ids, attention_mask=None, token_type_ids=None):   
      outputs = self.bert(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
      
      pooled_output = outputs[1]

      return pooled_output

tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model_checkpoint="bert-base-multilingual-cased"






# load pre-trained model on cuda (if available)
p_encoder = BertEncoder.from_pretrained(model_checkpoint).cuda()
q_encoder = BertEncoder.from_pretrained(model_checkpoint).cuda()


p_encoder.load_state_dict(torch.load("./p_encoder_fin.pth"))
q_encoder.load_state_dict(torch.load("./q_encoder_fin.pth"))

def to_cuda(batch):
  return tuple(t.cuda() for t in batch)
data_path="./data/wikipedia_documents.json"
with open(data_path,"r") as f:
    wiki=json.load(f)
corpus = [document['text'] for document_id, document in wiki.items()]
pickle_name = f"./data/dense_embedding.bin"
if os.path.isfile(pickle_name):
    with open(pickle_name,"rb") as file:
        wiki_embs=pickle.load(file)
else:
    
    with torch.no_grad():
      wiki_embs = []
      for text in corpus:
        p = tokenizer(text, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
        wiki_emb = p_encoder(**p).to('cpu').numpy()
        wiki_embs.append(wiki_emb)

    wiki_embs = torch.Tensor(wiki_embs).squeeze()  # (num_passage, emb_dim)
    with open(pickle_name,"wb") as file:
        pickle.dump(wiki_embs,file)
# TODO: embed query to dense vector

query = input("Query? ")
with torch.no_grad():
  q_seqs_val = tokenizer([query], padding="max_length", truncation=True, return_tensors='pt').to('cuda')
  q_emb = q_encoder(**q_seqs_val).to('cpu')  #(num_query, emb_dim)

result = torch.matmul(q_emb, torch.transpose(wiki_embs, 0, 1))
result.shape
k = 5
rank = torch.argsort(result, dim=1, descending=True).squeeze()
k = 5
print("[Search query]\n", query, "\n")

for i in range(k):
  print("Top-%d passage with score %.4f" % (i+1, result.squeeze()[rank[i]]))
  print(corpus[rank[i]])