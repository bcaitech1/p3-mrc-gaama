#dense
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm, trange
import argparse
import random
import torch
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel, AdamW, TrainingArguments, get_linear_schedule_with_warmup

import numpy as np

dataset = load_dataset("squad_kor_v1")


tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
model_checkpoint="bert-base-multilingual-cased"

#set seed
torch.manual_seed(2021)
torch.cuda.manual_seed(2021)
np.random.seed(2021)
random.seed(2021)

#prepare train dataset
training_dataset = dataset['train']

from torch.utils.data import (DataLoader, RandomSampler, TensorDataset)

q_seqs = tokenizer(training_dataset['question'], padding="max_length", truncation=True, return_tensors='pt')
p_seqs = tokenizer(training_dataset['context'], padding="max_length", truncation=True, return_tensors='pt')

train_dataset = TensorDataset(p_seqs['input_ids'], p_seqs['attention_mask'], p_seqs['token_type_ids'], 
                        q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids'])

class BertEncoder(BertPreTrainedModel):
  def __init__(self, config):
    super(BertEncoder, self).__init__(config)

    self.bert = BertModel(config)
    self.init_weights()
      
  def forward(self, input_ids, attention_mask=None, token_type_ids=None):   
      outputs = self.bert(input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
      
      pooled_output = outputs[1]

      return pooled_output

# load pre-trained model on cuda (if available)
p_encoder = BertEncoder.from_pretrained(model_checkpoint)
q_encoder = BertEncoder.from_pretrained(model_checkpoint)

if torch.cuda.is_available():
  p_encoder.cuda()
  q_encoder.cuda()

def train(args, dataset, p_model, q_model):
  
  # Dataloader
  train_sampler = RandomSampler(dataset)
  train_dataloader = DataLoader(dataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size)

  t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
  optimizer = AdamW([
                {'params': p_model.parameters()},
                {'params': q_model.parameters()}
            ], lr=args.learning_rate, weight_decay=args.weight_decay)
  scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

  # Start training!
  global_step = 0
  
  p_model.zero_grad()
  q_model.zero_grad()
  torch.cuda.empty_cache()
  
  train_iterator = trange(int(args.num_train_epochs), desc="Epoch")

  for _ in train_iterator:
    epoch_iterator = tqdm(train_dataloader, desc="Iteration")

    for step, batch in enumerate(epoch_iterator):
      q_encoder.train()
      p_encoder.train()
      
      if torch.cuda.is_available():
        batch = tuple(t.cuda() for t in batch)

      p_inputs = {'input_ids': batch[0],
                  'attention_mask': batch[1],
                  'token_type_ids': batch[2]
                  }
      
      q_inputs = {'input_ids': batch[3],
                  'attention_mask': batch[4],
                  'token_type_ids': batch[5]}
      
      p_outputs = p_model(**p_inputs)  # (batch_size, emb_dim)
      q_outputs = q_model(**q_inputs)  # (batch_size, emb_dim)


      # Calculate similarity score & loss
      sim_scores = torch.matmul(q_outputs, torch.transpose(p_outputs, 0, 1))  # (batch_size, emb_dim) x (emb_dim, batch_size) = (batch_size, batch_size)

      # target: position of positive samples = diagonal element 
      targets = torch.arange(0, len(sim_scores)).long()
      if torch.cuda.is_available():
        targets = targets.to('cuda')

      sim_scores = F.log_softmax(sim_scores, dim=1)

      loss = F.nll_loss(sim_scores, targets)
      print(loss)

      loss.backward()
      optimizer.step()
      scheduler.step()
      q_model.zero_grad()
      p_model.zero_grad()
      global_step += 1
      
      torch.cuda.empty_cache()
    if global_step%15000==0:
        torch.save(p_model.state_dict(),f"./p_encoder_{global_step}.pth")
        torch.save(q_model.state_dict(),f"./q_encoder_{global_step}.pth")


    
  return p_model, q_model

args = TrainingArguments(
    output_dir="dense_retireval",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=10,
    per_device_eval_batch_size=10,
    num_train_epochs=5, # For demonstration
    weight_decay=0.1
)

#train and save
p_encoder, q_encoder = train(args, train_dataset, p_encoder, q_encoder)
torch.save(p_encoder.state_dict(),"./p_encoder_fin.pth")
torch.save(q_encoder.state_dict(),"./q_encoder_fin.pth")


valid_corpus = list(set([example['context'] for example in dataset['validation']]))[:10]
sample_idx = random.choice(range(len(dataset['validation'])))
query = dataset['validation'][sample_idx]['question']
ground_truth = dataset['validation'][sample_idx]['context']

if not ground_truth in valid_corpus:
  valid_corpus.append(ground_truth)

print(query)
print(ground_truth, '\n\n')
###
def to_cuda(batch):
  return tuple(t.cuda() for t in batch)

####
with torch.no_grad():
  p_encoder.eval()
  q_encoder.eval()

  q_seqs_val = tokenizer([query], padding="max_length", truncation=True, return_tensors='pt').to('cuda')
  q_emb = q_encoder(**q_seqs_val).to('cpu')  #(num_query, emb_dim)

  p_embs = []
  for p in valid_corpus:
    p = tokenizer(p, padding="max_length", truncation=True, return_tensors='pt').to('cuda')
    p_emb = p_encoder(**p).to('cpu').numpy()
    p_embs.append(p_emb)

p_embs = torch.Tensor(p_embs).squeeze()  # (num_passage, emb_dim)

print(p_embs.size(), q_emb.size())

dot_prod_scores = torch.matmul(q_emb, torch.transpose(p_embs, 0, 1))
print(dot_prod_scores.size())

rank = torch.argsort(dot_prod_scores, dim=1, descending=True).squeeze()
print(dot_prod_scores)
print(rank)

####
k = 5
print("[Search query]\n", query, "\n")
print("[Ground truth passage]")
print(ground_truth, "\n")

for i in range(k):
  print("Top-%d passage with score %.4f" % (i+1, dot_prod_scores.squeeze()[rank[i]]))
  print(valid_corpus[rank[i]])