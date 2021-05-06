import argparse
import random
import os

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from datasets import load_from_disk, concatenate_datasets
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import (
    AutoTokenizer,
    AdamW,
    TrainingArguments,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)
from tqdm import tqdm, trange
from datetime import datetime
from pytz import timezone

from utils_qa import set_seed
from models import BertEncoder

os.environ["WANDB_PROJECT"] = "p-stage3-odqa-retriever"
os.environ["WANDB_LOG_MODEL"] = "true"


def main():
    set_seed(42)
    now = datetime.now(timezone("Asia/Seoul")).strftime("%Y-%m-%d-%H:%M:%S")
    args = TrainingArguments(
        output_dir="dense_retrieval",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=40,
        weight_decay=0.01,
    )

    model_checkpoint = "bert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    wandb.init(
        name=now,
        config={
            "encoder_backbone": model_checkpoint,
            "batch_size": args.per_device_train_batch_size,
            "initial_lr": args.learning_rate,
            "epochs": args.num_train_epochs,
            "weight_decay": args.weight_decay,
        },
    )

    dataset_path = "/opt/ml/input/data/train_dataset"
    datasets = load_from_disk(dataset_path)

    # Train encoder on question-context pair from train and validation datasets.
    training_dataset = concatenate_datasets(
        [
            datasets["train"].flatten_indices(),
            datasets["validation"].flatten_indices(),
        ]
    )

    q_seqs = tokenizer(
        training_dataset["question"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    p_seqs = tokenizer(
        training_dataset["context"],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    train_dataset = TensorDataset(
        p_seqs["input_ids"],
        p_seqs["attention_mask"],
        p_seqs["token_type_ids"],
        q_seqs["input_ids"],
        q_seqs["attention_mask"],
        q_seqs["token_type_ids"],
    )

    # load pre-trained model on cuda (if available)
    p_encoder = BertEncoder.from_pretrained(model_checkpoint)
    q_encoder = BertEncoder.from_pretrained(model_checkpoint)
    if torch.cuda.is_available():
        p_encoder.cuda()
        q_encoder.cuda()

    # self.p_encoder = BertEncoder.from_pretrained(
    #         "/opt/ml/p3-mrc-gaama/dense_encoder/p_encoder"
    #     ).cuda()
    #     self.q_encoder = BertEncoder.from_pretrained(
    #         "/opt/ml/p3-mrc-gaama/dense_encoder/q_encoder"
    #     ).cuda()

    save_to = "./dense_encoder"

    def train(args, dataset, p_model, q_model, load_from=None):
        # Dataloader
        train_sampler = RandomSampler(dataset)
        train_dataloader = DataLoader(
            dataset, sampler=train_sampler, batch_size=args.per_device_train_batch_size
        )

        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )
        optimizer = AdamW(
            [{"params": p_model.parameters()}, {"params": q_model.parameters()}],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(t_total * 0.06), num_training_steps=t_total
        )

        # Start training!
        global_step = 0
        start_epoch = 0

        if load_from is not None:
            checkpoint = torch.load(os.path.join(save_to, f"epoch-{load_from}.pt"))
            p_model.load_state_dict(checkpoint["p_model"])
            q_model.load_state_dict(checkpoint["q_model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = checkpoint["epoch"]

            steps = (
                len(train_dataloader) // args.gradient_accumulation_steps * start_epoch
            )
            print(
                f">>>> Continue training from epoch {load_from}, skipping first {steps} steps."
            )

            for _ in range(steps):
                scheduler.step()
                global_step += 1

        p_model.zero_grad()
        q_model.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = trange(int(args.num_train_epochs) - start_epoch, desc="Epoch")

        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")

            for step, batch in enumerate(epoch_iterator):
                q_encoder.train()
                p_encoder.train()

                if torch.cuda.is_available():
                    batch = tuple(t.cuda() for t in batch)

                p_inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                }

                q_inputs = {
                    "input_ids": batch[3],
                    "attention_mask": batch[4],
                    "token_type_ids": batch[5],
                }

                p_outputs = p_model(**p_inputs)  # (batch_size, emb_dim)
                q_outputs = q_model(**q_inputs)  # (batch_size, emb_dim)

                # Calculate similarity score & loss
                sim_scores = torch.matmul(
                    q_outputs, torch.transpose(p_outputs, 0, 1)
                )  # (batch_size, emb_dim) x (emb_dim, batch_size) = (batch_size, batch_size)

                # target: position of positive samples = diagonal element
                targets = torch.arange(0, args.per_device_train_batch_size).long()
                if torch.cuda.is_available():
                    targets = targets.to("cuda")

                sim_scores = F.log_softmax(sim_scores, dim=1)

                loss = F.nll_loss(sim_scores, targets)
                if step % 100 == 0:
                    print(f"Epoch {epoch+start_epoch}, step {step}: loss={loss}")
                    wandb.log(
                        {
                            "learning_rate": scheduler.get_last_lr()[0],
                            "loss": loss,
                        },
                        step=global_step,
                    )

                loss.backward()
                optimizer.step()
                scheduler.step()
                q_model.zero_grad()
                p_model.zero_grad()
                global_step += 1

                torch.cuda.empty_cache()

            if (epoch + start_epoch + 1) % 8 == 0:
                if not os.path.exists(save_to):
                    os.makedirs(save_to)
                torch.save(
                    {
                        "epoch": epoch + start_epoch + 1,
                        "p_model": p_model.state_dict(),
                        "q_model": q_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    os.path.join(save_to, f"epoch-{epoch + start_epoch}.pt"),
                )

        return p_model, q_model

    p_encoder, q_encoder = train(args, train_dataset, p_encoder, q_encoder)


if __name__ == "__main__":
    main()
