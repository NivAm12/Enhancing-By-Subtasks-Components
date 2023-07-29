from datasets import load_dataset, Dataset
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
import torch
import torch.nn as nn


def load_data(dataset_benchmark: str, dataset_name: str, train_samples: int, val_samples: int = -1,
              test_samples: int = -1):
    train_dataset = load_dataset(dataset_benchmark, dataset_name, split=f"train[:{train_samples}]")
    val_dataset = load_dataset(dataset_benchmark, dataset_name, split=f"validation[:{val_samples}]")
    test_dataset = load_dataset(dataset_benchmark, dataset_name, split=f"test[:{test_samples}]")

    return {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset,
    }


def preprocess_func(examples, tokenizer):
    return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True)


def preprocess_dataset(dataset: Dataset, tokenizer, preprocess_fn):
    tokenized_datasets = dataset.map(preprocess_fn, fn_kwargs={"tokenizer": tokenizer})
    tokenized_datasets = tokenized_datasets.select_columns(["input_ids", "token_type_ids", "attention_mask", "label"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")

    return tokenized_datasets


def get_dataloader(dataset, tokenizer, batch_size: int, shuffle: bool):
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt", padding=True)

    train_dataloader = DataLoader(
        dataset,
        collate_fn=data_collator,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=True,
        num_workers=0)

    return train_dataloader


def compute_metrics(logits, labels, metric, metric_type):
    if metric_type == 'accuracy':
        logits = nn.Sigmoid()(logits)
        logits = torch.round(logits).int()

    result = metric.compute(predictions=logits, references=labels)
    result = result['accuracy'] if metric_type == 'accuracy' else result['mse']

    return result