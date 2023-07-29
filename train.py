import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from transformers import AutoConfig, AutoTokenizer, AutoModel
from models.multiHeadModel import MultiHeadModel
from models.heads import ClassificationHead
import os
import argparse
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets import load_dataset, Dataset
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader
from evaluate import load


def train(multi_head_model: nn.Module, heads_props: dict, train_args: argparse.Namespace):
    """
    Trains a multi-head model.

    Args:
        multi_head_model (nn.Module): The multi-head model to train.
        heads_props (dict): A dictionary of head properties, where each key is the head name and each value is a dictionary of head properties.
        train_args (dict): A dictionary of training arguments.

    Returns:
        None.
    """
    optim = train_args.optim(multi_head_model.parameters(), lr=train_args.lr, betas=train_args.betas,
                             weight_decay=train_args.weight_decay)
    scheduler = CosineAnnealingLR(optim, T_max=train_args.epochs)

    # create the data loaders list
    train_loaders = [head_prop['train_loader'] for head_prop in heads_props.values()]
    val_loaders = [head_prop['val_loader'] for head_prop in heads_props.values()]

    for epoch in tqdm(range(train_args.epochs)):
        # calculate train and val losses per epoch
        train_epoch_loss, heads_train_losses, _ = run_epoch(multi_head_model, train_loaders, heads_props, train_args,
                                                            optim, scheduler, do_train=True)
        val_epoch_loss, heads_val_losses, head_evals_scores = run_epoch(multi_head_model, val_loaders, heads_props,
                                                                        train_args,
                                                                        do_train=False)

        wandb.log({
            'train_loss': train_epoch_loss,
            'val_loss': val_epoch_loss,
            **{f'{key}_loss': val for key, val in heads_train_losses.items()},
            **{f'val_{key}_loss': val for key, val in heads_val_losses.items()},
            **{f'{key}_score': val for key, val in head_evals_scores.items()}
        })

        # save the model at each epoch
        if not os.path.exists(train_args.save_path):
            os.mkdir(train_args.save_path)

        torch.save({
            'epoch': epoch,
            'model_state_dict': multi_head_model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
        }, f'{train_args.save_path}/multi_head_epoch{epoch}.pt')


def run_epoch(model, data_loaders, heads_props, train_args, optim=None, scheduler=None, do_train=True):
    epoch_loss = 0.0
    head_losses = {head_name: 0.0 for head_name in heads_props.keys()}  # Dictionary to store the losses for each head
    head_evals = {head_name: 0.0 for head_name in heads_props.keys()}  # Dictionary to store the losses for each head

    # prepare the model weights for training or validation:
    if do_train:
        model.train()
    else:
        model.eval()

    # iterate the batches simultaneously
    for i, combined_batch in enumerate(zip(*data_loaders)):
        step_loss = 0.0

        for task_batch, head_name in zip(combined_batch, heads_props.keys()):
            head_props = heads_props[head_name]
            critic = head_props['loss_func']
            task_batch = task_batch.to(train_args.device)

            with torch.set_grad_enabled(do_train):
                output = model(task_batch, head_name)
                loss = critic(output.squeeze(), task_batch['labels'].float()) if head_props[
                    'loss_func'] else output

                step_loss += loss * head_props['loss_weight']
                # Accumulate the head loss for the current batch
                head_losses[head_name] += loss.item()

                # evaluate 
                if not do_train:
                    head_evals[head_name] += compute_metrics(output, task_batch['labels'], head_props['eval_metric'],
                                                             head_props['eval_type'])

        epoch_loss += step_loss.item()

        if do_train:
            optim.zero_grad()
            step_loss.backward()
            optim.step()
            scheduler.step()

    # Normalize head and total losses by the number of batches to get the average loss per epoch
    num_batches = len(data_loaders[0])
    head_losses = {head_name: loss / num_batches for head_name, loss in head_losses.items()}
    head_evals = {head_name: loss / num_batches for head_name, loss in head_evals.items()}
    epoch_loss /= num_batches

    return epoch_loss, head_losses, head_evals


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


def get_datloader(dataset, tokenizer, batch_size: int, shuffle: bool):
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


def parse_args():
    parser = argparse.ArgumentParser(description="Script to train your model")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"],
                        help="Device to run training on")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--betas", nargs=2, type=float, default=[0.9, 0.999], help="Betas for AdamW optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--save_path", type=str, default="models/weights/multitask", help="Path to save model weights")
    parser.add_argument("--project", type=str, default="train_val_bertmed", help="Wandb project name to use for logs")
    return parser.parse_args()


if __name__ == '__main__':
    train_args = parse_args()
    setattr(train_args, 'optim', torch.optim.AdamW)
    torch.cuda.empty_cache()

    # ----------------------------- Model ------------------------------------------------------------
    model_name = 'bert-base-uncased'
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=config.max_position_embeddings)
    pre_trained_model = AutoModel.from_pretrained(model_name)

    # ----------------------------- Data ------------------------------------------------------------
    torch.manual_seed(train_args.seed)
    train_datasets_size = 2400
    val_dataset_size = 270

    # load data
    task1_data = load_data('glue', 'mrpc', train_samples=train_datasets_size, val_samples=val_dataset_size)
    task2_data = load_data('glue', 'rte', train_samples=train_datasets_size, val_samples=val_dataset_size)
    task3_data = load_data('glue', 'stsb', train_samples=train_datasets_size, val_samples=val_dataset_size)

    # preprocess
    task1_train_dataset = preprocess_dataset(task1_data['train'], tokenizer, preprocess_func)
    task2_train_dataset = preprocess_dataset(task2_data['train'], tokenizer, preprocess_func)
    task3_train_dataset = preprocess_dataset(task3_data['train'], tokenizer, preprocess_func)
    task1_val_dataset = preprocess_dataset(task1_data['val'], tokenizer, preprocess_func)
    task2_val_dataset = preprocess_dataset(task2_data['val'], tokenizer, preprocess_func)
    task3_val_dataset = preprocess_dataset(task3_data['val'], tokenizer, preprocess_func)

    # data loaders
    task1_train_dataloader = get_datloader(task1_train_dataset, tokenizer,
                                           batch_size=train_args.batch_size, shuffle=True)
    task2_train_dataloader = get_datloader(task2_train_dataset, tokenizer,
                                           batch_size=train_args.batch_size, shuffle=True)
    task3_train_dataloader = get_datloader(task3_train_dataset, tokenizer,
                                           batch_size=train_args.batch_size, shuffle=True)
    task1_val_dataloader = get_datloader(task1_val_dataset, tokenizer,
                                         batch_size=train_args.batch_size, shuffle=True)
    task2_val_dataloader = get_datloader(task2_val_dataset, tokenizer,
                                         batch_size=train_args.batch_size, shuffle=True)
    task3_val_dataloader = get_datloader(task3_val_dataset, tokenizer,
                                         batch_size=train_args.batch_size, shuffle=True)

    # ----------------------------- Headers ------------------------------------------------------------
    in_features = config.hidden_size
    task1_head = ClassificationHead(in_features=in_features, out_features=1)
    task2_head = ClassificationHead(in_features=in_features, out_features=1)
    task3_head = ClassificationHead(in_features=in_features, out_features=1)

    classifiers = torch.nn.ModuleDict({
        "task1_head": task1_head,
        "task2_head": task2_head,
        "task3_head": task3_head
    })

    multi_head_model = MultiHeadModel(pre_trained_model, classifiers)
    multi_head_model.to(train_args.device)

    heads_props = {
        "task1_head": {
            "train_loader": task1_train_dataloader,
            "val_loader": task1_val_dataloader,
            "loss_weight": 0.3,
            "loss_func": nn.BCEWithLogitsLoss(),
            "eval_metric": load('glue', 'mrpc'),
            "eval_type": "accuracy"
        },
        "task2_head": {
            "train_loader": task2_train_dataloader,
            "val_loader": task2_val_dataloader,
            "loss_weight": 0.3,
            "loss_func": nn.BCEWithLogitsLoss(),
            "eval_metric": load('glue', 'rte'),
            "eval_type": "accuracy"
        },
        "task3_head": {
            "train_loader": task3_train_dataloader,
            "val_loader": task3_val_dataloader,
            "loss_weight": 0.3,
            "loss_func": nn.MSELoss(),
            "eval_metric": load('mse'),
            "eval_type": "regression"
        }
    }

    run = wandb.init(
        project=train_args.project,
        config=vars(train_args)
    )

    train(multi_head_model, heads_props, train_args)
