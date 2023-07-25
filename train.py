import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from transformers import AutoConfig, AutoTokenizer, AutoModel
from data.acronymDataset import AcronymDataset
from data.RelationExtraction.MedicalNERDataset import MedicalNERDataset
from data.RelationExtraction.MedicalRCDataset import MedicalRCDataset
from models.multiHeadModel import MultiHeadModel
from models.heads import ClassificationHead, NERHead, RelationClassificationHead
import os
import argparse
from datasets import load_from_disk
from torch.optim.lr_scheduler import CosineAnnealingLR


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

    for epoch in tqdm(range(train_args.epochs)):
        epoch_loss = 0.0
        # prepare the model weights for training:
        multi_head_model.train()
        # create the data loaders list
        train_loaders = [head_prop['train_loader'] for head_prop in heads_props.values()]

        # iterate the batches simultaneously
        for i, combined_batch in enumerate(zip(*train_loaders)):
            step_loss = 0.0

            for task_batch, head_name in zip(combined_batch, heads_props.keys()):
                critic = heads_props[head_name]['loss_func']
                task_batch = task_batch.to(train_args.device)

                # loss 
                output = multi_head_model(task_batch, head_name)
                loss = critic(output.squeeze(), task_batch['labels'].float()) if heads_props[head_name][
                    'loss_func'] else output
                step_loss += loss * heads_props[head_name]['loss_weight']

            epoch_loss += step_loss.item()
            optim.zero_grad()
            step_loss.backward()
            optim.step()

            # Step the scheduler after each optimizer step
            scheduler.step()

        epoch_loss /= len(train_loaders[0])
        wandb.log({'loss per epoch': epoch_loss})

        # save the model at each epoch
        if not os.path.exists(train_args.save_path):
            os.mkdir(train_args.save_path)

        torch.save({
            'epoch': epoch,
            'model_state_dict': multi_head_model.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
        }, f'{train_args.save_path}/multi_head_epoch{epoch}.pt')


def parse_args():
    parser = argparse.ArgumentParser(description="Script to train your model")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"],
                        help="Device to run training on")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--betas", nargs=2, type=float, default=[0.9, 0.999], help="Betas for AdamW optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--save_path", type=str, default="models/weights", help="Path to save model weights")
    parser.add_argument("--project", type=str, default="nlp_project_huji", help="Wandb project name to use for logs")
    return parser.parse_args()


if __name__ == '__main__':
    train_args = parse_args()
    setattr(train_args, 'optim', torch.optim.AdamW)
    torch.cuda.empty_cache()

    # ----------------------------- Model ------------------------------------------------------------
    model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=config.max_position_embeddings)
    pre_trained_model = AutoModel.from_pretrained(model_name)

    # ----------------------------- Data ------------------------------------------------------------
    torch.manual_seed(train_args.seed)
    # acronym
    acronym_data_file_path = 'data/acronym_data.txt'
    acronym_dataset = AcronymDataset(file_path=acronym_data_file_path, tokenizer=tokenizer)
    train_loader_for_acronym, val_loader_for_acronym = acronym_dataset.get_dataloaders(train_size=0.9,
                                                                                       batch_size=64)
    # n2c2
    n2c2_dataset_path = 'data/RelationExtraction/n2c2_dataset'
    n2c2_dataset = load_from_disk(n2c2_dataset_path)

    # NER
    medical_ner_dataset = MedicalNERDataset(n2c2_dataset, tokenizer, train_size=0.9, batch_size=8)
    ner_dataloaders = medical_ner_dataset.get_dataloaders()
    ner_train_dataloader = ner_dataloaders["train"]
    ner_val_dataloader = ner_dataloaders["validation"]

    # RC
    medical_rc_dataset = MedicalRCDataset(n2c2_dataset, tokenizer, pre_trained_model, train_size=0.9, batch_size=64)
    rc_dataloaders = medical_rc_dataset.get_dataloaders()
    rc_train_dataloader = rc_dataloaders["train"]
    rc_val_dataloader = rc_dataloaders["validation"]

    # ----------------------------- Headers ------------------------------------------------------------
    in_features = config.hidden_size
    acronym_head = ClassificationHead(in_features=in_features, out_features=1)
    ner_head = NERHead(hidden_size=in_features, num_labels=len(medical_ner_dataset.id_2_label))
    rc_head = RelationClassificationHead(hidden_size=in_features, num_labels=1)

    classifiers = torch.nn.ModuleDict({
        "acronym_head": acronym_head,
        "ner_head": ner_head,
        "rc_head": rc_head
    })

    multi_head_model = MultiHeadModel(pre_trained_model, classifiers)
    multi_head_model.to(train_args.device)

    heads_props = {
        "acronym_head": {
            "train_loader": train_loader_for_acronym,
            "loss_weight": 0.2,
            "loss_func": torch.nn.BCEWithLogitsLoss(),
            "require_batch_for_forward": False
        },
        "ner_head": {
            "train_loader": ner_train_dataloader,
            "loss_weight": 0.4,
            "loss_func": None,
            "require_batch_for_forward": True
        },
        "rc_head": {
            "train_loader": rc_train_dataloader,
            "loss_weight": 0.4,
            "loss_func": torch.nn.BCEWithLogitsLoss(),
            "require_batch_for_forward": True
        }
    }

    run = wandb.init(
        project=train_args.project,
        config=train_args
    )

    train(multi_head_model, heads_props, train_args)
