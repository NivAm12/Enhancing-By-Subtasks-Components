import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from transformers import AutoConfig, AutoTokenizer, AutoModel
from data.acronymDataset import AcronymDataset
from models.multiHeadModel import MultiHeadModel
from models.heads import ClassificationHead
import os
import argparse



def train(multi_head_model: nn.Module, heads_props: dict, train_args: dict):
    """
    Trains a multi-head model.

    Args:
        multi_head_model (nn.Module): The multi-head model to train.
        heads_props (dict): A dictionary of head properties, where each key is the head name and each value is a dictionary of head properties.
        train_args (dict): A dictionary of training arguments.

    Returns:
        None.
    """

    # prepare the model weights for training:    
    multi_head_model.train()

    optim = train_args.optim(multi_head_model.parameters(), lr=train_args.lr, betas=train_args.betas,
                                weight_decay=train_args["weight_decay"])

    for epoch in tqdm(range(train_args.epochs)):
        epoch_loss = 0.0

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
                loss = critic(output.squeeze(), task_batch['labels'].float())
                step_loss += loss * heads_props[head_name]['loss_weight']

                if i % 20 == 0:
                    wandb.log({f'{head_name}_loss': loss})

            epoch_loss += step_loss.item()
            optim.zero_grad()
            step_loss.backward()
            optim.step()

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
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"], help="Device to run training on")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--betas", nargs=2, type=float, default=[0.9, 0.999], help="Betas for AdamW optimizer")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay for optimizer")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--save_path", type=str, default="models/weights", help="Path to save model weights")
    return parser.parse_args()


if __name__ == '__main__':

    train_args = parse_args()
    setattr(train_args, 'optim', torch.optim.AdamW)
    
    # model
    model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=config.max_position_embeddings)
    pre_trained_model = AutoModel.from_pretrained(model_name)

    # data
    torch.manual_seed(0)
    file_path = 'data/acronym_data.txt'
    dataset = AcronymDataset(file_path=file_path, tokenizer=tokenizer)
    data = dataset.data
    dataset.preprocss_dataset()

    train_loader_for_acronym, val_loader_for_acronym = dataset.get_dataloaders(train_size=0.6,
                                                                               batch_size=train_args["batch_size"])
    # train_loader2, _ = dataset.get_dataloaders(train_size=0.9, batch_size=32)

    in_features = config.hidden_size
    binari_head = ClassificationHead(in_features=in_features, out_features=1)
    # four_labels_head = ClassificationHead(in_features=in_features, out_features=4)

    classifiers = torch.nn.ModuleDict({
        "binari_head": binari_head,
        # "four_labels_head": four_labels_head
    })

    multi_head_model = MultiHeadModel(pre_trained_model, classifiers)
    multi_head_model.to(train_args['device'])

    heads_props = {
        "binari_head": {
            "train_loader": train_loader_for_acronym,
            "loss_weight": 1.0,
            "loss_func": torch.nn.BCEWithLogitsLoss()
        },
        # "four_labels_head": {
        #     "train_loader": train_loader2,
        #     "loss_weight": 0.8

        # }
    }

    run = wandb.init(
        project="test_nlp",
        config=train_args
    )

    train(multi_head_model, heads_props, train_args)
