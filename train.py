import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
from data.acronymDataset import AcronymDataset
from models.multiHeadModel import MultiHeadModel
from models.heads import ClassificationHead


def train(multi_head_model: nn.Module, heads_props: dict, train_args: dict):
    # prepare the model weights for training:    
    multi_head_model.train()
    for model_head in multi_head_model.heads.values():
        model_head.train()

    optim  = train_args['optim']

    for epoch in tqdm(range(train_args["epochs"])):
        # create the data loaders list
        train_loaders = [head_prop['train_loader'] for head_prop in heads_props.values()]

        # iterate the batches simultaneously
        for i, combined_batch in enumerate(zip(*train_loaders)):
            step_loss = 0
            optim.zero_grad()

            for task_batch, head_name in zip(combined_batch, heads_props.keys()):
                critic = heads_props[head_name]['loss_func']
                task_batch = task_batch.to(train_args['device'])
                
                # loss 
                output = multi_head_model(task_batch, head_name)
                loss = critic(output.squeeze(), task_batch['labels'].float())
                step_loss += loss * heads_props[head_name]['loss_weight']

            step_loss.backward()
            optim.step()



if __name__ == '__main__':
    device = 'mps'
    model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=config.max_position_embeddings)
    pre_trained_model = AutoModel.from_pretrained(model_name).to(device)

    torch.manual_seed(5)
    file_path = 'data/acronym_data.txt'
    dataset = AcronymDataset(file_path=file_path, tokenizer=tokenizer)
    data = dataset.data
    dataset.preprocss_dataset()

    in_features = config.hidden_size
    binari_head = ClassificationHead(in_features=in_features, out_features=1).to('mps')
    four_labels_head = ClassificationHead(in_features=in_features, out_features=4).to('mps')

    classifiers = torch.nn.ModuleDict({
        "binari_head": binari_head,
        # "four_labels_head": four_labels_head
    })

    multi_head_model = MultiHeadModel(pre_trained_model, classifiers)

    train_loader_for_acronym, _ = dataset.get_dataloaders(train_size=0.9, batch_size=8)
    # train_loader2, _ = dataset.get_dataloaders(train_size=0.9, batch_size=32)

    optim = torch.optim.AdamW(multi_head_model.parameters(), lr=0.001)

    train_args = {
        "epochs": 2,
        "device": "mps",
        "optim": optim
    }

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

    train(multi_head_model, heads_props, train_args)




