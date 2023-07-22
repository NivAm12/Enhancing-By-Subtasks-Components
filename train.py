import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
from data.acronymDataset import AcronymDataset


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
                print(loss)


if __name__ == '__main__':
    device = 'mps'
    model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=config.max_position_embeddings)
    pre_trained_model = AutoModel.from_pretrained(model_name).to(device)



