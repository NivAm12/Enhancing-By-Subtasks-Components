import torch
import torch.nn as nn
from tqdm import tqdm


def train(multi_head_model: nn.Module, heads_props: dict, train_args: dict):
    # prepare the model weights:
    multi_head_model.train()
    for model_head in multi_head_model.heads.values():
        model_head.train()

    for epoch in tqdm(range(train_args["epochs"])):
        # create the data loaders list, and iterate the batchs simultaneously
        head_prop = [head_prop['train_loader'] for head_prop in heads_props.values()]

        for i, combined_batch in enumerate(zip(*head_prop)):
            loss = None

            for task_batch, head_name in zip(combined_batch, heads_props.keys()):
                output = multi_head_model(task_batch, head_name)
                
