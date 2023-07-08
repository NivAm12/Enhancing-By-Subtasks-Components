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
            pass
            print(combined_batch[0].input_ids.shape, combined_batch[1].input_ids.shape)

