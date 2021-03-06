import argparse
import os
import time
from threading import Lock

import torch
import torchvision
import torch.multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.distributed.rpc as rpc

from torchvision.datasets import SVHN
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

from parameter_server import *
from boilerplate import *

# For deterministic runs
torch.manual_seed(0)

def get_exec_device():
    return torch.device('cpu')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N', help='number of nodes (default: 1)')
    parser.add_argument('-np', '--num_proc', default=1, type=int, help='number of procs per node')
    parser.add_argument('-nr', '--local_rank', default=0, type=int, help='ranking within the nodes')
    parser.add_argument('--batch_size', default=64, type=int, metavar='N', help='Batch size')
    parser.add_argument('--do_chkpt', default=False, action='store_true', help='Enable checkpointing')
    parser.add_argument('-a', '--address', default="localhost")
    parser.add_argument('-p', '--port', default="9955")
    parser.add_argument('-d', '--data_dir', default="../../data/")
    args = parser.parse_args()
    args.world_size = args.num_proc * args.nodes
    print(args)

    # Task 2: Assign IP address and port for master process, i.e. process with rank=0
    os.environ['MASTER_ADDR'] = args.address
    os.environ['MASTER_PORT'] = args.port

    # Spawns one or many processes untied to the first Python process that runs on the file.
    # This is to get around Python's GIL that prevents parallelism within independent threads.
    mp.spawn(train, nprocs=args.num_proc, args=(args,))

def load_datasets(batch_size, world_size, rank, data_dir):
    # Task 1: Choose an appropriate directory to download the datasets into
    root_dir = data_dir

    transform = transforms.Compose([
        transforms.CenterCrop([30, 30]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    extra_dataset = SVHN(root=root_dir, split='extra', download=True, transform=transform)
    train_dataset = SVHN(root=root_dir, split='train', download=True, transform=transform)
    dataset = torch.utils.data.ConcatDataset([train_dataset, extra_dataset])
    val_dataset = SVHN(root=root_dir, split='test', download=True, transform=transform)
    # print("Train dataset: {}".format(dataset))
    # print("Val dataset: {}".format(val_dataset))
    val_ds = val_dataset

    # Task 2: modify train loader to work with multiple processes
    # 1. Generate a DistributedSample instance with num_replicas = world_size
    #    and rank=rank.
    # 2. Set train_loader's sampler to the distributed sampler

    sampler = torch.utils.data.distributed.DistributedSampler(dataset=dataset, num_replicas=world_size, rank=rank)

    train_loader = torch.utils.data.DataLoader(dataset=dataset,   
                                                batch_size=batch_size,
                                                sampler=sampler,
                                                num_workers=0,
                                                pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader
  
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class DeepModel(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        # 6 hidden layers
        self.linear1 = nn.Linear(in_size, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 256)
        self.linear4 = nn.Linear(256, 128)
        self.linear5 = nn.Linear(128, 64)
        self.linear6 = nn.Linear(64, 32)
        # output layer
        self.linear7 = nn.Linear(32, out_size)
        
    def forward(self, xb):
        # Flatten the image tensors
        xb = xb.view(xb.size(0), -1)
        # Get intermediate outputs using hidden layer
        out = self.linear1(xb)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        out = F.relu(out)
        out = self.linear4(out)
        out = F.relu(out)
        out = self.linear5(out)
        out = F.relu(out)
        out = self.linear6(out)
        out = F.relu(out)
        # Get predictions using output layer
        out = self.linear7(out)
        return out

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=2)
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x, y=None):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class EASGDTrainer(Trainer):
    def __init__(self, model: torch.nn.Module, criterion: callable, optimizer: torch.optim.Optimizer, param_server_rref, rank: int, alpha: float, tau: int):
        super().__init__(model, criterion, optimizer)
        self.param_server_rref = param_server_rref
        self.rank = rank
        self.alpha = alpha
        self.tau = tau

    def step(self, *item):
        if (self.schedule.iteration + self.rank) % self.tau == 0:
            with torch.no_grad():
                param_diff = remote_method(ParameterServer.sync_params, self.param_server_rref, self.model)

                for (_, diff), (_, param) in zip(param_diff.items(), self.model.named_parameters()):
                    param.copy_(param - self.alpha * diff) 

        return super().step(*item)

def train(proc_num, args):
    rank = args.local_rank * args.num_proc + proc_num

    num_trainers = args.world_size-1

    moving_rate = .9 / num_trainers
    tau = 5

    # torch.distributed.init_process_group(backend='gloo', world_size=args.world_size, rank=rank, init_method='env://')

    if rank == 0:
        run_parameter_server(rank, args.world_size)
    else:
        # Task 2: Initialize distributed process group with following parameters,
        #  backend = 'gloo'
        #  init_method = 'env://'
        #  world_size = args.world_size
        #  rank = rank    

        rpc.init_rpc(name=f"trainer_{rank}", rank=rank, world_size=args.world_size)

        param_server_rref = rpc.remote("parameter_server", get_parameter_server, args=(ConvNet(), moving_rate))
        model = remote_method(ParameterServer.get_model, param_server_rref)
        train_loader, val_loader = load_datasets(batch_size=args.batch_size, world_size=num_trainers, rank=rank-1, args.data_dir)
        optimizer = torch.optim.SGD(model.parameters(), 1e-2, momentum=.9, weight_decay=0.0001)

        num_epochs = 1
        total_steps = len(train_loader) * num_epochs

        callbacks = [
            LogRank(rank),
            TrainingLossLogger(),
            TrainingAccuracyLogger(accuracy),
            Validator(val_loader, accuracy, rank=rank-1),
            TorchOnBatchLRScheduleCallback(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=total_steps),
            Timer(),
            Logger()
        ]

        trainer = EASGDTrainer(model, F.cross_entropy, optimizer, param_server_rref, rank, moving_rate, tau)
        schedule = TrainingSchedule(train_loader, num_epochs, callbacks, rank=rank-1)  
        
        start = time.time()

        trainer.train(schedule)

        end = time.time()
        print(end - start, " seconds to train")

        rpc.shutdown()

if __name__ == '__main__':
    main()
