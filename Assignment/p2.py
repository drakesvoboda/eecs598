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
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

from parameter_server import *

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
    args = parser.parse_args()
    args.world_size = args.num_proc * args.nodes
    print(args)

    # Task 2: Assign IP address and port for master process, i.e. process with rank=0
    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = "29500"

    # Spawns one or many processes untied to the first Python process that runs on the file.
    # This is to get around Python's GIL that prevents parallelism within independent threads.
    mp.spawn(train, nprocs=args.num_proc, args=(args,))

def load_datasets(batch_size, world_size, rank):
  # Task 1: Choose an appropriate directory to download the datasets into
  root_dir = '/media/drake/Passport Ultra/data/'

  extra_dataset = SVHN(root=root_dir, split='extra', download=True, transform=ToTensor())
  train_dataset = SVHN(root=root_dir, split='train', download=True, transform=ToTensor())
  dataset = torch.utils.data.ConcatDataset([train_dataset, extra_dataset])
  val_dataset = SVHN(root=root_dir, split='test', download=True, transform=ToTensor())
  print("Train dataset: {}".format(dataset))
  print("Val dataset: {}".format(val_dataset))
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
        self.fc1 = nn.Linear(1024, 84)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x, y=None):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def create_model():
  input_size = 3072
  num_classes = 10
  
  model = ConvNet(input_size, out_size=num_classes)

  # Printing sizes of model parameters
  for t in model.parameters():
      print(t.shape)

  return model

def eval_step(model, batch):
    images, labels = batch
    out = model(images)
    loss = F.cross_entropy(out, labels)
    acc = accuracy(out, labels)
    return {'val_loss': loss, 'val_acc': acc}

def calc_stats(outputs):
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

def evaluate(model, val_loader):
    outputs = [eval_step(model, batch) for batch in val_loader]
    return calc_stats(outputs)

def epoch_report(epoch, result):
    print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

def run_epochs(epochs, lr, model, train_loader, val_loader, rank, param_server_rref, tau, alpha, opt_func=torch.optim.SGD, do_checkpoint=False):
    history = []
    optimizer = opt_func(model.parameters(), lr, momentum=.9, weight_decay=0.0001)
    idx = 0
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:

            idx += 1

            if (idx + rank) % tau == 0:
                with torch.no_grad():
                    param_diff = remote_method(ParameterServer.sync_params, param_server_rref, model)

                    for (_, diff), (_, param) in zip(param_diff.items(), model.named_parameters()):
                        param.copy_(param - alpha * diff)

            images, labels = batch 
            # Task 1: Complete training loop
            # Step 1 Get model prediction, i.e. run the fwd pass
            outputs = model(images)
            # Step 2 Calculate cross entropy loss between out and labels
            loss = F.cross_entropy(outputs, labels)
            # Step 3 Reset the gradients to zero
            model.zero_grad()
            # Step 4 Run the backward pass
            loss.backward()
            # Step 5 Apply gradients using optimizer
            optimizer.step()

        # Validation phase
        result = evaluate(model, val_loader)
        epoch_report(epoch, result)
        history.append(result)
        if do_checkpoint:
            torch.save({
                'model_sate_dict': model.state_dict()
                }, 'model_{}.pt'.format(rank))
    return history

def train(proc_num, args):
    rank = args.local_rank * args.num_proc + proc_num

    moving_rate = .9 / (args.world_size-1)
    tau = args.world_size-1

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
        train_loader, val_loader = load_datasets(batch_size=args.batch_size, world_size=args.world_size-1, rank=rank-1)

        history = [evaluate(model, val_loader)]
        print("Pre-training accuracy'\n'{}".format(history))
        
        start = time.time()
        # Running the training for a total of 40 epochs, reducing the learning rate
        # progressively to reduce the likelyhood of "ping-pong"-ing over the minima
        history += run_epochs(10,1e-2, model, train_loader, val_loader, rank, param_server_rref, tau, moving_rate, do_checkpoint=args.do_chkpt)
        history += run_epochs(10,1e-3, model, train_loader, val_loader, rank, param_server_rref, tau, moving_rate, do_checkpoint=args.do_chkpt)
        history += run_epochs(20,1e-4, model, train_loader, val_loader, rank, param_server_rref, tau, moving_rate, do_checkpoint=args.do_chkpt)
        end = time.time()
        print(end - start, " seconds to train")

        writer = SummaryWriter(log_dir="runs_{}".format(rank))

        losses = [x['val_loss'] for x in history]
        for i in range(len(losses)):
            writer.add_scalar('Loss/val', losses[i], i)
        
        accuracies = [x['val_acc'] for x in history]
        for i in range(len(accuracies)):
            writer.add_scalar('Accuracy/val', accuracies[i], i)
    
        writer.close()

if __name__ == '__main__':
    main()
