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

# --------- Helper Methods --------------------

# On the local node, call a method with first arg as the value held by the
# RRef. Other args are passed in as arguments to the function called.
# Useful for calling instance methods. method could be any matching function, including
# class methods.
def call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)

# Given an RRef, return the result of calling the passed in method on the value
# held by the RRef. This call is done on the remote node that owns
# the RRef and passes along the given argument.
# Example: If the value held by the RRef is of type Foo, then
# remote_method(Foo.bar, rref, arg1, arg2) is equivalent to calling
# <foo_instance>.bar(arg1, arg2) on the remote node and getting the result
# back.

def remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)

# --------- Parameter Server --------------------
class ParameterServer():
    def __init__(self, model, alpha):
        super().__init__()
        self.global_model = model
        self.input_device = "cpu"
        self.alpha = alpha

    def get_model(self):
        return self.global_model

    def sync_params(self, model):      
        with torch.no_grad():
            ret = { }

            for (key, global_param), (_, param) in zip(self.global_model.named_parameters(), model.named_parameters()):
                diff = param - global_param
                ret[key] = diff
                global_param.copy_(global_param + self.alpha * diff)

            return ret

# The global parameter server instance.
param_server = None
# A lock to ensure we only have one parameter server.
global_lock = Lock()

def get_parameter_server(model=None, alpha=None):
    """
    Returns a singleton parameter server to all trainer processes
    """
    global param_server
    # Ensure that we get only one handle to the ParameterServer.
    with global_lock:
        if not param_server:
            # construct it once
            param_server = ParameterServer(model, alpha)
        return param_server

def run_parameter_server(rank, world_size):
    # The parameter server just acts as a host for the model and responds to
    # requests from trainers.
    # rpc.shutdown() will wait for all workers to complete by default, which
    # in this case means that the parameter server will wait for all trainers
    # to complete, and then exit.
    print("PS master initializing RPC")
    rpc.init_rpc(name="parameter_server", rank=rank, world_size=world_size)
    print("RPC initialized! Running parameter server...")
    rpc.shutdown()
    print("RPC shutdown on parameter server.")