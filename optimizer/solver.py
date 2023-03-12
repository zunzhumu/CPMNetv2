import torch
import numpy as np

def build_optimizer(model):
    if(hasattr(model,"module")):
        model=model.module

    params=model.parameters()
    optimizer = torch.optim.SGD(params,lr=0.001,momentum=0.9,weight_decay=0.0001)
    return optimizer

class scheduler():
    def __init__(self, optimizer):
        self.init_lr=[group["lr"] for group in optimizer.param_groups]
        self.optimizer=optimizer
        self.warmup_factor=0.001
        self.num_warmup_iters = 500
        self.lr_decay_factor = 0.1
        self.lr_decay_time = [80, 120]
        self.end_warm_up = False

    def linear_warmup(self,epoch,iteration):
        if(epoch>1) or self.end_warm_up:
            return
        new_lr = [self.compute_lr_by_iter(lr,iteration) for lr in self.init_lr]
        self.set_lr(new_lr)

    def lr_decay(self, epoch):
        new_lr = [self.compute_lr_by_epoch(lr,epoch) for lr in self.init_lr]
        self.set_lr(new_lr)

    def compute_lr_by_iter(self,lr,iteration):
        if(iteration == self.num_warmup_iters):
            self.end_warm_up = True
        return (1-(1-iteration/self.num_warmup_iters)*(1-self.warmup_factor))*lr

    def compute_lr_by_epoch(self, lr, epoch):
        lr_decay_time=np.array(self.lr_decay_time,dtype=np.int)
        index=np.nonzero(lr_decay_time<=epoch)[0]
        if(index.size==0):
            return lr
        num=index[-1].item()+1
        return lr*(self.lr_decay_factor**num)

    def set_lr(self, lrs):
        for params_group,new_lr in zip(self.optimizer.param_groups,lrs):
            params_group["lr"] = new_lr