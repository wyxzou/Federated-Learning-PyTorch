#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import pdb
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torch.utils.data.distributed
from torch.optim import Optimizer



# https://discuss.pytorch.org/t/a-problem-about-optimizer-param-groups-in-step-function/14463
class sparsetopSGD(Optimizer):
    def __init__(self, params, lr=0.1, topk=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, required=True):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, topk=topk, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        
        super(sparsetopSGD, self).__init__(params, defaults)

        
        gradient_size = 0
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['memory'] = torch.zeros_like(p.data)
                gradient_size += torch.numel(p.data)

        self.gradient_after_topk = torch.zeros(gradient_size)
        self.gradient_before_topk = torch.zeros(gradient_size)
        self.gradient_without_error = torch.zeros(gradient_size)

    def __setstate__(self, state):
        super(sparsetopSGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        #print("Length of param_groups: ", len(self.param_groups))

        if (len(self.param_groups) > 1):
            raise ValueError("TopK sparsification not available for more than one parameter group")

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            lr = group['lr']
            topk = group['topk']

            # lists used in the actual calculation of gradients
            param_size = []
            gradient_shape = []
            gradients = []


            gradients_without_error = []

            # pdb.set_trace()
            # print("parameter keys")
            # print(group['params'])

            for p in group['params']:   

                param_state = self.state[p]
                if p.grad is None:
                    continue

                d_p = p.grad
                corrected_gradient = group['lr'] * d_p

                # save for 
                gradients_without_error.append(torch.flatten(corrected_gradient.detach().clone()))


                corrected_gradient = param_state['memory'] + corrected_gradient

                # save gradient shape
                gradient_shape.append(corrected_gradient.shape)

                corrected_gradient = torch.flatten(corrected_gradient)


                # abs_corrected_gradient = abs(corrected_gradient)

                # d_p = p.grad.detach.clone()
                # d_p = torch.flatten(d_p)

                param_size.append(corrected_gradient.size(dim=0))
                gradients.append(corrected_gradient)

            if len(gradients) > 0:
                self.gradient_without_error = torch.cat(gradients_without_error, dim=0)

                all_gradients = torch.cat(gradients, dim=0)

                # self.raw_gradient = torch.cat(gradients_error_not_adjusted, dim=0)
                self.gradient_before_topk = all_gradients.detach().clone()

                abs_all_gradients = abs(all_gradients)

                _, indices = torch.topk(abs_all_gradients, int( (1 - topk) * all_gradients.shape[0]), dim=0, largest=False)

                all_gradients[indices] = 0

                self.gradient_after_topk = all_gradients.detach().clone()

                sparsified_gradients = torch.split(all_gradients, param_size)


            i = 0
            for p in group['params']:
                param_state = self.state[p]
                if p.grad is None:
                    continue

                d_p = p.grad
                corrected_gradient = group['lr'] * d_p
                corrected_gradient = param_state['memory'] + corrected_gradient

                sparsified_gradient = torch.reshape(sparsified_gradients[i], gradient_shape[i])
                param_state['memory'] = corrected_gradient - sparsified_gradient
                p.data.add_(sparsified_gradient, alpha=-1)
                i += 1

            # for p in group['params']:
            #     param_state = self.state[p]
            #     if p.grad is None:
            #         continue

            #     # self.sparsified_gradient[p] = 

            #     d_p = p.grad
            #     corrected_gradient = group['lr'] * d_p
            #     corrected_gradient = param_state['memory'] + corrected_gradient
            #     # initial = torch.clone(corrected_gradient).detach()

            #     dpsize = corrected_gradient.shape
                
            #     corrected_gradient = torch.flatten(corrected_gradient)

            #     abs_corrected_gradient = abs(corrected_gradient)
            #     _, indices = torch.topk(abs_corrected_gradient, int( (1 - topk) * d_p.shape[0]), dim=0, largest=False)
                
            #     update = torch.clone(corrected_gradient).detach()
                
            #     update[indices] = 0

            #     corrected_gradient = torch.reshape(corrected_gradient, dpsize)

            #     # if not torch.allclose(corrected_gradient, initial):
            #     #    print("Tensor not reshaped properly")

            #     update = torch.reshape(update, dpsize)
            #     param_state['memory'] = corrected_gradient - update

            #     p.data.add_(corrected_gradient, alpha=-1)

        return loss