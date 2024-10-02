#
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.autograd as autograd
from config import *



class MAML(torch.nn.Module):
    
    def __init__(self, model):
        super(MAML, self).__init__()
        self.model = model

    def _inner_iter(self, x, y, params, inner_args):
        """
        Performs one inner-loop iteration of MAML including the forward and
        backward passes and the parameter update.

        Args:
          x (float tensor, [n_shot, input_dim]): per-episode support set.
          y (int tensor, [n_shot]): per-episode support set labels.
          params (OrderedDict): the model parameters BEFORE the update.
          inner_args (dict): inner-loop optimization hyperparameters.

        Returns:
          updated_params (OrderedDict): the model parameters AFTER the update.
        """
        x.requires_grad_(True)
        logits = self.model(x, params)
        loss = F.binary_cross_entropy_with_logits(logits, y)

        
        grads = autograd.grad(loss, params.values(), create_graph=True)
        updated_params = OrderedDict()
        
        for (name, param), grad in zip(params.items(), grads):
            updated_param = param - inner_args['lr'] * grad
            updated_param.requires_grad_()
            updated_params[name] = updated_param
            
        return updated_params

    def _adapt(self, x, y, params, inner_args):
        """
        Performs inner-loop adaptation in MAML.

        Args:
          x (float tensor, [n_shot, input_dim]): per-episode support set.
          y (int tensor, [n_shot]): per-episode support set labels.
          params (OrderedDict): a dictionary of parameters at meta-initialization.
          inner_args (dict): inner-loop optimization hyperparameters.

        Returns:
          updated_params (OrderedDict): model parameters AFTER inner-loop adaptation.
        """
        for step in range(n_step):
            params = self._inner_iter(x, y, params, inner_args)
        return params


    def forward(self, x_shot, x_query, y_shot, inner_args, meta_args):
        """
        Args:
          x_shot (float tensor, [n_episode, n_shot, input_dim]): support sets.
          x_query (float tensor, [n_episode, n_query, input_dim]): query sets.
          y_shot (int tensor, [n_episode, n_shot]): support set labels.
          inner_args (dict): inner-loop optimization hyperparameters.
          meta_args (dict): meta-update hyperparameters.

        Returns:
          all_logits (float tensor, [n_episode, n_query, output_dim]):
            the logits for the query sets.
        """
        meta_train = meta_args['meta_train']
        params = OrderedDict(self.model.named_parameters())  # Initialize params with model's named parameters
        
        for name in list(params.keys()):
            if not params[name].requires_grad:
                params.pop(name)

        for iter in range(n_iter):
            
    
            params = OrderedDict(self.model.named_parameters())
            all_logits = []
            
            if meta_train:
                updated_params = self._adapt(x_shot, y_shot, params, inner_args)
                logits = self.model(x_query, updated_params)
            else:
                logits = self.model(x_query, params)

            all_logits.append(logits)

        return torch.stack(all_logits, dim=0)

