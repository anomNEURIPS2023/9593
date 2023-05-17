import numpy as np
from typing import List, Sequence, Tuple
import math
from collections import deque
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import types
from collections import OrderedDict


class class_pathways(object):
    def __init__(self, keep_ratio, history_len=10, num_tasks=1):
        self.all_scores = None
        self.keep_ratio = keep_ratio
        self.record_score = collections.defaultdict(dict)
        self.last_mask = collections.defaultdict(dict)
        for task in range(num_tasks):
            self.record_score['actor'][f'task{task}'] = deque(maxlen=history_len)
            self.record_score['critic'][f'task{task}'] = deque(maxlen=history_len)
            self.last_mask['actor'][f'task{task}'] = []
            self.last_mask['critic'][f'task{task}'] = []

        self.prune_param_score = 0

    # Monkey-patch the Linear and Conv2d layer to learn the multiplicative mask
    # instead of the weights
    def monkey_patch(self, net):
        for layer in net.modules():  # for name, layer in net.named_modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
                layer.weight.requires_grad = False

            # Override the forward methods:
            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(mod_forward_conv2d, layer)

            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(mod_forward_linear, layer)

    def get_keep_masks(self, net, type, task):
        grads_abs = []
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                grads_abs.append(torch.abs(layer.weight_mask.grad))

        # Gather all scores in a single vector and normalise
        all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
        norm_factor = torch.sum(all_scores)
        all_scores.div_(norm_factor)


        self.record_score[type][f'task{task}'].append(all_scores.detach().cpu().numpy())
        all_scores = torch.tensor(np.mean(np.stack(list(self.record_score[type][f'task{task}']),0),0)).to(grads_abs[0].device)



        num_params_to_keep = int(len(all_scores) * self.keep_ratio)
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]

        # to avoid faulty threhold
        if acceptable_score == 0:
            return self.last_mask[type][f'task{task}']

        self.prune_param_score = torch.sum(all_scores[all_scores < acceptable_score]).detach().cpu().numpy()

        keep_masks = []
        for g in grads_abs:
            keep_masks.append(((g / norm_factor) >= acceptable_score).float())

        print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks])))

        self.last_mask[type][f'task{task}'] = keep_masks

        return keep_masks


    def get_masks(self, net, replay_buffer, task, itr=1):
        """NOTE: do for all required networks
        """
        self.monkey_patch(net.actor)
        self.monkey_patch(net.critic)
        self.monkey_patch(net.critic_target)
        # make these variables
        net.update_init_configure(replay_buffer, itr, task)
        return (self.get_keep_masks(net.actor, type='actor', task=task)), (self.get_keep_masks(net.critic, type='critic', task=task))

class EnsembleLinearLayer(nn.Module):
    """Efficient linear layer for ensemble models."""

    # TODO: important - The behavior depends on the dimensionality of the tensors
    # https://pytorch.org/docs/stable/generated/torch.matmul.html

    def __init__(
        self, num_members: int, in_size: int, out_size: int, bias: bool = True
    ):
        super().__init__()
        self.num_members = num_members
        self.in_size = in_size
        self.out_size = out_size
        self.weight = nn.Parameter(
            torch.empty(self.num_members, self.in_size, self.out_size)
        )
        self.weight_mask = torch.ones(self.num_members, self.in_size, self.out_size).to(device='cuda')
        self.weight_mask.requires_grad = False # important, otherwise value will be changed during backprop
        if bias:
            self.bias = nn.Parameter(torch.empty(self.num_members, 1, self.out_size))
            self.use_bias = True
        else:
            self.use_bias = False

        self.elite_models: List[int] = None
        self.use_only_elite = False
        self.reset_parameters()

    def forward(self, x):
        if self.use_only_elite:
            xw = x.matmul(self.weight[self.elite_models, ...]*self.weight_mask[self.elite_models, ...])
            if self.use_bias:
                return xw + self.bias[self.elite_models, ...]
            else:
                return xw
        else:
            xw = x.matmul(self.weight*self.weight_mask) # torch.sum(self.weight_mask)/(self.weight_mask.shape[0]*self.weight_mask.shape[1]*self.weight_mask.shape[2])
            if self.use_bias:
                return xw + self.bias
            else:
                return xw

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self) -> str:
        return (
            f"num_members={self.num_members}, in_size={self.in_size}, "
            f"out_size={self.out_size}, bias={self.use_bias}"
        )

    def set_elite(self, elite_models: Sequence[int]):
        self.elite_models = list(elite_models)

    def toggle_use_only_elite(self):
        self.use_only_elite = not self.use_only_elite





def load_keep_masks(path):
    keep_masks = np.load(f'{path}/keep_masks.npy', allow_pickle=True).item()
    return keep_masks
def unhook(model):
    prunable_layers = filter(lambda layer: isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear),
                             model.modules())
    for layer in prunable_layers:
        layer.weight._backward_hooks = OrderedDict()
def mod_forward_conv2d(self, x):
    return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                    self.stride, self.padding, self.dilation, self.groups)
def mod_forward_linear(self, x):
    return F.linear(x, self.weight * self.weight_mask, self.bias)
def monkey_patch(model, mask_layers):
    prunable_layers = filter(lambda layer: isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear),
                             model.modules())
    for layer, mask in zip(prunable_layers, mask_layers):
        layer.weight_mask = mask
        # Override the forward methods:
        if isinstance(layer, nn.Conv2d):
            layer.forward = types.MethodType(mod_forward_conv2d, layer)
        if isinstance(layer, nn.Linear):
            layer.forward = types.MethodType(mod_forward_linear, layer)
def init_mask(model):
    mask = []
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            mask.append(torch.ones_like(layer))
    return torch.tensor(mask)
def common_weight(mask1, mask2):
    """ inputs are masks"""
    M = 0
    for m1, m2 in zip(mask1, mask2):
        M += (m1 * m2).sum().item()
    return M
def get_abs_sps(model):
    nonzero = total = 0
    for name, param in model.named_parameters():
        tensor = param.data
        nz_count = torch.count_nonzero(tensor)
        total_params = tensor.numel()
        nonzero += nz_count
        total += total_params
    abs_sps = 100 * (total - nonzero) / total
    return abs_sps, total, nonzero
def get_abs_sps_each_layer(model):
    total = []
    nonzero = []
    abs_sps = []

    for m in model.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            t = m.weight.numel()
            n = torch.count_nonzero(m._parameters['weight']).item()  # torch.count_nonzero(m._parameters['weight']) ; torch.nonzero(m.weight).shape[0]
            total.append(t)
            nonzero.append(n)
            abs_sps.append(round( 100 * ((t - n) / t), 2))
    return abs_sps, total, nonzero
def sparse_weights(model):
    res = OrderedDict()
    for name, param in model.named_parameters():
        res[name] = param.to_sparse()
    return res
def apply_prune_mask(net, keep_masks, fixed_weight=0.):

    # Before I can zip() layers and pruning masks I need to make sure they match
    # one-to-one by removing all the irrelevant modules:
    prunable_layers = filter(
        lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
            layer, nn.Linear), net.modules())

    for layer, keep_mask in zip(prunable_layers, keep_masks):
        assert (layer.weight.shape == keep_mask.shape)

        def hook_factory(keep_mask):
            """
            The hook function can't be defined directly here because of Python's
            late binding which would result in all hooks getting the very last
            mask! Getting it through another function forces early binding.
            """

            def hook(grads):
                return grads * keep_mask

            return hook

        # mask[i] == 0 --> Prune parameter
        # mask[i] == 1 --> Keep parameter

        # Step 1: Set the masked weights to zero (NB the biases are ignored)
        # Step 2: Make sure their gradients remain zero
        if fixed_weight==-1:
            pass
        else:
            layer.weight.data[keep_mask == 0.] = 0.
        layer.weight.register_hook(hook_factory(keep_mask)) # register hook is backward hook
# for multiple pruning
def apply_prune_multiple_mask(net, keep_masks, fixed_weight=0.):

    # Before I can zip() layers and pruning masks I need to make sure they match
    # one-to-one by removing all the irrelevant modules:
    prunable_layers = filter(
        lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
            layer, nn.Linear), net.modules())


    for layer, keep_mask, keep_mask_2 in zip(prunable_layers, keep_masks['task0'], keep_masks['task1']):
        assert (layer.weight.shape == keep_mask.shape)
        assert (layer.weight.shape == keep_mask_2.shape)

        def hook_factory(keep_mask, keep_mask_2, trigger):
            """
            The hook function can't be defined directly here because of Python's
            late binding which would result in all hooks getting the very last
            mask! Getting it through another function forces early binding.
            """

            def hook(grads):
                if trigger == 0:
                    #print('triggered 0')
                    return grads * keep_mask
                elif trigger == 1:
                    #print('triggered 1')
                    return grads * keep_mask_2
            return hook

        # mask[i] == 0 --> Prune parameter
        # mask[i] == 1 --> Keep parameter

        # Step 1: Set the masked weights to zero (NB the biases are ignored)
        # Step 2: Make sure their gradients remain zero
        if fixed_weight==-1:
            pass
        else:
            layer.weight.data[keep_mask == 0.] = 0. # TODO
        layer.weight.register_hook(hook_factory(keep_mask, keep_mask_2, trigger=net.trigger_mask)) # register hook is backward hook


def sync_weights(main_net, copy_net):
    "sync weights of the main_net to copy_net"
    # Before I can zip() layers and pruning masks I need to make sure they match
    # one-to-one by removing all the irrelevant modules:
    main_layers = filter(lambda layer: isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear), main_net.modules())
    copy_layers = filter(lambda layer: isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear),copy_net.modules())

    for m_layer, c_layer in zip(main_layers, copy_layers):
        assert (m_layer.weight.shape == c_layer.weight.shape)
        c_layer.weight.data = copy.deepcopy(m_layer.weight.data)


