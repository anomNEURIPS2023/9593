import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import types

from collections import OrderedDict

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



def mod_forward_conv2d(self, x):
        return F.conv2d(x, self.weight * self.weight_mask, self.bias,
                        self.stride, self.padding, self.dilation, self.groups)
def mod_forward_linear(self, x):
        return F.linear(x, self.weight * self.weight_mask, self.bias)


def configure_pathway(net, keep_ratio, replay_buffer, iterations=1, batch_size=100, vae=True, value_func=False):
    net = copy.deepcopy(net)
    def monkey_patch(net):
        for layer in net.modules(): # for name, layer in net.named_modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
                #nn.init.xavier_normal_(layer.weight)
                layer.weight.requires_grad = False
            # Override the forward methods:
            if isinstance(layer, nn.Conv2d):
                layer.forward = types.MethodType(mod_forward_conv2d, layer)
            if isinstance(layer, nn.Linear):
                layer.forward = types.MethodType(mod_forward_linear, layer)
    monkey_patch(net.actor)
    monkey_patch(net.critic)
    if vae:
        monkey_patch(net.vae)
    elif value_func:
        monkey_patch(net.value)
    else:
        print('no vae or value network is activated')
    net.train(replay_buffer, iterations, batch_size, using_snip=True)

    def get_keep_masks(net):
        grads_abs = []
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                grads_abs.append(torch.abs(layer.weight_mask.grad))
        # Gather all scores in a single vector and normalise
        all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
        norm_factor = torch.sum(all_scores)
        all_scores.div_(norm_factor)
        num_params_to_keep = int(len(all_scores) * keep_ratio)
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]
        keep_masks = []
        for g in grads_abs:
            keep_masks.append(((g / norm_factor) >= acceptable_score).float())
        print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks])))

        return keep_masks
    if vae:
        return (get_keep_masks(net.actor)), (get_keep_masks(net.critic)), (get_keep_masks(net.vae))
    elif value_func:
        return (get_keep_masks(net.actor)), (get_keep_masks(net.critic)),  (get_keep_masks(net.value))
    else:
        return (get_keep_masks(net.actor)), (get_keep_masks(net.critic))


def apply_mask(net, keep_masks, fixed_weight=0.):
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
