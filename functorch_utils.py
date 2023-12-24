"""
Helper function for our geodesic systems using functorch.
"""

import torch
from functorch import grad, jvp, make_functional, vjp, make_functional_with_buffers, hessian, jacfwd, jacrev, vmap


def get_params_structure(vector, true_params):
    # method that should take a vector and shape it into a structure similar to the original
    # params
    list_new_weights = []
    pointer = 0
    for sub_weights in true_params:
        # num param
        num_param = sub_weights.numel()
        # get data
        my_param = vector[pointer : (pointer + num_param)].view_as(sub_weights)
        list_new_weights.append(my_param)
        pointer += num_param

    return tuple(list_new_weights)


def stack_gradient(grad, n_params, n_examples):
    flatten_grad_per_example = torch.zeros((n_examples, n_params))
    idx = 0
    for a in grad:
        for i in range(n_examples):
            _g = a[i, :]
            _flat_g = _g.flatten()
            flatten_grad_per_example[i, idx : (idx + len(_flat_g))] = _flat_g
        idx += len(_flat_g)
    # compute the sum along the batch/n_example dimension
    return flatten_grad_per_example.sum(0)


def stack_gradient2(grad, n_params):
    grad_flat = torch.zeros(n_params)
    idx = 0
    for a in grad:
        b = a.sum(0)
        b_flat = b.flatten()
        grad_flat[idx : (idx + len(b_flat))] = b_flat
        idx += len(b_flat)

    # compute the sum along the batch/n_example dimension
    return grad_flat


def custum_hvp(f, primals, tangents, strict=False):
    return jvp(grad(f), primals, tangents, strict=strict)


def stack_gradient3(grad, n_params):
    grad_flat = torch.zeros(n_params)
    idx = 0
    for a in grad:
        b_flat = a.flatten()
        grad_flat[idx : (idx + len(b_flat))] = b_flat
        idx += len(b_flat)
    return grad_flat
