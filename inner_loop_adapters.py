import torch
from collections import OrderedDict


def update_parameters_gd(model, loss, step_size=0.5, first_order=False, log_var=None):
    """Update the parameters of the model, with one step of gradient descent."""
    grads = torch.autograd.grad(loss, model.meta_parameters(),
        create_graph=not first_order, allow_unused=True) # allow_unused is necessary for when not all the output heads are used

    params = OrderedDict()
    for (name, param), grad in zip(model.meta_named_parameters(), grads):
        if grad is None: # the gradients of the output heads that are not used will be None
            continue
        params[name] = param - step_size * grad

    if log_var:
        grad = torch.autograd.grad(loss, log_var, create_graph=not first_order)
        log_var = log_var - step_size * grad[0]

    return params
