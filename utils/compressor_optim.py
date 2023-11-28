import torch.optim as optim

def optim_add_compressor(net, main_optim, aux_learning_rate = 1e-3):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = set(
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    )
    aux_parameters = set(
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    )

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    main_optim.add_param_group({'params': (params_dict[n] for n in sorted(list(parameters)))})
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(list(aux_parameters))),
        lr=aux_learning_rate,
    )
    return aux_optimizer

