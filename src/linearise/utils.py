import torch
import os
import numpy as np
from copy import deepcopy
from torch import linalg as LA
from torch_utils import parameters_to_vector, vector_to_parameters

def apply_model_on_data(net_input, model, cfg):
    test_scaling = cfg.get('implicit_scaling_except_for_test_data')
    if test_scaling is not None and test_scaling != 1.:
        if not cfg.recon_from_randn:
            net_input = test_scaling * net_input
        output = model(net_input)
        output = output / test_scaling
    else:
        output = model(net_input)

    return output

def compute_jacobian_single_batch(input, model, out_dim, cfg, return_on_cpu=False):
    jac = []
    model.eval()
    f = apply_model_on_data(input, model, cfg.mdl).view(-1)
    for o in range(out_dim):
        f_o = f[o]
        model.zero_grad()
        f_o.backward(retain_graph=True)
        jacs_o = agregate_flatten_weight_grad(model, cfg.spct.skip_layers).detach()
        jac.append(jacs_o)
    return torch.stack(jac, dim=0) if not return_on_cpu else torch.stack(jac, dim=0).cpu()

def agregate_flatten_weight_grad(model, skip_layers):
    grads_o = []
    for name, params in model.named_parameters():
        if name not in skip_layers:
            grads_o.append(params.grad.flatten())
    return torch.cat(grads_o)

def apply_perturbed_model(input, model, omega, skip_layers, cfg):
    model.eval()
    with torch.no_grad(): 
        params = model.named_parameters()
        params_vec = parameters_to_vector(params, skip_layers)
        pert_params_vec = params_vec + omega
        vector_to_parameters(pert_params_vec, model.named_parameters(), skip_layers)
        out = apply_model_on_data(input, model, cfg)
    vector_to_parameters(params_vec, model.named_parameters(), skip_layers)
    return out 

def set_eps_andrei(params, omega):
    eps = np.sqrt(torch.finfo(params.dtype).eps)
    return eps * (1 + LA.norm(params, ord=float('inf'))) \
        / LA.norm(omega, ord=float('inf'))

def get_eps(params, omega, mode='andrei'):
    if mode == 'preset':
        return 1e-6
    elif mode == 'andrei':
        return set_eps_andrei(params, omega)

def central_diff_Jvp_approx(input, model, store_device, skip_layers, cfg):

    params = parameters_to_vector(model.named_parameters(), skip_layers)
    vec_params_shape = params.size()
    omega = torch.zeros(*vec_params_shape).normal_(0, 1).to(store_device)
    eps = get_eps(params, omega)
    f_add = apply_perturbed_model(input, model, eps*omega, skip_layers, cfg)
    f_min = apply_perturbed_model(input, model, -eps*omega, skip_layers, cfg)
    return (f_add - f_min) / (2 * eps)

def randomised_SVD_jacobian(input, model, ray_trafo, cfg, return_on_cpu=False):

    model.eval()
    store_device = ('cpu' if input.device == torch.device('cpu') else 'cuda:0')
    ray_trafo = ray_trafo.to(store_device) if ray_trafo is not None else None 

    forward_map_list = []
    for _ in range(cfg.spct.n_projs):
        with torch.no_grad():
            if ray_trafo is not None:
                diff_approx = ray_trafo(central_diff_Jvp_approx(input, model, store_device, cfg.spct.skip_layers, cfg.mdl))
            else:
                diff_approx = central_diff_Jvp_approx(input, model, store_device, cfg.spct.skip_layers, cfg.mdl)
            forward_map_list.append(diff_approx.view(1, -1).cpu())
    forward_map = torch.cat(forward_map_list).t()

    q, _ = torch.qr(forward_map, some=True)
    q = q.to(store_device)

    b_t_ = []
    for l in range(q.size()[-1]):
        q_l = q[:, l]
        if ray_trafo is not None:
            out = ray_trafo(apply_model_on_data(input, model, cfg.mdl)).view(-1)
        else:
            out = apply_model_on_data(input, model, cfg.mdl).view(-1)

        model.zero_grad()
        out.backward(q_l.view(-1))
        b_l = agregate_flatten_weight_grad(model, cfg.spct.skip_layers)
        b_t_.append(b_l.cpu())

    b_matrix = torch.stack(b_t_, dim=0)
    _, s, vh = torch.svd_lowrank(b_matrix.cpu(), q=cfg.spct.n_projs, niter=2, M=None)
    return (s.to(store_device), vh.t().to(store_device)) if not return_on_cpu else (s.cpu(), vh.t().cpu())