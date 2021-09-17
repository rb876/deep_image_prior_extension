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

def compute_jacobian_single_batch(input, model, out_dim, skip_layers, cfg, return_on_cpu=False):
    jac = []
    model.eval()
    f = apply_model_on_data(input, model, cfg).view(-1)
    for o in range(out_dim):
        f_o = f[o]
        model.zero_grad()
        f_o.backward(retain_graph=True)
        jacs_o = agregate_flatten_weight_grad(model, skip_layers).detach()
        jac.append(jacs_o)
    return torch.stack(jac, dim=0) if not return_on_cpu else torch.stack(jac, dim=0).cpu()

def agregate_flatten_weight_grad(model, skip_layers):
    grads_o = []
    for name, params in model.named_parameters():
        if name not in skip_layers:
            grads_o.append(params.grad.flatten())
    return torch.cat(grads_o)

def apply_perturbed_model(input, model, omega, skip_layers, cfg):
    model = deepcopy(model)
    model.eval()
    with torch.no_grad(): 
        params = model.named_parameters()
        pert_params = parameters_to_vector(params, skip_layers) + omega
        vector_to_parameters(pert_params, model.named_parameters(), skip_layers)
    return apply_model_on_data(input, model, cfg)

def set_eps_andrei(params, omega):
    eps = np.sqrt(torch.finfo(params.dtype).eps)
    return eps * (1 + LA.norm(params, ord=float('inf'))) \
        / LA.norm(omega, ord=float('inf'))

def get_eps(params, omega, mode='andrei'):
    if mode == 'preset':
        return 1e-6
    elif mode == 'andrei':
        return set_eps_andrei(params, omega)

def central_diff_approx(input, model, store_device, skip_layers, cfg):

    params = parameters_to_vector(model.named_parameters(), skip_layers)
    vec_params_shape = params.size()
    omega = torch.zeros(*vec_params_shape).normal_(0, 1).to(store_device)
    eps = get_eps(params, omega)
    f_add = apply_perturbed_model(input, model, eps*omega, skip_layers, cfg)
    f_min = apply_perturbed_model(input, model, -eps*omega, skip_layers, cfg)
    return (f_add - f_min) / (2 * eps)

def randomised_SVD_jacobian(input, model, n_rank, ray_trafo, skip_layers, cfg, return_on_cpu=False):

    model.eval()
    store_device = 'cpu' if input.device == torch.device('cpu') else 'cuda:0'
    forward_maps = []

    if ray_trafo is not None: 
        ray_trafo = ray_trafo.to(store_device)

    for _ in range(n_rank):
        if ray_trafo is not None:
            diff_approx = central_diff_approx(input, model, store_device, skip_layers, cfg)
            diff_approx = ray_trafo(diff_approx.view(1, *input.shape))
        elif ray_trafo is None:  
            diff_approx = central_diff_approx(input, model, store_device, skip_layers, cfg)
        else: 
            raise KeyError
        forward_maps.append(diff_approx.view(1, -1))

    forward_map = torch.cat(forward_maps).t()
    q, _ = torch.qr(forward_map.cpu(), some=True) # on cpu
    b_t_ = []
    for l in range(q.size()[-1]):
        q_l = q[:, l].to(store_device).detach()
        if ray_trafo is not None: 
            out = ray_trafo(apply_model_on_data(input, model, cfg)).view(-1)
        else: 
            out = apply_model_on_data(input, model, cfg).view(-1)
        model.zero_grad()
        out.backward(q_l.view(-1))
        b_l = agregate_flatten_weight_grad(model, skip_layers)
        b_t_.append(b_l.detach())

    b_matrix = torch.stack(b_t_, dim=0)
    _, s, vh = torch.svd_lowrank(b_matrix.cpu(), q=n_rank, niter=2, M=None) # on cpu
    return (s.to(store_device), vh.t().to(store_device)) if not return_on_cpu else (s.cpu(), vh.t().cpu())

''' 
Helper Subroutines 
'''

def get_fbp_from_loader(testloader, store_device):

    unpck = [el[1] for el in testloader]
    fbp = unpck[0].unsqueeze(dim=0)
    return fbp.to(store_device)

def get_params(reconstructor, cfg, filename, skip_layers):

    state_dict = torch.load(os.path.join(cfg.lin.path_to_checkpoints,
                            filename))
    reconstructor.model.load_state_dict(state_dict)
    params = \
        parameters_to_vector(reconstructor.model.named_parameters(),
                             skip_layers)
    return params

def compute_angle_btw_2vcts(vec1, vec2):

    inner_product = (vec1 * vec2).sum(dim=0)
    vec1_norm = vec1.pow(2).sum(dim=0).pow(0.5)
    vec2_norm = vec2.pow(2).sum(dim=0).pow(0.5)
    cos = inner_product / (2 * vec1_norm * vec2_norm)
    return torch.rad2deg(torch.acos(cos))

def singular_values_comparison_plot(s1, label1, s2, label2):
    import matplotlib.pyplot as plt 

    N1 = len(s1)
    N2 = len(s2)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(list(range(N1)), s1, '-.', linewidth=2.5, color='green', label=label1)
    ax.plot(list(range(N2)), s2, '-.', linewidth=2.5, color='red', label=label2, alpha=.5)
    ax.grid()
    ax.legend()
    ax.set_title('Singular Values')
    fig.savefig('singular_values_comparison_plot.pdf')

def singular_vecors_comparison_plot(v1, v2): 
    import matplotlib.pyplot as plt 
 
    vec1_len = list(range(v1.shape[1]))
    n_proj = v1.shape[0]
    fig, axs = plt.subplots(6, 5, figsize=(25, 15),  facecolor='w', edgecolor='k', constrained_layout=True)
    axs = axs.flatten()
    for i in range(n_proj):
        axs[i].plot(vec1_len, v1[i, :], '-', color='green')
        axs[i].plot(vec1_len, v2[i, :], '-', color='red', alpha=.5)
        axs[i].set_title(" n_proj = %s" % str(i))
    fig.savefig('singular_vectors_comparison_plot.pdf'.format(str(i)))

def projected_diff_params_plot(delta_params, v):
    import matplotlib.pyplot as plt 

    n_proj = v.shape[0]
    prj_error = []
    for i in range(n_proj):
        prj_error.append(torch.abs(torch.dot(delta_params, v[i, :])))
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(list(range(n_proj)), prj_error, '-.', linewidth=2.5, color='green')
    ax.grid()
    fig.savefig('projected_error_plot.pdf')
  