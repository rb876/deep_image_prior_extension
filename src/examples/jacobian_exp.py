
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from copy import deepcopy

class SimpleNet(torch.nn.Module):
    """
    A 2 hidden-layer neural network
    """
    def __init__(self, input_dim, out_dim, hidden_dim):
        super(SimpleNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, out_dim),
            nn.Tanh()
            )
        self.net.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.net(x)
        return x

def agregate_flatten_weight_grad(model):
    grads_o = []
    for name, params in model.named_parameters():
        grads_o.append(params.grad.flatten())
    return torch.cat(grads_o)

def compute_jacobian_single_batch(input, model, out_dim):
    jac = []
    model.eval()
    for o in range(out_dim):
        f = model(input).squeeze()
        f_o = f[o]
        model.zero_grad()
        f_o.backward()
        jacs_o = agregate_flatten_weight_grad(model).detach()
        jac.append(jacs_o)
    return torch.stack(jac, dim=0)

def apply_perturbed_model(input, model, omega):
    model = deepcopy(model)
    params = model.parameters()
    pert_params = parameters_to_vector(params) + omega
    vector_to_parameters(pert_params, model.parameters())
    return model(input)

def central_diff_approx(input, model, eps=1e-6):
    vec_params_shape = parameters_to_vector(model.parameters()).size()
    omega = eps * torch.zeros(*vec_params_shape).normal_(0, 1)
    f_add = apply_perturbed_model(input, model, omega)
    f_min = apply_perturbed_model(input, model, -omega)
    return (f_add - f_min)/2*eps

def randomised_SVD_jacobian(input, model, n_rank):

    model.eval()
    forward_maps = []
    for _ in range(n_rank):
        forward_maps.append(central_diff_approx(input, model))
    forward_map = torch.cat(forward_maps).transpose(-2, -1)
    q, _ = torch.linalg.qr(forward_map)
    b_t_ = []
    for l in range(q.size()[-1]):
        q_l = q[:, l].detach()
        out = model(input) @ q_l
        model.zero_grad()
        out.backward()
        b_l = agregate_flatten_weight_grad(model).detach()
        b_t_.append(b_l)
    b_matrix = torch.stack(b_t_, dim=0)
    u, s, vh = torch.svd_lowrank(b_matrix, q=n_rank, niter=2, M=None)
    v = vh.transpose(-2, -1).conj()
    return s, v

if __name__ == "__main__":

    seed=42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    n_input, n_out, n_rank = 10, 100, 50
    input = torch.zeros((1, n_input)).normal_(0, 1)
    net = SimpleNet(input_dim=n_input, out_dim=n_out, hidden_dim=50)
    import time
    start = time.time()
    jac = compute_jacobian_single_batch(input, net, n_out)
    end = time.time()
    print(end - start)
    u, s, vh = torch.linalg.svd(jac, full_matrices=True)
    v = vh.transpose(-2, -1).conj()
    start = time.time()
    s_approx, v_approx = randomised_SVD_jacobian(input, net, n_rank)
    end = time.time()
    print(end - start)
    print('MSE first 20 singular values: {}'.format(torch.mean((s[:len(s_approx)] - s_approx)**2)))
    import matplotlib.pyplot as plt
    plt.plot(list(range(0, n_out)), s);
    plt.plot(list(range(0, n_rank)), s_approx); plt.show()
    print(s_approx)
