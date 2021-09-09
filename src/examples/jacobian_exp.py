
import torch
import torch.nn as nn
from copy import deepcopy

class SimpleNet(torch.nn.Module):
    """
    A 2 hidden-layer neural network
    """
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 25),
            nn.LeakyReLU(0.2),
            nn.Linear(25, 25),
            nn.LeakyReLU(0.2),
            nn.Linear(25, 200),
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

def agregate_flatten_weight_grad(model, default_device):
    grads_o = []
    for name, params in model.named_parameters():
        grads_o.append(params.grad.flatten().to(device=default_device))
    return torch.cat(grads_o)

def compute_jacobian_single_batch(input, model, out_dim):
    jac = []
    model.eval()
    for o in range(out_dim):
        f = model(input).squeeze()
        f_o = f[o]
        model.zero_grad()
        f_o.backward()
        store_device = 'cpu' if f_o.device == torch.device('cpu') else 'cuda:0'
        jacs_o = agregate_flatten_weight_grad(model, default_device=store_device).detach()
        jac.append(jacs_o)
    return torch.stack(jac, dim=0)

def apply_model(input, params, model):
    nn.utils.vector_to_parameters(params, model.parameters())
    return model(input)

def compute_central_diff_approx(input, model, eps=1e-3):

    omega = eps*torch.zeros_like(nn.utils.parameters_to_vector(model.parameters())).normal_(0, 1)
    model_1, model_2 = deepcopy(model), deepcopy(model)
    params_add = nn.utils.parameters_to_vector(model_1.parameters()) + eps * omega
    f_add = apply_model(input, params_add, model_1)
    params_min = nn.utils.parameters_to_vector(model_2.parameters()) - eps * omega
    f_min = apply_model(input, params_min, model_2)
    return (f_add - f_min) /2*eps

def randomised_SVD_jacobian(input, model, n_rank=50):

    forward_maps = []
    for _ in range(n_rank):
        forward_maps.append(compute_central_diff_approx(input, model))
    forward_map = torch.cat(forward_maps).transpose(-2, -1)
    q, _ = torch.linalg.qr(forward_map)
    b_t_ = []
    for l in range(q.size()[-1]):
        q_l = q[:, l]
        out_l = model(input) @ q_l
        model.zero_grad()
        out_l.backward(retain_graph=True)
        store_device = 'cpu' if out_l.device == torch.device('cpu') else 'cuda:0'
        b_l = agregate_flatten_weight_grad(model, default_device=store_device).detach()
        b_t_.append(b_l)
    b_matrix = torch.stack(b_t_, dim=0)
    u, s, vh = torch.linalg.svd(b_matrix, full_matrices=False)
    v = vh.transpose(-2, -1).conj()
    return s, v


if __name__ == "__main__":

    seed=42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    input = torch.zeros((1, 10)).normal_(0, 1)
    net = SimpleNet()
    jac = compute_jacobian_single_batch(input, net, 200)
    u, s, vh = torch.linalg.svd(jac, full_matrices=False)
    v = vh.transpose(-2, -1).conj()
    s_approx, v_approx = randomised_SVD_jacobian(input, net)
    print('MSE first 20 singular values: {}'.format(torch.mean((s[:len(s_approx)] - s_approx)**2)))
    import matplotlib.pyplot as plt
    plt.plot(list(range(0, 200)), s);
    plt.plot(list(range(0, 50)), s_approx); plt.show()
