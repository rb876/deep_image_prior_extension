import torch
from torch.nn import MSELoss
from odl.contrib.torch import OperatorModule

class Attack:

    """
    Base class for all attacks.
    """

    def __init__(self, name, model, ray_trafos):
        """
        Initializes adversarial attack state.

        Arguments:
        ----------
        name: name of the attack
        model: :class:`torch.Tensor` to attack
        """

        self.name = name
        self.model = model
        self.ray_trafos = ray_trafos
        self._training_mode = False
        self.device = torch.device(('cuda:0'
                                    if torch.cuda.is_available() else 'cpu'
                                   ))

    def forward(self, *inputs):
        """
        It defines the computation performed at every call.
        Should be overridden by all subclasses.
        """

        raise NotImplementedError

    def __call__(self, *inputs):
        training_mode = self.model.training

        if self._training_mode:
            self.model.train()
        else:
            self.model.eval()  # default option

        adv_fbp = self.forward(*inputs)

        if training_mode:
            self.model.train()

        return adv_fbp


class PGDAttack(Attack):

    def __init__(self, model, ray_trafos, steps=10, eps=1, alpha=0.1, eps_for_division=1e-10):
        super().__init__('PGD-L2', model, ray_trafos)
        self.steps = steps
        self.eps = eps
        self.alpha = alpha
        self.eps_for_division = eps_for_division
        self.smooth_pinv_ray_trafo_module = \
            self.ray_trafos['smooth_pinv_ray_trafo_module'
                            ].to(self.device)
        self.criterion = MSELoss()

    def forward(self, obs, gt, randn_mask=None):

        batch_size = len(gt)
        obs = obs.clone().detach().to(self.device)
        gt = gt.clone().detach().to(self.device)
        adv_obs = obs.clone().detach()

        costs = []
        for i in range(self.steps):
            adv_obs.requires_grad_(True)
            outputs = \
                self.model(self.smooth_pinv_ray_trafo_module(adv_obs))
            cost = self.criterion(outputs, gt)
            costs.append(cost.item())

            # Update adversarial observation

            grad = torch.autograd.grad(cost, adv_obs,
                    retain_graph=False, create_graph=False)[0]
            grad_norms = torch.norm(grad.view(batch_size, -1), p=2,
                                    dim=1) + self.eps_for_division
            grad = grad / grad_norms.view(batch_size, 1, 1, 1)
            adv_obs = adv_obs.detach() + self.alpha * grad.sign()

            delta = adv_obs - obs
            delta_norms = torch.norm(delta.view(batch_size, -1), p=2,
                    dim=1)
            factor = self.eps / delta_norms
            factor = torch.min(factor, torch.ones_like(delta_norms))
            delta = delta * factor.view(-1, 1, 1, 1)
            adv_obs = (obs + delta)

        return self.smooth_pinv_ray_trafo_module(adv_obs), costs
