import torch
import torch.nn as nn
import snntorch as snn

__all__ = ["LIF", "stable_sigmoid"]


class _StableSigmoidSurrogate(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_: torch.Tensor, slope: float) -> torch.Tensor:
        ctx.save_for_backward(input_)
        ctx.slope = float(slope)
        return (input_ > 0).to(input_.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (input_,) = ctx.saved_tensors
        sig = torch.sigmoid(ctx.slope * input_)
        grad_input = grad_output * ctx.slope * (1.0 - sig)
        return grad_input, None

def stable_sigmoid(slope: float = 4.0):
    """Return a stable Heaviside-forward / sigmoid-gradient surrogate."""
    slope = float(slope)

    def inner(input_: torch.Tensor) -> torch.Tensor:
        return _StableSigmoidSurrogate.apply(input_, slope)

    return inner

class LIF(snn.Leaky):
    """Single-step snnTorch LIF matched to the original SpikingJelly dynamics.

    The upstream model used SpikingJelly LIF with ``decay_input=True``::

        V[t] = (1 - 1/tau) * V[t-1] + (1/tau) * X[t]

    snnTorch Leaky uses ``beta * U[t-1] + I[t]``.  We therefore map
    ``beta = 1 - 1/tau`` and scale each input current by ``1/tau``.

    This class is intentionally *single step*.  It accepts no time dimension
    and performs no temporal loop.
    """

    def __init__(self, tau: float = 2.0, threshold: float = 1.0):
        if tau <= 1.0:
            raise ValueError(f"tau must be > 1, got {tau}")
        self.tau =  float(tau)
        self.input_scale = 1.0 / self.tau
        beta = 1.0 - self.input_scale
        super().__init__(
            beta=beta,
            threshold=threshold,
            spike_grad=stable_sigmoid(slope=4.0),
            init_hidden=True,
            reset_mechanism="zero",
            reset_delay=False,
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.mem.shape != input_.shape:
            self.mem = torch.zeros_like(input_)

        beta = self.beta.clamp(0, 1)
        charged = beta * self.mem + input_ * self.input_scale
        spk = self.fire(charged)

        reset =  spk.detach()
        self.mem = charged * (1.0 - reset)
        return spk