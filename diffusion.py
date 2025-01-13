import os

import torch
from config import UNetConfig
from torch import Tensor
from unet import UNet
from utils import get_noise


def linear_beta_scheduler(T, beta_1: float, beta_T: float) -> Tensor:
    timesteps = torch.arange(1, T + 1)
    return beta_1 + (beta_T - beta_1) * (timesteps - 1) / (T - 1)


class Diffusion:
    def __init__(
        self,
        model_config: UNetConfig,
        num_diffusion_timesteps: int = 1000,
        device: str = "cpu",
        weights_path: str | None = None,
        beta_1: float = 0.001,
        beta_T: float = 0.02,
    ):
        self.T = num_diffusion_timesteps
        self.device = device
        self.model_config = model_config
        self.model = UNet(self.model_config).to(self.device)
        if weights_path:
            self.load_weights(weights_path)
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.betas = linear_beta_scheduler(self.T, self.beta_1, self.beta_T).to(
            self.device
        )
        self.alphas = 1 - self.betas
        self.alphas_bar = self.alphas.cumprod(dim=0)
        self.alphas_bar_prev = torch.cat(
            (torch.tensor([1.0], device=self.device), self.alphas_bar[:-1]), dim=0
        )
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_bar_prev) / (1.0 - self.alphas_bar)
        )
        self.log_variance = torch.log(
            torch.cat((self.posterior_variance[1:2], self.betas[1:]), dim=0)
        )
        print(self.posterior_variance[1:2], self.posterior_variance[1:2].shape)

    def load_weights(self, weights_path: str):
        self.weights_path = weights_path
        print("Loading pretrained model")
        self.model.load_state_dict(torch.load(weights_path, weights_only=True))

    def p_sample_loop(self, num_samples: int = 1) -> Tensor:
        self.model.eval()
        with torch.no_grad():
            img_t = get_noise(num_samples, self.model_config, self.device)
            for t in reversed(
                torch.arange(start=1, end=self.T + 1, device=self.device)
            ):
                sqrt_alpha_inv = 1 / torch.sqrt(self.alphas[t - 1])
                eps_scale = (1 - self.alphas[t - 1]) / torch.sqrt(
                    1 - self.alphas_bar[t - 1]
                )
                t_tensor = t.repeat(num_samples)
                eps_theta: Tensor = self.model(img_t, t_tensor)
                z = (
                    get_noise(num_samples, self.model_config, device=self.device)
                    if t > 1
                    else torch.tensor([0.0], device=self.device)
                )
                var = torch.exp(0.5 * self.log_variance[t - 1])
                img_t = sqrt_alpha_inv * (img_t - eps_scale * eps_theta) + var * z

        return img_t.to("cpu")


if __name__ == "__main__":
    import time

    model_config = UNetConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    weights_path = os.path.join("models", "diffusion.pth")
    diffusion = Diffusion(
        model_config=model_config, device=device, weights_path=weights_path
    )
    start = time.time()
    img = diffusion.p_sample_loop().squeeze()
    end = time.time()
    print("time:", start - end)
