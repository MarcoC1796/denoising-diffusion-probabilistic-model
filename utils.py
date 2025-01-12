import torch


class AlphaBarCache:
    def __init__(self, T, beta_1=10e-4, beta_T=0.02, device="cpu"):
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T
        self.device = device
        beta_t = self.linear_beta_scheduler()
        self.alpha_bar_t = torch.cumprod(1 - beta_t, dim=0)

    def __call__(self, t):
        return self.alpha_bar_t[t - 1]

    def linear_beta_scheduler(self):
        timesteps = torch.arange(1, self.T + 1, device=self.device)
        return self.beta_1 + (self.beta_T - self.beta_1) * (timesteps - 1) / (
            self.T - 1
        )


def get_noise(batch_size, config, device):
    eps = torch.normal(
        0,
        1,
        size=(batch_size, config.in_img_C, config.in_img_S, config.in_img_S),
        device=device,
    )
    return eps
