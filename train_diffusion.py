import time

import torch
from config import UNetConfig
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets  # type: ignore
from torchvision.transforms import (  # type: ignore
    ToTensor,
    v2,
)
from unet import UNet
from utils import AlphaBarCache, get_noise


class DiffusionTrainer:
    def __init__(
        self,
        model: UNet,
        model_config: UNetConfig,
        device: str = "cpu",
        epochs: int = 10,
        batch_size: int = 64,
        lr_rate: float = 0.001,
        max_train_steps: int | None = None,
        max_eval_steps: int | None = 100,
        num_diffusion_timesteps: int = 1000,
        shuffle: bool = True,
    ):
        self.model = model
        self.model_config = model_config
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_train_steps = max_train_steps
        self.max_eval_steps = max_eval_steps
        self.T = num_diffusion_timesteps
        self.shuffle = shuffle
        self.device = device
        self.alpha_bar_cache = AlphaBarCache(T=self.T, device=self.device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=lr_rate
        )  # , lr=2 * 10e-4)
        self.curr_step = 0
        self.curr_test_loss = -1.0
        self.curr_dt_test = -1.0

    def load_data(self) -> None:
        transforms = v2.Compose([ToTensor(), v2.Resize((256, 256))])
        train_data = datasets.CelebA(
            root="data", split="train", download=True, transform=transforms
        )
        test_data = datasets.CelebA(
            root="data", split="valid", download=True, transform=transforms
        )
        self.train_dataloader = DataLoader(
            train_data, batch_size=self.batch_size, shuffle=self.shuffle
        )
        self.eval_dataloader = DataLoader(
            test_data, batch_size=self.batch_size, shuffle=True
        )

    def train(self, log_steps=10, eval_steps=100) -> None:
        size = len(self.train_dataloader.dataset)  # type: ignore
        self.model.train()
        for (
            step,
            (x_0, _),
        ) in enumerate(self.train_dataloader):
            if self.max_train_steps is not None and step >= self.max_train_steps:
                break
            t_start = time.perf_counter()
            x_0 = x_0.to(self.device)
            x_0 = (x_0 * 2) - 1
            assert x_0.max() <= 1.0
            assert x_0.min() >= -1.0

            eps = get_noise(self.batch_size, self.model_config, self.device)
            t = torch.randint(
                low=1, high=self.T + 1, size=(self.batch_size,), device=self.device
            )
            alpha_bar_t = self.alpha_bar_cache(t).reshape(self.batch_size, 1, 1, 1)
            x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * eps

            epsilon_pred = self.model(x_t, t)
            loss = self.loss_fn(epsilon_pred, eps)

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            self.curr_step += 1

            t_end = time.perf_counter()

            if step > 0 and (step + 1) % log_steps == 0:
                if (step + 1) % eval_steps == 0:
                    self.curr_test_loss, self.curr_dt_test = self.eval()
                dt = t_end - t_start
                current = (step + 1) * self.batch_size
                images_per_sec = self.batch_size / dt
                pix_per_sec = self.batch_size * self.model.config.in_img_S**2 / dt
                print(
                    f"step {self.curr_step:4d} | loss: {loss.item():.6f} | dt: {dt * 1000:.2f}ms | img/sec: {images_per_sec:.2f} | pix/sec: {pix_per_sec:.2f} | {current:>5d}/{size:>5d} | test loss: {self.curr_test_loss:.6f} | dt test loss: {self.curr_dt_test:.2f}"
                )

    def eval(self) -> tuple[float, float]:
        t_start = time.time()
        num_batches = len(self.eval_dataloader)
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for steps, (x_0, _) in enumerate(self.eval_dataloader):
                if self.max_eval_steps is not None and steps >= self.max_eval_steps:
                    break
                x_0 = x_0.to(self.device)
                batch_size = len(x_0)
                eps = get_noise(batch_size, self.model_config, self.device)
                t = torch.randint(
                    low=1, high=self.T + 1, size=(batch_size,), device=self.device
                )
                alpha_bar_t = self.alpha_bar_cache(t).reshape(batch_size, 1, 1, 1)
                x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * eps
                pred = self.model(x_t, t)
                test_loss += self.loss_fn(pred, eps).item()

        test_loss /= num_batches
        t_end = time.time()
        dt = t_end - t_start

        return test_loss, dt

    def save_model(self, save_path: str):
        torch.save(self.model.state_dict(), save_path)

    def run_training(self, save_path: str = "diffusion.pth") -> None:
        self.load_data()
        temp_max_eval_steps = self.max_eval_steps
        self.max_eval_steps = 200
        print("Getting initial Test Loss ...")
        initial_test_loss, dt = self.eval()
        self.max_eval_steps = temp_max_eval_steps
        print(f"Initial Test Loss: {initial_test_loss:.6f} | dt: {dt}")
        self.curr_test_loss, self.curr_dt_test = self.eval()
        print(f"Test time: {self.curr_dt_test:.6f}")
        self.curr_step = 0
        for t in range(self.epochs):
            print(f"Epoch {t + 1}/{self.epochs}\n-------------------------------")
            self.train()

        temp_max_eval_steps = self.max_eval_steps
        self.max_eval_steps = None
        final_test_loss, dt = self.eval()
        self.max_eval_steps = temp_max_eval_steps
        print(f"Final Test Loss: {final_test_loss:.6f} | dt: {dt}")
        print("Saving model")
        self.save_model(save_path=save_path)
        print("Done!")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    config = UNetConfig(
        in_img_S=256, in_img_C=3, ch_mult=(1, 1, 2, 2, 4, 4), dropout_rate=0.0
    )
    model = UNet(config).to(device)
    trainer = DiffusionTrainer(
        model=model,
        model_config=config,
        device=device,
        epochs=1,
        batch_size=4,
        lr_rate=0.00002,
        max_train_steps=1000,
        max_eval_steps=20,
    )
    trainer.run_training()


if __name__ == "__main__":
    main()

# TODO
# save checkpoints
# gradient clipping
# fire cli
# logging
