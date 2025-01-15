import time
import mlflow

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
        grad_accum_steps: int | None = None,
        lr_rate: float = 0.001,
        max_train_steps: int | None = None,
        max_eval_steps: int | None = 100,
        num_diffusion_timesteps: int = 1000,
        shuffle: bool = True,
        save_checkpoint_steps: int | None = None,
    ):
        self.model = model
        self.model_config = model_config
        self.epochs = epochs
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.max_train_steps = max_train_steps
        self.max_eval_steps = max_eval_steps
        self.T = num_diffusion_timesteps
        self.shuffle = shuffle
        self.device = device
        self.alpha_bar_cache = AlphaBarCache(T=self.T, device=self.device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            params=self.model.parameters(), lr=lr_rate
        )
        self.curr_step = 0
        self.curr_test_loss = -1.0
        self.curr_dt_test = -1.0
        self.save_checkpoint_steps = save_checkpoint_steps

    def load_data(self) -> None:
        transforms = v2.Compose([ToTensor(), v2.Resize((64, 64))])
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
        curr_grad_accum_step = 0
        for (
            epoch_step,
            (x_0, _),
        ) in enumerate(self.train_dataloader):
            if (
                self.max_train_steps is not None
                and self.curr_step >= self.max_train_steps
            ):
                break
            if self.grad_accum_steps is None or curr_grad_accum_step == 0:
                t_start = time.perf_counter()

            x_0 = x_0.to(self.device)
            x_0 = (x_0 * 2) - 1
            # assert x_0.max() <= 1.0
            # assert x_0.min() >= -1.0
            batch_size = len(x_0)

            eps = get_noise(batch_size, self.model_config, self.device)
            t = torch.randint(
                low=1, high=self.T + 1, size=(batch_size,), device=self.device
            )
            alpha_bar_t = self.alpha_bar_cache(t).reshape(batch_size, 1, 1, 1)
            x_t = torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * eps

            epsilon_pred = self.model(x_t, t)
            loss = self.loss_fn(epsilon_pred, eps)
            if self.grad_accum_steps is not None:
                loss /= self.grad_accum_steps
            loss.backward()

            curr_grad_accum_step += 1
            if (
                self.grad_accum_steps is not None
                and curr_grad_accum_step < self.grad_accum_steps
            ):
                continue

            self.optimizer.step()
            self.optimizer.zero_grad()

            self.curr_step += 1
            curr_grad_accum_step = 0

            mlflow.log_metric("loss", loss.item(),step=self.curr_step, synchronous=False)

            if (
                self.save_checkpoint_steps is not None
                and self.curr_step % self.save_checkpoint_steps == 0
            ):
                self.save_model("checkpoint.pth")

            t_end = time.perf_counter()

            if self.curr_step > 0 and (self.curr_step) % log_steps == 0:
                if (self.curr_step) % eval_steps == 0:
                    self.curr_test_loss, self.curr_dt_test = self.eval()
                dt = t_end - t_start
                current = (epoch_step + 1) * batch_size
                images_per_sec = batch_size / dt
                pix_per_sec = batch_size * self.model.config.in_img_S**2 / dt
                print(
                    f"step {self.curr_step:4d} | loss: {loss.item():.6f} | dt: {dt * 1000:.2f}ms | img/sec: {images_per_sec:.2f} | pix/sec: {pix_per_sec:.2f} | {current:>5d}/{size:>5d} | test loss: {self.curr_test_loss:.6f} | dt test loss: {self.curr_dt_test:.2f}"
                )

        if curr_grad_accum_step != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def eval(self) -> tuple[float, float]:
        t_start = time.time()
        num_batches = 0
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
                num_batches +=1

        test_loss /= num_batches + 1e-5
        t_end = time.time()
        dt = t_end - t_start

        return test_loss, dt

    def save_model(self, save_path: str):
        torch.save(self.model.state_dict(), save_path)

    def run_training(
        self,
        save_path: str = "diffusion.pth",
        log_steps: int = 10,
        eval_steps: int = 100,
    ) -> None:
        self.load_data()
        print("Getting initial Eval Loss ...")
        self.curr_test_loss, self.curr_dt_test = self.eval()
        print(f"Test time: {self.curr_dt_test:.6f}")
        self.curr_step = 0
        for t in range(self.epochs):
            print(f"Epoch {t + 1}/{self.epochs}\n-------------------------------")
            self.train(log_steps=log_steps, eval_steps=eval_steps)

        print("Saving model")
        self.save_model(save_path=save_path)
        print("Done!")




if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    config = UNetConfig(
        in_img_S=64,
        in_img_C=3,
        ch_mult=(1, 2, 2, 2),
        attn_resolutions=(16,),
        dropout_rate=0.0,
    )
    model = UNet.from_pth(config, 'checkpoint_3.pth').to(device)
    trainer = DiffusionTrainer(
        model=model,
        model_config=config,
        device=device,
        epochs=1,
        batch_size=8,
        grad_accum_steps=None,
        lr_rate=0.0001,
        max_train_steps=10,
        max_eval_steps=20,
        save_checkpoint_steps=5,
    )
    with mlflow.start_run():
        trainer.run_training(log_steps=1)

# TODO
# gradient clipping