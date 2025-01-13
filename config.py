from pydantic import BaseModel


class UNetConfig(BaseModel):
    in_img_S: int = 32
    in_img_C: int = 3
    in_img_val_range: tuple[float, float] = (0.0, 1.0)
    n_ch: int = 128
    ch_mult: tuple[int, ...] = (1, 2, 2, 2)
    num_res_blocks: int = 2
    attn_resolutions: tuple[int] = (16,)
    dropout_rate: float = 0.1
