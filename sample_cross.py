from torch.utils.data import DataLoader
import einops
import numpy as np
import torch
from PIL import Image
from cldm.model import create_model, load_state_dict
from ldm.models.diffusion.ddim import DDIMSampler
from kinetics import Kinetics700InterpolateBase
from annotator.hed import TorchHEDdetector
from pathlib import Path

# p = "./train_log/kin_hed_dropout1/lightning_logs/version_1/checkpoints/epoch=6-step=595385.ckpt"
p = "../ControlNet/train_log/kin_hed_cross_dropout2/lightning_logs/version_5/checkpoints/epoch=11-step=233754.ckpt"


model = create_model("./models/cldm_v15_cross.yaml").cuda()

model.load_state_dict(load_state_dict(p, location="cuda"))
ddim_sampler = DDIMSampler(model)
frozenClipImageEmbedder = model.style_encoder

ddim_steps = 50
strength = 1
eta = 0
scale = 5
batch_size = 4
seq_length = 60


dataset = Kinetics700InterpolateBase(
    sequence_time=None,
    sequence_length=seq_length,  # 2 seconds
    size=512,
    resize_size=None,
    random_crop=None,
    pixel_range=2,
    interpolation="bicubic",
    mode="val",
    data_path="/export/compvis-nfs/group/datasets/kinetics-dataset/k700-2020",
    dataset_size=1.0,
    filter_file="../ControlNet/data_val.json",
    flow_only=False,
    include_full_sequence=True,
    include_hed=True,
)

torch.manual_seed(42)
dl = DataLoader(dataset, shuffle=False, batch_size=1)
_iter = iter(dl)

apply_hed = TorchHEDdetector()

out_dir = "results/cross/time_diff_0"

Path(out_dir + "/pred").mkdir(parents=True, exist_ok=True)
Path(out_dir + "/true").mkdir(parents=True, exist_ok=True)


# runs on run currently

for i in range(250):
    # styles, structures = get_sequence(dl)

    styles_batch = []
    target_batch = []
    while len(styles_batch) < batch_size:
        # styles, structures = get_sequence(dl)
        batch = next(_iter)

        # first element of batch (full sequence)
        styles = batch["sequence"][0]

        # make sure the sequence has the desired length (as specified above)
        if styles.shape[0] < seq_length:
            print("seq too short; skipping...")
            continue

        styles = (styles + 1.0) * 127.5
        styles = styles.clip(0, 255).type(torch.uint8)

        # take image at specifc index
        styles_batch.append(styles[0])

        # he set offset
        target_batch.append(styles[0])

    # first element of batch (full sequence)
    styles = torch.cat(styles_batch, dim=0).cuda()
    target = torch.cat(target_batch, dim=0)

    structures = apply_hed(target.clone()) / 255.0
    control = einops.rearrange(structures.clone(), "b h w c -> b c h w").cuda()

    B, C, H, W = control.shape

    style_embedding = frozenClipImageEmbedder(styles)

    c_style = style_embedding.last_hidden_state
    c_embed = style_embedding.pooler_output
    c_prompt = model.get_learned_conditioning([""] * B)

    uc_style = torch.zeros_like(c_style)
    uc_embed = torch.zeros_like(c_embed)
    uc_prompt = model.get_learned_conditioning([""] * B)

    cond = {
        "c_concat": [control],
        "c_crossattn": [c_prompt],
        "c_style": [c_style],
        "c_embed": [c_embed],
    }
    un_cond = {
        "c_concat": [control],
        "c_crossattn": [uc_prompt],
        "c_style": [uc_style],
        "c_embed": [uc_embed],
    }
    shape = (4, H // 8, W // 8)

    # only need to potentially change this for guess mode
    model.control_scales = [strength] * 13

    samples, intermediates = ddim_sampler.sample(
        ddim_steps,
        B,
        shape,
        cond,
        verbose=False,
        eta=eta,
        unconditional_guidance_scale=scale,
        unconditional_conditioning=un_cond,
    )

    x_samples = model.decode_first_stage(samples) * 127.5 + 127.5
    x_samples = einops.rearrange(x_samples, "b c h w -> b h w c")
    x_samples = x_samples.cpu().numpy().clip(0, 255).astype(np.uint8)

    # target = batch["intermediate_frame"]
    # target = (target + 1.0) * 127.5
    # target = target.clip(0, 255).type(torch.uint8)

    for j in range(B):
        Image.fromarray(x_samples[j]).save(f"{out_dir}/pred/img_{i}-{j}.png")
        Image.fromarray(target[j].cpu().numpy()).save(f"{out_dir}/true/img_{i}-{j}.png")
