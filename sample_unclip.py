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
p = "./train_log/kin_hed_unclip3/lightning_logs/version_4/checkpoints/epoch=1-step=159230.ckpt"


def make_conditionings_from_input(img, model, num=1):
    with torch.no_grad():
        adm_cond = model.embedder(img)
        weight = 1
        if model.noise_augmentor is not None:
            noise_level = 0
            c_adm, noise_level_emb = model.noise_augmentor(
                adm_cond,
                noise_level=einops.repeat(
                    torch.tensor([noise_level]).to(model.device), "1 -> b", b=num
                ),
            )
            adm_cond = torch.cat((c_adm, noise_level_emb), 1) * weight
        adm_uc = torch.zeros_like(adm_cond)
    return adm_cond, adm_uc, weight


def main():
    model = create_model("./models/cldm_unclip-h-inference.yaml").cuda()
    model.load_state_dict(load_state_dict(p, location="cuda"))
    ddim_sampler = DDIMSampler(model)

    ddim_steps = 50
    strength = 1
    eta = 0
    scale = 5
    batch_size = 4
    seq_length = 4

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
        include_hed=False,
    )

    torch.manual_seed(42)
    dl = DataLoader(dataset, shuffle=False, batch_size=1)
    _iter = iter(dl)

    apply_hed = TorchHEDdetector()

    out_dir = "results/unclip/time_diff_0"

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

            # take image at specifc index
            styles_batch.append(styles[[0]])

            # he set offset
            target_batch.append(styles[[0]])

        # first element of batch (full sequence)
        styles = torch.cat(styles_batch, dim=0)
        styles = einops.rearrange(styles, "b h w c -> b c h w").cuda()

        target = torch.cat(target_batch, dim=0)
        target = (target + 1.0) * 127.5
        target = target.clip(0, 255).type(torch.uint8)

        structures = apply_hed(target.clone()) / 255.0
        control = einops.rearrange(structures.clone(), "b h w c -> b c h w").cuda()

        B, C, H, W = control.shape

        adm_cond, adm_uc, w = make_conditionings_from_input(styles, model, num=B)

        c_prompt = model.get_learned_conditioning([""] * B)

        uc_prompt = model.get_learned_conditioning([""] * B)

        cond = {
            "c_concat": [control],
            "c_crossattn": [c_prompt],
            "c_adm": adm_cond,
        }
        un_cond = {
            "c_concat": [control],
            "c_crossattn": [uc_prompt],
            "c_adm": adm_uc,
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
            Image.fromarray(target[j].cpu().numpy()).save(
                f"{out_dir}/true/img_{i}-{j}.png"
            )


if __name__ == "__main__":
    main()
