"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.nn.functional as F

from guided_diffusion import logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)

def main():
    args = create_argparser().parse_args()

    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(th.load(args.model_path, map_location="cpu"))
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    model.to(device)
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(th.load(args.classifier_path, map_location="cpu"))
    classifier.to(device)
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    all_noised_samples = []
    all_unnoised_samples = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        classes = th.randint(low=0, high=NUM_CLASSES, size=(args.batch_size,), device=device)
        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample, intermediate_samples = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=device,
            return_intermediate_samples=True
        )  # intermediate_samples: list of dict, "sample" and "unnoised_sample" are tensor images
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        intermediate_noised_samples = []
        intermediate_unnoised_samples = []
        for intmd_sample in intermediate_samples:
            
            noised_sample = intmd_sample["sample"]
            noised_sample = ((noised_sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            noised_sample = noised_sample.permute(0, 2, 3, 1)
            noised_sample = noised_sample.contiguous()

            unnoised_sample = intmd_sample["unnoised_sample"]
            unnoised_sample = ((unnoised_sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            unnoised_sample = unnoised_sample.permute(0, 2, 3, 1)
            unnoised_sample = unnoised_sample.contiguous()

            intermediate_noised_samples.append(noised_sample.cpu().numpy())
            intermediate_unnoised_samples.append(unnoised_sample.cpu().numpy())
        logger.log(f"created {len(intermediate_noised_samples)} noised samples")
        logger.log(f"created {len(intermediate_unnoised_samples)} unnoised samples")

        all_images.append(sample.cpu().numpy())
        all_labels.append(classes.cpu().numpy())
        all_noised_samples.append(np.stack(intermediate_noised_samples, axis=0))  # list of (N_steps, H, W, 3) images
        all_unnoised_samples.append(np.stack(intermediate_noised_samples, axis=0))  # list of (N_steps, H, W, 3) images
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]

    noised_samples = np.stack(all_noised_samples, axis=0)  # (N_samples, N_steps, H, W, 3)
    unnoised_samples = np.stack(all_unnoised_samples, axis=0)  # (N_samples, N_steps, H, W, 3)
    logger.log(f"noised_samples: {noised_samples.shape}")
    logger.log(f"unnoised_samples: {unnoised_samples.shape}")
    
    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    logger.log(f"saving to {out_path}")
    np.savez(out_path, arr, label_arr)
    out_path = os.path.join(logger.get_dir(), f"intermediate_samples_{shape_str}.npz")
    np.savez(out_path, noised_samples, unnoised_samples)
    
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()