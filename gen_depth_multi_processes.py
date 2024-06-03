import numpy as np
import argparse
from PIL import Image
import os
import huggingface_hub
import pandas as pd
import argparse
from glob import glob
from multiprocessing import Pool, current_process
from tqdm import tqdm
import json
from diffusers import MarigoldDepthPipeline
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--rel_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--model_path", type=str, default="prs-eth/marigold-depth-lcm-v1-0"
    )
    parser.add_argument("--check_images", default=False, action="store_true")
    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = os.path.join(args.dataset_path, "wd_tagger.json")
    else:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    if args.rel_path is None:
        args.rel_path = args.dataset_path

    return args


def is_image(image_path):
    image_types = ["png", "jpg", ".peg", "gif", "webp", "bmp", "jpeg"]
    if image_path.split(".")[-1] not in image_types:
        return False
    # try:
    #     Image.open(image_path).convert("RGBA")
    # except Exception:
    #     print(f"Error opening {image_path}")
    #     return False
    else:
        return True


def is_valid_image(image_path):
    try:
        Image.open(image_path).convert("RGBA")
    except Exception:
        print(f"Error opening {image_path}")
        return False
    else:
        return True


def init_subprocess(model_path):
    global pipe
    pipe = MarigoldDepthPipeline.from_pretrained(
        model_path,
        torch_dtyoe=torch.float16,
    ).to(f"cuda:{(current_process()._identity[0] - 1)%8}")

    pipe.set_progress_bar_config(disable=True)


def process(image_path):
    global pipe
    save_path = image_path.replace("imgs_unzip", "depths")
    if os.path.exists(save_path):
        return None
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        return None

    result = pipe(image, num_inference_steps=4).prediction
    result = pipe.image_processor.visualize_depth(result)[0]
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    result.save(save_path)


if __name__ == "__main__":
    args = parse_args()

    image_paths = glob(f"{args.dataset_path}/**", recursive=True)
    image_paths = [image_path for image_path in image_paths if is_image(image_path)]

    print(f"num images:{len(image_paths)}")
    print("gen tags")
    with Pool(
        processes=args.num_processes,
        initializer=init_subprocess,
        initargs=(args.model_path,),
    ) as p:
        results = list(tqdm(p.imap(process, image_paths), total=len(image_paths)))
