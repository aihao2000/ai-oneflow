import numpy as np
import onnxruntime as rt
import argparse
from PIL import Image, ImageFile
import os
import huggingface_hub
import pandas as pd
import argparse
from glob import glob
from multiprocessing import Pool, current_process
from tqdm import tqdm
import json

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=".")
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--save_path", type=str, default=None)
    args = parser.parse_args()

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


def init_subprocess(device, num_gpus):
    global predictor
    predictor = Predictor(
        device=device, device_id=(current_process()._identity[0] - 1) % num_gpus
    )


if __name__ == "__main__":
    args = parse_args()

    image_paths = glob(f"{args.dataset_path}/**", recursive=True)
    image_paths = [image_path for image_path in image_paths if is_image(image_path)]

    with Pool() as p:
        results = list(
            tqdm(p.imap(is_valid_image, image_paths), total=len(image_paths))
        )
    error_image_paths = [
        image_paths[i] + "\n" for i in range(len(image_paths)) if not results[i]
    ]
    with open(args.save_path, "w") as f:
        f.writelines(error_image_paths)
