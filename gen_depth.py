from PIL import Image
import os
import argparse
from glob import glob
from tqdm import tqdm
import json
from diffusers import MarigoldDepthPipeline
from torch.utils.data import DataLoader
import torch


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


def safe_load(image_path):
    image = None
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        pass
    return image


if __name__ == "__main__":
    pipe = MarigoldDepthPipeline.from_pretrained(
        "/mnt/nj-aigc/usr/aihao/workspace/deeplearning-content/models/prs-eth/marigold-lcm-v1-0",
        variant="fp16",
        torch_dtype=torch.float16,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    image_paths = glob(
        f"/mnt/nj-public02/dataset/aihao-datasets/JourneyDB/JourneyDB/data/train/imgs_unzip/**",
        recursive=True,
    )
    image_paths = [image_path for image_path in image_paths if is_image(image_path)]
    data_loader = DataLoader(image_paths, batch_size=10)

    for batch in tqdm(data_loader):
        data = [(image_path, safe_load(image_path)) for image_path in batch]
        data = [(image_path, image) for image_path, image in data if image is not None]
        data_map = {}
        for image_path, image in data:
            if image.size not in data_map.keys():
                data_map[image.size] = []
            data_map[image.size].append((image_path, image))
        for mini_batch in data_map.values():
            images = [image for image_path, image in mini_batch]
            image_paths = [image_path for image_path, image in mini_batch]
            save_paths = [
                image_path.replace("imgs_unzip", "depths") for image_path in image_paths
            ]
            results = pipe(images, num_inference_steps=1).prediction
            results = pipe.image_processor.visualize_depth(results)
            for result, save_path in zip(results, save_paths):
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                result.save(save_path)
