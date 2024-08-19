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
from ultralytics import YOLO
from diffusers.utils import load_image
from ultralytics.utils import LOGGER

LOGGER.setLevel("ERROR")


ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=".")
    parser.add_argument("--model_path", type=str, default="yolov8x6_animeface.pt")
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--rel_path", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=1)
    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = os.path.join(args.dataset_path, "face_boxes.json")
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


def process(image_path):
    global model
    image = Image.open(image_path)
    results = model(
        image,
        save=False,
        conf=0.3,
        iou=0.5,
    )
    return results[0].boxes.xyxy.tolist()


def init_subprocess(model_path, num_gpus):
    global model
    model = YOLO(model_path).to(
        f"cuda:{(current_process()._identity[0] - 1) % num_gpus}"
    )


if __name__ == "__main__":
    args = parse_args()

    image_paths = glob(f"{args.dataset_path}/**", recursive=True)
    image_paths = [image_path for image_path in image_paths if is_image(image_path)]

    with Pool(
        processes=args.num_processes,
        initializer=init_subprocess,
        initargs=(
            args.model_path,
            args.num_gpus,
        ),
    ) as p:
        results = list(tqdm(p.imap(process, image_paths), total=len(image_paths)))

    face_boxes = {}
    for image_path, boxes in zip(image_paths, results):
        face_boxes[os.path.relpath(image_path, args.rel_path)] = boxes

    with open(args.save_path, "w") as f:
        json.dump(face_boxes, f, indent=4)
