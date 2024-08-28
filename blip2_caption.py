import numpy as np
import onnxruntime as rt
import argparse
from PIL import Image, ImageFile
import os
import huggingface_hub
import argparse
from glob import glob
from multiprocessing import Pool, current_process
from tqdm import tqdm
import json
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", nargs="+", type=str, default=".")
    parser.add_argument("--resume", default=False, action="store_true")
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--rel_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=1)
    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = os.path.join(args.dataset_path, "blip2_captions.json")
    else:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    if args.rel_path is None:
        args.rel_path = args.dataset_path

    return args


def gen_captions(image_path):
    global model
    global processor
    image = Image.open(image_path)
    inputs = processor(image, return_tensors="pt").to(model.device)
    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[
        0
    ].strip()
    return generated_text


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


def init_subprocess(model_path, num_gpus):
    global model
    global processor
    processor = Blip2Processor.from_pretrained(model_path)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_path,
        device_map=f"cuda:{(current_process()._identity[0] - 1) % num_gpus}",
    )


if __name__ == "__main__":
    args = parse_args()
    if isinstance(args.dataset_path, list):
        print(
            args.dataset_path,
        )
        image_paths = []
        for single_dataset_path in args.dataset_path:
            image_paths = image_paths + glob(
                f"{single_dataset_path}/**", recursive=True
            )
    else:
        image_paths = glob(f"{args.dataset_path}/**", recursive=True)

    image_paths = [image_path for image_path in image_paths if is_image(image_path)]
    if args.resume:
        with open(args.save_path, "r") as f:
            prompts = json.load(f)

        image_paths = [
            image_path
            for image_path in image_paths
            if os.path.relpath(image_path, args.rel_path) not in prompts.keys()
        ]
    else:
        prompts = {}
    print(f"num images:{len(image_paths)}")
    print("gen tags")
    with Pool(
        processes=args.num_processes,
        initializer=init_subprocess,
        initargs=(args.model_path, args.num_gpus),
    ) as p:
        results = list(tqdm(p.imap(gen_captions, image_paths), total=len(image_paths)))

    for image_path, prompt in zip(image_paths, results):
        prompts[os.path.relpath(image_path, args.rel_path)] = prompt

    with open(args.save_path, "w") as f:
        json.dump(prompts, f, indent=4)
