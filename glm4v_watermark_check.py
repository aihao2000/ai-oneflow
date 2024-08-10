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
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=".")
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=1)
    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = os.path.join(args.dataset_path, "wartermark_image_paths.txt")
    else:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    return args


def vqa(
    tokenizer,
    model,
    image,
    query="Describe the character in the picture as concisely as possible.",
):
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "image": image, "content": query}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(
        model.device
    )  # chat mod
    gen_kwargs = {
        "max_length": 1000,
        "do_sample": True,
        "top_k": 1,
        "no_repeat_ngram_size": 5,
    }
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(outputs[0])
        response = response.split("<|endoftext|>")[0]

    return response


def image_watermark_check(image_path):
    global model
    global tokenizer
    image = Image.open(image_path)
    response = vqa(
        tokenizer,
        model,
        image,
        "Does this image have any watermarks or trademarks? Please reply yes or no.",
    ).lower()
    return "yes" in response


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
    global tokenizer
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        .to(f"cuda:{(current_process()._identity[0] - 1) % num_gpus}")
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )


if __name__ == "__main__":
    args = parse_args()

    image_paths = glob(f"{args.dataset_path}/**", recursive=True)
    image_paths = [image_path for image_path in image_paths if is_image(image_path)]

    print(f"num images:{len(image_paths)}")
    print("gen tags")
    with Pool(
        processes=args.num_processes,
        initializer=init_subprocess,
        initargs=(args.model_path, args.num_gpus),
    ) as p:
        results = list(
            tqdm(p.imap(image_watermark_check, image_paths), total=len(image_paths))
        )

    watermark_image_paths = []

    for image_path, result in zip(image_paths, results):
        if result:
            watermark_image_paths.append(image_path + "\n")

    with open(args.save_path, "w") as f:
        f.writelines(watermark_image_paths)
