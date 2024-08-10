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

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=".")
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--rel_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_gpus", type=int, default=1)
    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = os.path.join(args.dataset_path, "wd_tagger.json")
    else:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    if args.rel_path is None:
        args.rel_path = args.dataset_path

    return args


def gen_tags(image_path):
    global tokenizer
    global model
    image = Image.open(image_path)
    query = "Use concise language to describe the content of the image."

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "image": image, "content": query}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(
        device
    )  # chat mod
    gen_kwargs = {
        "max_length": 2500,
        "do_sample": True,
        "top_k": 1,
        "no_repeat_ngram_size": 5,
    }
    with torch.no_grad():
        outputs = vlm.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(outputs[0])
        response = response.split("<|endoftext|>")[0]

    return response


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


def init_subprocess(model_path, device, num_gpus):
    global tokenizer
    global model
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        .to(device)
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
        initargs=(args.device, args.num_gpus),
    ) as p:
        results = list(tqdm(p.imap(gen_tags, image_paths), total=len(image_paths)))

    prompts = {}

    for image_path, prompt in zip(image_paths, results):
        prompts[os.path.relpath(image_path, args.rel_path)] = prompt

    with open(args.save_path, "w") as f:
        json.dump(prompts, f, indent=4)
