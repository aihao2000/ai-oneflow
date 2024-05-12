import numpy as np
import onnxruntime as rt
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=".")
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--rel_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--check_images", default=False, action="store_true")
    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = os.path.join(args.dataset_path, "wd_tagger.json")
    else:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    if args.rel_path is None:
        args.rel_path = args.dataset_path

    return args


# Dataset v3 series of models:
SWINV2_MODEL_DSV3_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
CONV_MODEL_DSV3_REPO = "SmilingWolf/wd-convnext-tagger-v3"
VIT_MODEL_DSV3_REPO = "SmilingWolf/wd-vit-tagger-v3"

# Dataset v2 series of models:
MOAT_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-moat-tagger-v2"
SWIN_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-swinv2-tagger-v2"
CONV_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-convnext-tagger-v2"
CONV2_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-convnextv2-tagger-v2"
VIT_MODEL_DSV2_REPO = "SmilingWolf/wd-v1-4-vit-tagger-v2"

# Files to download from the repos
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

kaomojis = [
    "0_0",
    "(o)_(o)",
    "+_+",
    "+_-",
    "._.",
    "<o>_<o>",
    "<|>_<|>",
    "=_=",
    ">_<",
    "3_3",
    "6_9",
    ">_o",
    "@_@",
    "^_^",
    "o_o",
    "u_u",
    "x_x",
    "|_|",
    "||_||",
]


def load_labels(dataframe):
    name_series = dataframe["name"]
    name_series = name_series.map(
        lambda x: x.replace("_", " ") if x not in kaomojis else x
    )
    tag_names = name_series.tolist()

    rating_indexes = list(np.where(dataframe["category"] == 9)[0])
    general_indexes = list(np.where(dataframe["category"] == 0)[0])
    character_indexes = list(np.where(dataframe["category"] == 4)[0])
    return tag_names, rating_indexes, general_indexes, character_indexes


def mcut_threshold(probs):
    """
    Maximum Cut Thresholding (MCut)
    Largeron, C., Moulin, C., & Gery, M. (2012). MCut: A Thresholding Strategy
     for Multi-label Classification. In 11th International Symposium, IDA 2012
     (pp. 172-183).
    """
    sorted_probs = probs[probs.argsort()[::-1]]
    difs = sorted_probs[:-1] - sorted_probs[1:]
    t = difs.argmax()
    thresh = (sorted_probs[t] + sorted_probs[t + 1]) / 2
    return thresh


class Predictor:
    def __init__(
        self,
        repo_path=SWINV2_MODEL_DSV3_REPO,
        resume_download=False,
        cache_dir=".cache",
        device="cuda",
        device_id=0,
    ):

        os.makedirs(os.path.join(cache_dir, repo_path), exist_ok=True)

        if (
            os.path.exists(os.path.join(cache_dir, repo_path, MODEL_FILENAME))
            and os.path.exists(os.path.join(cache_dir, repo_path, LABEL_FILENAME))
            and not resume_download
        ):
            csv_path = os.path.join(cache_dir, repo_path, LABEL_FILENAME)
            model_path = os.path.join(cache_dir, repo_path, MODEL_FILENAME)
        else:
            csv_path = huggingface_hub.hf_hub_download(
                repo_path,
                LABEL_FILENAME,
                local_dir=f"{cache_dir}/{repo_path}",
                local_dir_use_symlinks=False,
            )
            model_path = huggingface_hub.hf_hub_download(
                repo_path,
                MODEL_FILENAME,
                local_dir=f"{cache_dir}/{repo_path}",
                local_dir_use_symlinks=False,
            )

        tags_df = pd.read_csv(csv_path)
        sep_tags = load_labels(tags_df)

        self.tag_names = sep_tags[0]
        self.rating_indexes = sep_tags[1]
        self.general_indexes = sep_tags[2]
        self.character_indexes = sep_tags[3]

        if device == "cpu":
            providers = None
            provider_options = None
        else:
            providers = ["CUDAExecutionProvider"]
            provider_options = [{"device_id": device_id}]

        model = rt.InferenceSession(
            model_path, providers=providers, provider_options=provider_options
        )
        _, height, width, _ = model.get_inputs()[0].shape
        self.model_target_size = height

        self.model = model

    def prepare_image(self, image):
        target_size = self.model_target_size

        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

        # Pad image to square
        image_shape = image.size
        max_dim = max(image_shape)
        pad_left = (max_dim - image_shape[0]) // 2
        pad_top = (max_dim - image_shape[1]) // 2

        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        # Resize
        if max_dim != target_size:
            padded_image = padded_image.resize(
                (target_size, target_size),
                Image.BICUBIC,
            )

        # Convert to numpy array
        image_array = np.asarray(padded_image, dtype=np.float32)

        # Convert PIL-native RGB to BGR
        image_array = image_array[:, :, ::-1]

        return np.expand_dims(image_array, axis=0)

    def predict(
        self,
        image,
        general_thresh=0.35,
        general_mcut_enabled=False,
        character_thresh=0.85,
        character_mcut_enabled=False,
    ):

        image = self.prepare_image(image)

        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        preds = self.model.run([label_name], {input_name: image})[0]

        labels = list(zip(self.tag_names, preds[0].astype(float)))

        # First 4 labels are actually ratings: pick one with argmax
        ratings_names = [labels[i] for i in self.rating_indexes]
        rating = dict(ratings_names)

        # Then we have general tags: pick any where prediction confidence > threshold
        general_names = [labels[i] for i in self.general_indexes]

        if general_mcut_enabled:
            general_probs = np.array([x[1] for x in general_names])
            general_thresh = mcut_threshold(general_probs)

        general_res = [x for x in general_names if x[1] > general_thresh]
        general_res = dict(general_res)

        # Everything else is characters: pick any where prediction confidence > threshold
        character_names = [labels[i] for i in self.character_indexes]

        if character_mcut_enabled:
            character_probs = np.array([x[1] for x in character_names])
            character_thresh = mcut_threshold(character_probs)
            character_thresh = max(0.15, character_thresh)

        character_res = [x for x in character_names if x[1] > character_thresh]
        character_res = dict(character_res)

        sorted_general_strings = sorted(
            general_res.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        sorted_general_strings = [x[0] for x in sorted_general_strings]
        sorted_general_strings = (
            ", ".join(sorted_general_strings).replace("(", "\(").replace(")", "\)")
        )

        return sorted_general_strings, rating, character_res, general_res


def gen_tags(image_path):
    global predictor
    return predictor.predict(
        Image.open(image_path).convert("RGBA"),
        general_thresh=0.35,
        general_mcut_enabled=False,
        character_thresh=0.85,
        character_mcut_enabled=False,
    )[0]


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


def init_subprocess(device):
    global predictor
    predictor = Predictor(device=device, device_id=current_process()._identity[0] - 1)


if __name__ == "__main__":
    args = parse_args()

    image_paths = glob(f"{args.dataset_path}/**", recursive=True)
    image_paths = [image_path for image_path in image_paths if is_image(image_path)]
    
    if args.check_images:
        print("check images")
        with Pool() as p:
            results = list(
                tqdm(p.imap(is_valid_image, image_paths), total=len(image_paths))
            )
        image_paths = [image_paths[i] for i in range(len(image_paths)) if results[i]]

    print(f"num images:{len(image_paths)}")
    print("gen tags")
    with Pool(
        processes=args.num_processes,
        initializer=init_subprocess,
        initargs=(args.device,),
    ) as p:
        results = list(tqdm(p.imap(gen_tags, image_paths), total=len(image_paths)))

    prompts = {}
    if os.path.exists(args.save_path):
        print(f"{args.save_path} exists")
        with open(args.save_path, "r") as f:
            json.load(prompts, f)

    for image_path, prompt in zip(image_paths, results):
        if os.path.relpath(image_path, args.rel_path) in prompts.keys():
            if isinstance(prompts[os.path.relpath(image_path, args.rel_path)], str):
                prompts[os.path.relpath(image_path, args.rel_path)] = [
                    prompts[os.path.relpath(image_path, args.rel_path)],
                    prompt,
                ]
            elif isinstance(prompts[os.path.relpath(image_path, args.rel_path)], list):
                prompts[os.path.relpath(image_path, args.rel_path)] = prompts[
                    os.path.relpath(image_path, args.rel_path)
                ] + [prompt]
            else:
                print(
                    f"invalid prompt type { os.path.relpath(image_path, args.rel_path)}:"
                )
                print(prompts[os.path.relpath(image_path, args.rel_path)])
        else:
            prompts[os.path.relpath(image_path, args.rel_path)] = prompt

    with open(args.save_path, "w") as f:
        json.dump(prompts, f, indent=4)
