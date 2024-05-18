import os
from controlnet_aux.processor import Processor
from tqdm import tqdm
from multiprocessing import Pool
from glob import glob
from PIL import Image
import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=".")
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--check_images", default=False, action="store_true")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--processor_id", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = os.path.join(args.dataset_path, "..", args.processor_id)

    os.makedirs(args.save_path, exist_ok=True)

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


def init_subprocess(processor_id):
    global processor
    processor = Processor(processor_id)


def get_annotation(parameters):
    image_path, save_path = parameters
    global processor
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    image = Image.open(image_path)
    result = processor(image, to_pil=True)
    result.save(save_path)


if __name__ == "__main__":
    args = parse_args()
    image_paths = glob(f"{args.dataset_path}/**", recursive=True)
    image_paths = [image_path for image_path in image_paths if is_image(image_path)]

    save_paths = [
        os.path.join(args.save_path, os.path.relpath(image_path, args.dataset_path))
        for image_path in image_paths
    ]
    input_parameters = [
        (image_path, save_path)
        for image_path, save_path in zip(image_paths, save_paths)
    ]

    if args.check_images:
        print("check images")
        with Pool() as p:
            results = list(
                tqdm(
                    p.imap(is_valid_image, image_paths),
                    total=len(image_paths),
                )
            )
        image_paths = [image_paths[i] for i in range(len(image_paths)) if results[i]]

    print(f"num images: {len(image_paths)}")
    print("gen image_annotation:")
    with Pool(
        processes=args.num_processes,
        initializer=init_subprocess,
        initargs=(args.processor_id,),
    ) as p:
        results = list(
            tqdm(p.imap(get_annotation, input_parameters), total=len(image_paths))
        )
