import os
import sys
from io import BytesIO
from typing import Final
import torch
from diffusers import StableDiffusionPipeline
from utils import TEMP_DIRECTORY_PATH, DATASETS_DIRECTORY_PATH, PICKLE, Block, resize_image


DATASET_NAME: Final[str] = "faces.pkl"
IMAGES_PER_KEYWORD: Final[int] = 2500

KEYWORDS: Final[list] = ["smiling", "sad"]

DATASET_FILE_PATH: Final[str] = os.path.join(DATASETS_DIRECTORY_PATH, DATASET_NAME)
if os.path.exists(DATASET_FILE_PATH):
    print("Dataset file path already exists. Operation aborted.")
    sys.exit(1)

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def get_pipe() -> StableDiffusionPipeline:
    """
    Returns a StableDiffusionPipeline instance with the specified settings.

    Returns:
        StableDiffusionPipeline: A StableDiffusionPipeline instance.
    """

    is_cuda_available = torch.cuda.is_available()
    device = "cuda" if is_cuda_available else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained(
        "Sourabh2/Human_Face_generator",
        torch_dtype = torch.float16 if is_cuda_available else torch.float32,
        use_safetensors = True,
        variant = "fp16",
    )

    pipe.to(device)

    return pipe


def generate_ai_image(pipe: StableDiffusionPipeline, about: str) -> bytes:
    """
    Generate an image using the given pipe and about information.

    Args:
        pipe (StableDiffusionPipeline): The pipe to use for generating the image.
        about (str): The about information for the image.

    Returns:
        str: The generated image as bytes.
    """

    prompt = "a " + about + " face of a woman"
    image = pipe(prompt=prompt, num_inference_steps=250).images[0]

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    return image_bytes


def get_file_path(keyword: str) -> str:
    """
    Generate a file path based on the provided keyword.

    Args:
        keyword (str): The keyword used to generate the file path. Spaces in the
                       keyword will be replaced with underscores.

    Returns:
        str: The full file path for the generated file path.
    """

    return os.path.join(
        TEMP_DIRECTORY_PATH, keyword.replace(" ", "_") + ".pkl"
    )


def main() -> None:
    """
    Main function to generate and store AI-generated faces.
    """

    pipe = None

    for keyword in KEYWORDS:
        block = Block(file_path = get_file_path(keyword))
        if block.size >= IMAGES_PER_KEYWORD:
            continue

        for i in range(IMAGES_PER_KEYWORD - block.size):
            if pipe is None:
                pipe = get_pipe()

            image = generate_ai_image(pipe, keyword)
            image = resize_image(image)
            block.add_data(i, image)

    keys_and_images = {}
    for keyword in KEYWORDS:
        file_path = get_file_path(keyword)

        keys_and_images[keyword] = PICKLE.load(file_path)
        os.remove(file_path)

    PICKLE.dump(DATASET_FILE_PATH, {"type": "image", "keys": keys_and_images})


if __name__ == "__main__":
    main()
