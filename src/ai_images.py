import os
import sys
from io import BytesIO
from typing import Final
import torch
from diffusers import DiffusionPipeline
from utils import TEMP_DIRECTORY_PATH, DATASETS_DIRECTORY_PATH, PICKLE, Block, resize_image

DATASET_NAME: Final[str] = "things.pkl"
IMAGES_PER_KEYWORD: Final[int] = 250

KEYWORDS: Final[list] = [
    "hiking trail", "island", "duck", "temple", "wheat field", "bluebonnet field", "bamboo",
    "zebra", "jellyfish", "eclipse", "wildflower meadow", "crystal cave",
    "polar bear", "desert oasis", "surfing", "forest", "pagoda", "ancient ruins",
    "glowing jellyfish", "sloth", "snowy owl", "bioluminescent bay",
    "bonsai tree", "hot air balloon", "beach", "gondola", "parrot", "orchid",
    "ocean", "starfish", "river", "car", "castle", "desert", "butterfly",
    "starry sky", "fireworks", "tiger", "mountain", "waterfall",
    "bamboo grove", "sunflower", "peacock", "beaver", "dolphin", "sunny beach",
    "chapel", "maple tree", "canal", "whale", "fairy lights", "windmill",
    "panda", "giant tortoise", "sushi", "starry night", "lily pads", "statue",
    "avalanche", "seahorse", "toucan", "dragonfly", "barn", "space shuttle",
    "sand dunes", "ice cave", "rainbow", "robin", "mountain peak", "water lilies",
    "cherry blossom", "koala", "moon", "puffin", "wisteria", "rice terraces",
    "swan", "tropical fish", "pyramid", "cat", "penguin", "garden", "glacier",
    "aurora", "firefly", "giraffe", "thunderstorm", "carnival", "aurora borealis",
    "nebula", "canyon", "cottage", "bamboo forest", "cave", "carousel", "sakura tree",
    "cactus", "mushroom", "redwood forest", "coral reef", "cathedral", "mountain lake",
    "lighthouse", "lakeside", "northern lights", "city skyline", "fox", "jungle",
    "hibiscus", "arctic wolf", "squirrel", "rose garden", "vineyard", "universe",
    "palm tree", "lavender field", "iceberg", "ancient temple", "sunset", "kangaroo",
    "forest", "lotus flower", "snowflake", "eagle", "glacier bay", "corn field",
    "tent", "koi pond", "monarch butterfly", "moonlit beach", "lizard", "dog",
    "snake", "bird", "camel", "lion", "cow"
]

# For Animals:
# KEYWORDS: Final[list] = [
#    "dog", "cat", "bird", "fish", "rabbit", "mouse", "horse", "cow",
#    "pig", "sheep", "chicken", "duck", "goose", "turkey", "deer", "bear",
#    "fox", "squirrel", "raccoon", "elephant", "giraffe", "lion", "tiger",
#    "cheetah", "leopard", "zebra", "hippopotamus", "rhino", "gorilla",
#    "chimpanzee", "orangutan", "koala", "kangaroo", "platypus", "dolphin",
#    "whale", "shark", "octopus", "squid", "jellyfish", "turtle", "frog",
#    "toad", "salamander", "newt", "lizard", "snake", "crocodile", "alligator",
#    "tortoise"
#]

# For smiling / not-smiling Dogs:
# KEYWORDS: Final[list] = [
#    "close-up of a smiling dog", "close-up of a sad dog"
#]

DATASET_FILE_PATH: Final[str] = os.path.join(DATASETS_DIRECTORY_PATH, DATASET_NAME)
if os.path.exists(DATASET_FILE_PATH):
    print("Dataset file path already exists. Operation aborted.")
    sys.exit(1)

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def get_pipe() -> DiffusionPipeline:
    """
    Returns a DiffusionPipeline instance with the specified settings.

    Returns:
        DiffusionPipeline: A DiffusionPipeline instance.
    """

    is_cuda_available = torch.cuda.is_available()
    device = "cuda" if is_cuda_available else "cpu"

    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype = torch.float16 if is_cuda_available else torch.float32,
        use_safetensors = True,
        variant = "fp16",
    )

    pipe.to(device)

    return pipe


def generate_ai_image(pipe: DiffusionPipeline, about: str) -> bytes:
    """
    Generate an image using the given pipe and about information.

    Args:
        pipe (DiffusionPipeline): The pipe to use for generating the image.
        about (str): The about information for the image.

    Returns:
        str: The generated image as bytes.
    """

    prompt = "a photo of a " + about
    image = pipe(prompt=prompt).images[0]

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
    Main function to generate and store AI-generated images based on specified keywords.
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
