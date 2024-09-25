import os
import sys
from typing import Final
from duckduckgo_search import DDGS
import duckduckgo_search.exceptions
from utils import DATASETS_DIRECTORY_PATH, handle_exception,\
    init_dataset, download_files_parallel_and_save_to_one_file


DATASET_NAME: Final[str] = "things.pkl"
IMAGES_PER_KEYWORD: Final[int] = 200

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

DATASET_FILE_PATH: Final[str] = os.path.join(DATASETS_DIRECTORY_PATH, DATASET_NAME)
if os.path.exists(DATASET_FILE_PATH):
    print("Dataset file path already exists. Operation aborted.")
    sys.exit(1)


def main() -> None:
    """
    Main function to download images based on predefined keywords and save them to a dataset file.

    Returns:
        None: This function does not return any value.
            It performs file operations and prints messages to the console.
    """

    init_dataset(DATASET_FILE_PATH)

    for keyword in KEYWORDS:
        for _ in range(3):
            try:
                image_search_result = DDGS().images(
                    keywords = keyword + ' images',
                    type_image = 'photo', max_results = IMAGES_PER_KEYWORD
                )
            except (duckduckgo_search.exceptions.TimeoutException,
                    duckduckgo_search.exceptions.ConversationLimitException,
                    duckduckgo_search.exceptions.DuckDuckGoSearchException,
                    duckduckgo_search.exceptions.RatelimitException) as exc:

                handle_exception(exc)
            else:
                break


        image_urls = [image.get("image") for image in image_search_result[:10]
                      if image.get("image") is not None]

        image_urls.extend([
            image.get("image") for image in image_search_result[10:]
            if keyword.lower() in image.get("title", keyword).lower() and\
            image.get("image") is not None
        ])

        download_files_parallel_and_save_to_one_file(DATASET_FILE_PATH, keyword, image_urls)


if __name__ == "__main__":
    main()
