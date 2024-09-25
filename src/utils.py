import os
import json
import gzip
import pickle
import socket
import http.client
import urllib.error
import urllib.request
from threading import Lock
from base64 import b64encode
from traceback import format_exc
from typing import Final, Tuple, Optional, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy


CURRENT_DIRECTORY_PATH: Final[str] = os.path.dirname(os.path.abspath(__file__))\
    .replace("\\", "/").replace("//", "/").replace("//", "/").replace("/src", "")
DATASETS_DIRECTORY_PATH: Final[str] = os.path.join(CURRENT_DIRECTORY_PATH, "datasets")
TEMP_DIRECTORY_PATH: Final[str] = os.path.join(CURRENT_DIRECTORY_PATH, "tmp")

if not os.path.exists(TEMP_DIRECTORY_PATH):
    os.makedirs(TEMP_DIRECTORY_PATH, exist_ok = True)


def handle_exception(exception: Exception) -> None:
    """
    Handles an exception.

    Args:
        exception (Exception): The exception to handle.
    """

    traceback = format_exc()
    print(exception, traceback)


def convert_image_to_base64(image_data: bytes, image_type: str = "webp") -> str:
    """
    Converts image data to a Base64-encoded data URL.

    Args:
        image_data (bytes): The raw image data to be converted.
        image_type (str): The type of the image (e.g., "jpeg", "png", "webp"). 
                                    Defaults to "webp".

    Returns:
        str: A Base64-encoded data URL representing the image.
    """

    encoded_image = b64encode(image_data).decode('utf-8')

    data_url = f"data:image/{image_type};base64,{encoded_image}"

    return data_url


def resize_image(image_data: bytes, size: Tuple[int, int] = (200, 200),
                 new_quality: Optional[int] = 70) -> bytes:
    """
    Resize an image to a specified size and optionally adjust the quality.

    Args:
        image_data (bytes): The input image data in bytes.
        size (Tuple[int, int], optional): A tuple specifying the desired
            output size (width, height). Defaults to (200, 200).
        new_quality (Optional[int], optional): The desired quality for the output
            image (only for WebP format). Defaults to 70. If None, the default
            encoder setting is used.

    Returns:
        bytes: The resized image data in WebP format.
    """

    numpy_array = numpy.frombuffer(image_data, numpy.uint8)

    image = cv2.imdecode(numpy_array, cv2.IMREAD_UNCHANGED)
    resized_img = cv2.resize(image, size, interpolation=cv2.INTER_AREA)

    encode_params = None
    if new_quality is not None:
        encode_params = [cv2.IMWRITE_WEBP_QUALITY, new_quality]

    _, img_encoded = cv2.imencode('.webp', resized_img, encode_params)
    return img_encoded.tobytes()


def download_file(url: str) -> Optional[bytes]:
    """
    Downloads a file from a given URL and returns its content as bytes.

    Args:
        url (str): The URL from which to download the file.

    Returns:
        Optional[bytes]: The file content in bytes if the download is successful.
    """

    request = urllib.request.Request(
        url, headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                " (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.3"
            )
        }
    )

    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            image_data = response.read()

        return image_data

    except (urllib.error.HTTPError, urllib.error.URLError,
            socket.timeout, FileNotFoundError, PermissionError,
            http.client.RemoteDisconnected, UnicodeEncodeError,
            TimeoutError, http.client.IncompleteRead,
            http.client.HTTPException, ConnectionResetError,
            ConnectionAbortedError, ConnectionRefusedError,
            ConnectionError) as exc:
        handle_exception(exc)

    return None


def can_read(file_path: str) -> bool:
    """
    Checks if a file can be read.

    Args:
        file_path (str): The name to the file to check.

    Returns:
        bool: True if the file can be read, False otherwise.    
    """

    if not os.path.isfile(file_path):
        return False

    return os.access(file_path, os.R_OK)


def can_write(file_path: str, content_size: Optional[int] = None) -> bool:
    """
    Checks if a file can be written to.

    Args:
        file_path (str): The path to the file to check.
        content_size (Optional[int]): The size of the content to write to the file.

    Returns:
        bool: True if the file can be written to, False otherwise.    
    """

    directory_path = os.path.dirname(file_path)
    if not os.path.isdir(directory_path):
        return False

    if not os.access(directory_path, os.W_OK):
        return False

    if content_size is not None:
        if not os.path.getsize(directory_path)\
            + content_size <= os.stat(directory_path).st_blksize:

            return False

    return True


def read(file_path: str, as_bytes: bool = False, default: Any = None) -> Any:
    """
    Reads a file.
    
    Args:
        file_path (str): The path to the file to read.
        default (Any, optional): The default value to return if the file
                                 does not exist. Defaults to None.
        as_bytes (bool, optional): Whether to return the file as bytes. Defaults to False.

    Returns:
        Any: The contents of the file, or the default value if the file does not exist.
    """

    if not can_read(file_path):
        return default

    try:
        with open(file_path, "r" + ("b" if as_bytes else ""),
                  encoding = None if as_bytes else "utf-8") as file:
            return file.read()

    except (FileNotFoundError, IsADirectoryError, IOError,
            PermissionError, ValueError, UnicodeDecodeError,
            TypeError, OSError) as exc:
        handle_exception(exc)

    return default


def write(file_path: str, content: Any) -> bool:
    """
    Writes a file.

    Args:
        file_path (str): The path to the file to write to.
        content (Any): The content to write to the file.

    Returns:
        bool: True if the file was written successfully, False otherwise.
    """

    if not can_write(file_path, len(content)):
        return False

    try:
        with open(file_path, "w" + ("b" if isinstance(content, bytes) else "")) as file:
            file.write(content)

        return True

    except (FileNotFoundError, IsADirectoryError, IOError,
            PermissionError, ValueError, TypeError, OSError) as exc:
        handle_exception(exc)

    return False


file_locks: dict = {}
WRITE_EXECUTOR = ThreadPoolExecutor()


class CachedFile:
    """
    A interface for an file type with caching.
    """


    def __init__(self) -> None:
        self._data = {}


    def _get_cache(self, file_path: str) -> Any:
        """
        Gets the cached value for the given file path.

        Args:
            file_path (str): The path to the file to get the cached value for.

        Returns:
            Any: The cached value for the given file path.
        """

        return self._data.get(file_path)


    def _set_cache(self, file_path: str, value: Any) -> None:
        """
        Sets the cached value for the given file path.

        Args:
            file_path (str): The path to the file to set the cached value for.
            value (Any): The value to set the cached value to.
        
        Returns:
            None
        """

        self._data[file_path] = value


    def _load(self, file_path: str) -> Any:
        """
        Loads the file.

        Args:
            file_path (str): The path to the file to load.

        Returns:
            Any: The loaded file.
        """

        return read(file_path)


    def _dump(self, data: Any, file_path: str) -> bool:
        """
        Writes the data to the file.

        Args:
            data (Any): The data to write to the file.
            file_path (str): The path to the file to write to.

        Returns:
            bool: True if the file was written successfully, False otherwise.
        """

        return write(file_path, data)


    def load(self, file_path: str,
             default: Any = None) -> Any:
        """
        Loads the file.

        Args:
            file_path (str): The path to the file to load.
            default (Any, optional): The default value to return if the file
                                     does not exist. Defaults to None.

        Returns:
            Any: The loaded file.
        """

        file_data = self._get_cache(file_path)

        if file_data is None:
            if not can_read(file_path):
                print("Cannot read", file_path)
                return default

            if file_path not in file_locks:
                file_locks[file_path] = Lock()

            with file_locks[file_path]:
                try:
                    data = self._load(file_path)
                except (FileNotFoundError, IsADirectoryError, IOError,
                        PermissionError, ValueError, json.JSONDecodeError,
                        pickle.UnpicklingError, UnicodeDecodeError) as exc:
                    handle_exception(exc)
                else:

                    self._set_cache(file_path, data)
                    return data

            return default

        return file_data


    def dump(self, file_path: str, data: Any, as_thread: bool = False) -> bool:
        """
        Dumps the data to the file.

        Args:
            file_path (str): The path to the file to dump the data to.
            data (Any): The data to dump to the file.
            as_thread (bool, optional): Whether to dump the data as a thread. Defaults to False.
        
        Returns:
            bool: True if the data was dumped successfully, False otherwise.
        """

        file_directory = os.path.dirname(file_path)

        if not can_write(file_directory, len(data)):
            return False

        if file_path not in file_locks:
            file_locks[file_path] = Lock()

        self._set_cache(file_path, data)

        try:
            if as_thread:
                WRITE_EXECUTOR.submit(self._dump, data, file_path)
            else:
                self._dump(data, file_path)
        except (FileNotFoundError, IsADirectoryError, IOError,
                PermissionError, ValueError, TypeError,
                pickle.PicklingError, OSError, RuntimeError) as exc:
            handle_exception(exc)

        return True


class PICKLEFile(CachedFile):
    """
    A pickle file type with caching.
    """


    def _load(self, file_path: str) -> Any:
        """
        Loads the file.

        Args:
            file_path (str): The path to the file to load.

        Returns:
            Any: The loaded file.
        """

        with open(file_path, 'rb') as file:
            return pickle.load(file)


    def _dump(self, data: Any, file_path: str) -> None:
        """
        Writes the data to the file.

        Args:
            data (Any): The data to write to the file.
        
        Returns:
            bool: True if the file was written successfully, False otherwise.
        """

        with file_locks[file_path]:
            with open(file_path, 'wb') as file:
                pickle.dump(data, file)


PICKLE = PICKLEFile()


def init_dataset(file_path: str, dataset_type: str = "images") -> None:
    """
    Create a new dataset pickle file.

    Parameters:
        file_path (str): The path where the pickle file will be created.
        dataset_type (str): The type of dataset (default is "images").

    Returns:
        None
    """

    PICKLE.dump(file_path, {
        "type": dataset_type,
        "keys": {}
    })


def set_key(file_path: str, key: str, value: Any) -> None:
    """
    Updates or adds a key-value pair in a pickle file.

    Args:
        file_path (str): The path to the pickle file.
        key (str): The key to set or update.
        value (Any): The value to associate with the key.

    Returns:
        None
    """

    data = PICKLE.load(file_path, default={})
    keys = data.get("keys", {})

    keys[key] = value
    data["keys"] = keys

    PICKLE.dump(file_path, data)


def download_and_compress(url: str) -> Optional[bytes]:
    """
    Downloads a file from the specified URL and compresses its contents.

    Args:
        url (str): The URL of the file to download.

    Returns:
        Optional[bytes]: The compressed bytes of the downloaded file, or None 
        if the download fails.
    """

    downloaded_bytes = download_file(url)
    if downloaded_bytes is None:
        return None

    compressed_bytes = gzip.compress(downloaded_bytes, 9)
    return compressed_bytes


def decompress(compressed_bytes: bytes) -> Optional[bytes]:
    """
    Decompresses the given gzip-compressed bytes.

    Args:
        compressed_bytes (bytes): The gzip-compressed data to decompress.

    Returns:
        Optional[bytes]: The decompressed data.
    """

    try:
        return gzip.decompress(compressed_bytes)
    except OSError as exc:
        handle_exception(exc)

    return None


def download_files_parallel_and_save_to_one_file(file_path: str, key: str,
                                                 file_urls: List[str]) -> None:
    """
    Downloads multiple files in parallel, compresses them, and saves the results to a single file.

    Args:
        file_path (str): The path to the file where the compressed data will be saved.
        key (str): The key under which the compressed data will be stored.
        file_urls (List[str]): A list of URLs of the files to download.

    Returns:
        None
    """

    downloaded_files = []

    with ThreadPoolExecutor(os.cpu_count() * 3) as executor:
        futures = []
        for url in file_urls:
            future = executor.submit(download_and_compress, url)
            futures.append(future)

        for future in as_completed(futures):
            compressed_bytes = future.result()

            if compressed_bytes is not None:
                downloaded_files.append(compressed_bytes)

    if len(downloaded_files) < 2:
        return

    set_key(file_path, key, downloaded_files)


def find_missing_numbers_in_range(range_start: int, range_end: int, data: list):
    """
    Finds missing numbers in a specified range based on provided data.

    Args:
        range_start (int): The starting value of the range (exclusive).
        range_end (int): The ending value of the range (inclusive).
        data (list): A list of items, where each item contains at least one 
                     number to check against the range.

    Returns:
        list: A list of numbers that are missing from the specified range.
    """

    numbers = list(range(range_start + 1, range_end + 1))

    for item in data:
        if item[0] in numbers:
            numbers.remove(item[0])

    return numbers


class Block:
    """
    Functions for saving data in blocks instead of alone.
    """


    def __init__(self, file_path: str, block_size: int = 1) -> None:
        """
        Initializes a Block object.

        Args:
            file_path (str): The name of the file to write the block to.
            block_size (int): How big each block is.
        """

        self.file_path = file_path
        self.block_size = block_size
        self.blocks = {}


    def _get_id(self, index: int) -> int:
        """
        Returns the nearest block index based on the given index and block size.

        Args:
            index (int): The index value.

        Returns:
            int: The nearest block index.
        """

        remains = index % self.block_size

        if remains == 0:
            return index

        return index + (self.block_size - remains)


    def _write_data(self, block_data: tuple) -> None:
        """
        Writes data to a file while ensuring thread safety using locks.

        Args:
            block_data (tuple): A tuple containing data to be written to the file.
        """

        data = PICKLE.load(self.file_path, default=[])

        for _, new_data in block_data:
            if new_data is not None:
                data.append(new_data)

        PICKLE.dump(self.file_path, data, True)


    def add_data(self, index: int, new_data: bytes) -> Tuple[bool, Optional[int]]:
        """
        Adds new data to the specified index in the data structure, and writes the block to file
        if all expected data within the block range is present.

        Args:
            index (int): The index where the new data should be added.
            new_data (bytes): The data to be added, if any.

        Returns:
            Tuple[bool, Optional[int]]: A tuple indicating success and the block ID.
        """

        block_id = self._get_id(index)

        block = self.blocks.get(block_id, [])
        block.append((index, new_data))
        self.blocks[block_id] = block

        missing = find_missing_numbers_in_range(block_id - self.block_size, block_id, block)
        if 1 in missing:
            missing.remove(1)

        if len(missing) == 0:
            self._write_data(block)
            del self.blocks[block_id]

            return True, block_id

        return False, block_id


    @property
    def size(self) -> int:
        """
        Returns the size of the data structure.

        Checks if the file exists and if so, loads the data from the file using the 'load' function.
        Returns the length of the loaded data. If the file does not exist, returns 0.

        Returns:
            int: The size of the data structure.
        """

        data = PICKLE.load(self.file_path, default = [])
        return len(data)
