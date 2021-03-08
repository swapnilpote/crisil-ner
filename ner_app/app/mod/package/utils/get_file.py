import os
import logging
import zipfile
import requests

from app.mod.packages.genie.utils.config import DATA_PATH
from app.mod.packages.genie import logging

logger = logging.getLogger(__name__)


def download_file(url: str, filename: str) -> str:
    """[File download functionality from internet]

    Args:
        url (str): [Url of the file which needs to be downloaded]
        filename (str): [Final name of the file]

    Returns:
        str: [Saved filepath to be used for further use]
    """
    # TODO
    # 1. Add tqdm to track download progress

    fpath = os.path.join(DATA_PATH, filename)

    if not os.path.exists(fpath):
        try:
            r = requests.get(url, stream=True)
            logger.info(f"Downloaded {filename} file.")

            with open(fpath, "wb") as f:
                f.write(r.content)
                logger.info(
                    f"Successfully stored {filename} file inside {DATA_PATH} folder"
                )
        except Exception as e:
            logger.error(f"Unable to download/store {filename} file due to {e}.")
    else:
        logger.info(f"{filename} already exist inside {fpath}")

    return fpath


def unpack_file(filepath: str) -> bool:
    """[Unpack file different kind of compressed files]

    Args:
        filepath (str): [Pass the path of the compressed file]

    Returns:
        bool: [If file is uncompressed then return True otherwise False]
    """
    # TODO
    # 1. Add condition to handle different kind of file extension for uncompressing

    boolean = True
    try:
        with zipfile.ZipFile(filepath, "r") as f:
            f.extractall(DATA_PATH)
    except Exception as e:
        logger.error(f"Unable to extract {filepath} file")
        boolean = False

    return boolean