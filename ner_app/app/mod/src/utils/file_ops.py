import os
import logging
import requests

from tqdm import tqdm
from zipfile import ZipFile

from .config import DATA_PATH


def get_file(fname: str, url: str) -> tuple:
    fpath = os.path.join(DATA_PATH, fname)
    fdownloaded = False
    funzipped = False

    if not os.path.exists(fpath):
        try:
            logging.info(f"Downloading {fname}...")
            response = requests.get(url, stream=True)

            total_size_in_bytes = int(response.headers.get("content-length", 0))
            block_size = 1024
            progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)

            with open(fpath, "wb") as f:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    f.write(data)

            if fname.endswith(".zip"):
                logging.info(f"Extracting {fname}...")
                with ZipFile(fpath, "r") as f:
                    for file in tqdm(iterable=f.namelist(), total=len(f.namelist())):
                        f.extract(member=file)

        except Exception as e:
            logging.error(
                f"Error in downloading or unzipping file check below error.\n {e}"
            )

    return fpath, fdownloaded, funzipped


def read_file(fname: str) -> tuple:
    with open(fname, "r") as f:
        lines = f.readlines()

    x, y, unique_y = [[]], [[]], ["pad"]

    for line in lines:
        if line == "\n":
            x.append(list())
            y.append(list())
        else:
            split_y, split_x = line.lower().split("\t")
            x[-1].append(split_x.strip())
            y[-1].append(split_y.strip())

            if split_y.strip() not in unique_y:
                unique_y.append(split_y.split()[0])

    return x, y, sorted(unique_y)