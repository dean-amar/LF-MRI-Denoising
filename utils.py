import argparse
import os
import yaml
import matplotlib
import cv2

import matplotlib.pyplot as plt
import numpy as np
import logging as logger

from pathlib import Path
from IPython.core.display_functions import clear_output

logger.basicConfig(level=logger.INFO)


def setupPlotter():
    matplotlib.use("TkAgg")
    os.environ["MPLBACKEND"] = "TkAgg"
    clear_output(wait=True)


def parseArgs():
    logger.info("parsing cli")
    parser = argparse.ArgumentParser(description="denoiser arguments")

    parser.add_argument(
        "--h",
        type=int,
        default=15,
        help="Smoothing parameter"
    )

    parser.add_argument(
        "--small_window_size",
        type=int,
        default=3,
        help="Size of the processing window"
    )

    parser.add_argument(
        "--big_window_size",
        type=int,
        default=21,
        help="Size of the search window"
    )

    parser.add_argument(
        "--plot",
        type=bool,
        default=False,
        help="plot image output"
    )

    parser.add_argument(
        "--save",
        type=bool,
        default=False,
        help="save denoised image"
    )

    parser.add_argument(
        "--single",
        type=bool,
        default=False,
        help="process single image"
    )

    parser.add_argument(
        "--process_all",
        type=bool,
        default=False,
        help="process all dataset"
    )

    parsedArgs = parser.parse_args()

    logger.info(f" args set:\n"
                f"small-window-size: {parsedArgs.small_window_size}\n"
                f"big-window-size: {parsedArgs.big_window_size}\n"
                f"h: {parsedArgs.h}\n"
                f"plot: {parsedArgs.plot}\n"
                f"save: {parsedArgs.save}\n"
                f"process mode: single: {parsedArgs.single}, all-data: {parsedArgs.process_all}\n"
                )

    return parsedArgs


def readYamlConfig(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    def normalize(v):
        if isinstance(v, str):
            v = os.path.expandvars(v)
            return Path(v).expanduser()
        return v

    logger.info(f"yaml:\n{yaml.dump({k: str(v) for k, v in cfg.items()}, sort_keys=False)}")
    return {k: normalize(v) for k, v in cfg.items()}


def showGray(img, plotNum, title=""):
    plt.subplot(3, 3, plotNum)
    plt.imshow(img, cmap='gray')
    plt.title(title)


# readImage from path and convert it to gray scale image.
def readImage(path=""):
    logger.info(f"reading image from path: {path}")
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)


def saveImage(img, path=""):
    p = Path(os.path.expandvars(path)).expanduser()
    if p.suffix == "":  # treat as directory
        p = p / "image.png"

    p.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(img, np.ndarray) and img.dtype.kind == "f":
        scale = 255.0 if img.size and img.max() <= 1.0 else 1.0
        img = np.clip(img * scale, 0, 255).astype(np.uint8)

    ok = cv2.imwrite(str(p), img)
    if not ok:
        raise IOError(f"cv2.imwrite failed for '{p}' (unsupported format or invalid image).")
    return p


# patchImage creates a region of interest of the given image.
def patchImage(img, top, bottom, left, right):
    mask = np.zeros_like(img, dtype=bool)
    mask[top:bottom, left:right] = True
    return mask
