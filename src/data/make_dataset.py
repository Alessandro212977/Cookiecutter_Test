# -*- coding: utf-8 -*-
import logging
import os
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from torch.utils.data.dataset import TensorDataset


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    mnist(input_filepath, output_filepath)


def mnist(input_filepath, output_filepath):
    # exchange with the corrupted mnist dataset

    # TRAIN
    rel_path = "corruptmnist/train_0.npz"
    with np.load(os.path.join(input_filepath, rel_path)) as data:
        imgs, labels = data["images"], data["labels"]

    for i in range(1, 4):
        rel_path = "corruptmnist/train_{}.npz".format(i)

        with np.load(os.path.join(input_filepath, rel_path)) as data:
            imgs = np.concatenate((imgs, data["images"]))
            labels = np.concatenate((labels, data["labels"]))

    torch_imgs, torch_labels = (
        torch.from_numpy(imgs).unsqueeze(1),
        torch.from_numpy(labels),
    )
    train = TensorDataset(torch_imgs.float(), torch_labels)

    # TEST
    rel_path = "corruptmnist/test.npz"
    with np.load(os.path.join(input_filepath, rel_path)) as data:
        imgs, labels = data["images"], data["labels"]

    torch_imgs, torch_labels = (
        torch.from_numpy(imgs).unsqueeze(1),
        torch.from_numpy(labels),
    )
    test = TensorDataset(torch_imgs.float(), torch_labels)

    torch.save(train, os.path.join(output_filepath, "train_dataset.pt"))
    torch.save(test, os.path.join(output_filepath, "test_dataset.pt"))


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
