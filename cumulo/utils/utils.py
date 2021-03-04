import netCDF4 as nc4
import numpy as np
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from cumulo.data.loader import read_npz

import datetime


def get_datetime(year, day, hour=0, minute=0, second=0):
    """ Returns month and day given a day of a year"""

    dt = datetime.datetime(year, 1, 1, hour, minute, second) + datetime.timedelta(days=day - 1)
    return dt


def get_file_time_info(radiance_filename, split_char='MYD021KM.A'):
    time_info = radiance_filename.split(split_char)[1]
    year, abs_day = time_info[:4], time_info[4:7]
    hour, minute = time_info[8:10], time_info[10:12]

    return year, abs_day, hour, minute


def minutes_since(year, abs_day, hour, minute, ref_year=2008, ref_abs_day=1, ref_hour=0, ref_minute=0, ref_second=0):
    dt = get_datetime(year, abs_day, hour, minute)
    ref_dt = get_datetime(ref_year, ref_abs_day, ref_hour, ref_minute, ref_second)

    return int((dt - ref_dt).total_seconds() // 60)


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return h, m, s


def make_directory(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def include_cloud_mask(labels, cloud_mask):
    labels[labels >= 0] += 1
    return labels * cloud_mask


def get_dataset_statistics(dataset, nb_classes, tile_size, nb_samples=None):
    weights = np.zeros(nb_classes)
    m = np.zeros(13)
    s = np.zeros(13)
    if nb_samples is None:
        nb_samples = len(dataset)
    nb_tiles = nb_samples
    rads, labels, _ = dataset[0]
    if len(rads.shape) == 4:
        nb_tiles *= rads.shape[0]

    batch_size = 1
    for sample in tqdm(range(nb_samples)):
        rads, labels, _ = dataset[sample]
        if len(rads.shape) == 4:
            batch_size = rads.shape[0]
        for i in range(batch_size):
            crads = rads[i].numpy()
            clabels = labels[i].numpy()
            weights += np.histogram(clabels, bins=range(nb_classes + 1), normed=False)[0]
            m += np.mean(crads, axis=(1, 2))
        print(f"Sample: {sample}")
        print(f"weights: {weights / np.sum(weights)}")
        print(f"mean: {m / (sample * rads.shape[0])}")

    m /= nb_tiles
    m = m.reshape((13, 1, 1))

    for sample in tqdm(range(nb_samples)):
        rads, labels = dataset[sample]
        for i in range(batch_size):
            crads = rads[i].numpy()
            s += np.sum((crads - m)**2, (1, 2))
        print(f"weights: {weights / np.sum(weights)}")
        print(f"mean: {m}")

    s /= nb_tiles * tile_size ** 2
    std = np.sqrt(s)
    std = std.reshape((13, 1, 1))
    weights = weights / np.sum(weights)
    weights_div = 1 / (np.log(1.02 + weights))
    return weights, weights_div, m, std


class GlobalNormalizer(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        return (image - self.mean) / self.std


class LocalNormalizer(object):

    def __call__(self, image):
        for b in range(image.shape[0]):
            means = np.mean(image[b], axis=(1, 2)).reshape((image[b].shape[0], 1, 1))
            stds = np.std(image[b], axis=(1, 2)).reshape((image[b].shape[0], 1, 1))
            image[b] = (image[b] - means) / stds
        return image


class TileExtractor:

    def __init__(self, tile_size, offset):
        """
        Args:
             tile_size: size of tiles (size of input to network)
             offset: size difference between input and output of network (e.g. due to valid convolutions)
        """

        self.tile_size = tile_size
        self.offset = offset

    def __call__(self, radiances, labels):
        """
        Can be used to split a swath into multiple tiles.

        Args:
            radiances: channels of full swath
            labels: merge result of sparse cloud class GT and dense cloud mask GT

        Returns:
            tiles and locations where locations are the positions of the output of the network
            (including the possible offset, see initialization).
        """

        img_width = radiances.shape[1]
        img_height = radiances.shape[2]

        output_size = self.tile_size - 2 * self.offset
        nb_outputs_row = (img_width - 2 * self.offset) // output_size
        nb_outputs_col = (img_height - 2 * self.offset) // output_size

        tiles = []
        label_tiles = []
        locations = []

        for i in range(nb_outputs_row):
            for j in range(nb_outputs_col):
                tiles.append(radiances[:, i * output_size: 2 * self.offset + (i + 1) * output_size, j * output_size: 2 * self.offset + (j + 1) * output_size])
                label_tiles.append(labels[:, i * output_size: 2 * self.offset + (i + 1) * output_size, j * output_size: 2 * self.offset + (j + 1) * output_size])
                locations.append(((self.offset + i * output_size, self.offset + (i + 1) * output_size),
                                  (self.offset + j * output_size, self.offset + (j + 1) * output_size)))

        # gather tiles from border regions
        for i in range(nb_outputs_row):
            tiles.append(radiances[:, i * output_size: 2 * self.offset + (i + 1) * output_size, img_height - self.tile_size:img_height])
            label_tiles.append(labels[:, i * output_size: 2 * self.offset + (i + 1) * output_size, img_height - self.tile_size:img_height])
            locations.append(((self.offset + i * output_size, self.offset + (i + 1) * output_size),
                              (self.offset + img_height - self.tile_size, img_height - self.offset)))

        for j in range(nb_outputs_col):
            tiles.append(radiances[:, img_width - self.tile_size:img_width, j * output_size: 2 * self.offset + (j + 1) * output_size])
            label_tiles.append(labels[:, img_width - self.tile_size:img_width, j * output_size: 2 * self.offset + (j + 1) * output_size])
            locations.append(((self.offset + img_width - self.tile_size, img_width - self.offset),
                              (self.offset + j * output_size, self.offset + (j + 1) * output_size)))

        tiles.append(radiances[:, img_width - self.tile_size:img_width, img_height - self.tile_size:img_height])
        label_tiles.append(labels[:, img_width - self.tile_size:img_width, img_height - self.tile_size:img_height])
        locations.append(((self.offset + img_width - self.tile_size, img_width - self.offset),
                          (self.offset + img_height - self.tile_size, img_height - self.offset)))

        tiles = np.stack(tiles)
        label_tiles = np.stack(label_tiles)
        locations = np.stack(locations)

        return tiles, label_tiles, locations


def get_tile_sampler(dataset, allowed_idx=None, ext="npz"):
    indices = []
    paths = dataset.file_paths.copy()

    if allowed_idx is not None:
        paths = [paths[i] for i in allowed_idx]

    for i, swath_path in enumerate(paths):
        swath, *_ = read_npz(swath_path)

        indices += [(i, j) for j in range(swath.shape[0])]

    return SubsetRandomSampler(indices)


def tile_collate(swath_tiles):
    data = np.vstack([tiles for _, tiles, _, _, _ in swath_tiles])
    target = np.hstack([labels for *_, labels in swath_tiles])

    return torch.from_numpy(data).float(), torch.from_numpy(target).long()

