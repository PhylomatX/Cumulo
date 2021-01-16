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


def get_dataset_statistics(dataset, nb_classes, batch_size, tile_size, device='cuda'):
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    weights = np.zeros(nb_classes)
    sum_x = torch.zeros(13)
    std = torch.zeros(1, 13, 1, 1)
    sum_x = sum_x.to(device)
    std = std.to(device)
    nb_tiles = 0

    for tiles, labels in tqdm(dataloader):
        nb_tiles += len(tiles)
        tiles = tiles.to(device)
        # class weights
        weights += np.histogram(labels, bins=range(nb_classes + 1), normed=False)[0]
        sum_x += torch.sum(tiles, (0, 2, 3))

    nb_pixels = nb_tiles * tile_size
    m = (sum_x / nb_pixels).reshape(1, 13, 1, 1)

    for tiles, _ in dataloader:
        tiles = tiles.to(device)
        std += torch.sum((tiles - m).pow(2), (0, 2, 3), keepdim=True)

    s = ((std / nb_pixels) ** 0.5)
    weights = weights / np.sum(weights)
    weights_div = 1 / (np.log(1.02 + weights))
    m = m.cpu()
    s = s.cpu()
    return weights, weights_div, m.reshape(13, 1, 1).numpy(), s.reshape(13, 1, 1).numpy()


class Normalizer(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        return (image - self.mean) / self.std


class TileExtractor(object):

    def __init__(self, t_width=3, t_height=3):

        self.t_width = t_width
        self.t_height = t_height

    def __call__(self, image):

        img_width = image.shape[1]
        img_height = image.shape[2]

        nb_tiles_row = img_width // self.t_width
        nb_tiles_col = img_height // self.t_height

        tiles = []
        locations = []

        for i in range(nb_tiles_row):
            for j in range(nb_tiles_col):
                tile = image[:, i * self.t_width: (i + 1) * self.t_width, j * self.t_height: (j + 1) * self.t_height]

                tiles.append(tile)

                locations.append(
                    ((i * self.t_width, (i + 1) * self.t_width), (j * self.t_height, (j + 1) * self.t_height)))

        tiles = np.stack(tiles)
        locations = np.stack(locations)

        return tiles, locations


# ------------------------------------------------------------ CUMULO HELPERS

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
