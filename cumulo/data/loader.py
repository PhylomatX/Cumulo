import glob
import numpy as np
import os
import torch
import random
from torch.utils.data import Dataset
from cumulo.utils.training import sample_n_tiles_with_labels
from cumulo.utils.basics import read_nc
from cumulo.utils.evaluation import divide_into_tiles


class CumuloDataset(Dataset):

    def __init__(self, d_path, normalizer=None, indices=None, prediction_mode: bool = False,
                 batch_size: int = 1, tile_size: int = 256, rotation_probability: float = 0,
                 valid_convolution_offset=0, most_frequent_clouds_as_GT=False, exclude=None,
                 filter_cloudy_labels=True):
        self.root_dir = d_path
        self.file_paths = glob.glob(os.path.join(d_path, "*.nc"))
        self.normalizer = normalizer
        self.prediction_mode = prediction_mode
        self.batch_size = batch_size
        self.tile_size = tile_size
        self.rotation_probability = rotation_probability
        self.valid_convolution_offset = valid_convolution_offset
        self.most_frequent_clouds_as_GT = most_frequent_clouds_as_GT
        self.filter_cloudy_labels = filter_cloudy_labels

        if len(self.file_paths) == 0:
            raise FileNotFoundError("Dataloader found no files.")

        if exclude is None:
            exclude = []

        if indices is not None:
            self.file_paths = [self.file_paths[i] for i in indices if self.file_paths[i][-17:] not in exclude]

    def __len__(self):
        return len(self.file_paths)

    # noinspection PyUnboundLocalVariable
    def __getitem__(self, idx):
        """
        In prediction mode, this function divides a swath into tiles which fully cover the swath. In training mode,
        it draws `self.batch_size` (randomly shifted) tiles from the current nc file. nc file loading is costly. This
        method has been tested against generating and saving the tiles as npz files before training. Loading nc files
        directly was equally fast and provides more flexibility.

        When using this dataset with multiple parallel workers, each worker opens only one nc file, draws multiple tiles
        and therefore returns a batch of tiles.
        """
        if self.prediction_mode:
            radiances, cloud_mask, labels = read_nc(self.file_paths[idx], filter_most_freqent=self.most_frequent_clouds_as_GT)
            radiances, locations = divide_into_tiles(self.tile_size, self.valid_convolution_offset, radiances)
            if self.normalizer is not None:
                radiances = self.normalizer(radiances)
            return self.file_paths[idx], radiances, locations, cloud_mask, labels
        else:
            radiances = None
            next_file = idx
            while radiances is None:
                # --- If one nc file has no labeled pixels for the given tile size, another random nc file is used ---
                radiances, cloud_mask, labels = read_nc(self.file_paths[next_file], filter_most_freqent=self.most_frequent_clouds_as_GT)
                radiances, cloud_mask, labels = sample_n_tiles_with_labels(radiances, cloud_mask, labels, tile_size=self.tile_size, n=self.batch_size,
                                                                           valid_convolution_offset=self.valid_convolution_offset,
                                                                           filter_cloudy_labels=self.filter_cloudy_labels)
                next_file = np.random.randint(0, len(self.file_paths))
            if self.normalizer is not None:
                radiances = self.normalizer(radiances)
            for sample in range(len(radiances)):
                if random.random() < self.rotation_probability:
                    radiances[sample] = np.rot90(radiances[sample], axes=(1, 2))
                    cloud_mask[sample] = np.rot90(cloud_mask[sample])
                    labels[sample] = np.rot90(labels[sample])
            return torch.from_numpy(radiances), torch.from_numpy(labels), torch.from_numpy(cloud_mask)
