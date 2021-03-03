import glob
import numpy as np
import os
import netCDF4 as nc4
import torch
import random
from torch.utils.data import Dataset
from cumulo.data.nc_tile_extractor import sample_random_tiles_from_track

radiances_nc = ['ev_250_aggr1km_refsb_1', 'ev_250_aggr1km_refsb_2', 'ev_1km_emissive_29', 'ev_1km_emissive_33',
                'ev_1km_emissive_34', 'ev_1km_emissive_35', 'ev_1km_emissive_36', 'ev_1km_refsb_26',
                'ev_1km_emissive_27',
                'ev_1km_emissive_20', 'ev_1km_emissive_21', 'ev_1km_emissive_22', 'ev_1km_emissive_23']
coordinates_nc = ['latitude', 'longitude']
properties_nc = ['cloud_water_path', 'cloud_optical_thickness', 'cloud_effective_radius',
                 'cloud_phase_optical_properties',
                 'cloud_top_pressure', 'cloud_top_height', 'cloud_top_temperature', 'cloud_emissivity',
                 'surface_temperature']
rois_nc = 'cloud_mask'
labels_nc = 'cloud_layer_type'


# ------------------------------------------------------------ CUMULO HELPERS

def get_class_occurrences(labels):
    """ 
    Takes in a numpy.ndarray of size (nb_instances, W, H, nb_layers=10) describing for each pixel the types of
    clouds identified at each of the 10 heights and returns a numpy.ndarray of size (nb_points, 8) counting the
    number of times one of the 8 type of clouds was spotted vertically over a whole instance.
    The height information is then lost. 
    """
    occurrences = np.zeros((labels.shape[0], 8))

    for occ, lab in zip(occurrences, labels):

        values, counts = np.unique(lab, return_counts=True)

        for v, c in zip(values, counts):

            if v > -1:  # unlabeled pixels are marked with -1, ignore them
                occ[v] = c
    return occurrences


def get_most_frequent_label(labels):
    """ labels should be of size (nb_instances, ...).

        Returns the most frequent label for each whole instance.
    """
    labels = labels.squeeze()
    mask = np.any(labels != -1, axis=2)
    lpixels = labels[mask]

    rpixels = np.zeros(len(lpixels))

    for ix, lpixel in enumerate(lpixels):
        occ = np.zeros(8)
        uniques, counts = np.unique(lpixel, return_counts=True)
        for v, c in zip(uniques, counts):
            if v != -1:
                occ[v] = c
        rpixels[ix] = np.argmax(occ).astype(float)

    labels_flat = np.ones_like(labels[..., 0]) * -1
    labels_flat[mask] = rpixels

    return labels_flat


def read_nc(nc_file):
    """return masked arrays, with masks indicating the invalid values"""
    file = nc4.Dataset(nc_file, 'r', format='NETCDF4')
    f_radiances = np.vstack([file.variables[name][:] for name in radiances_nc])
    f_rois = file.variables[rois_nc][:]
    f_labels = file.variables[labels_nc][:]
    file.close()
    return f_radiances, f_rois, f_labels


def read_npz(npz_file):
    file = np.load(npz_file)
    return file['radiances'], file['labels']


def include_cloud_mask(labels, cloud_mask):
    labels = labels.copy()
    labels[labels >= 0] += 1
    return labels * cloud_mask


class CumuloDataset(Dataset):

    def __init__(self, d_path, ext="nc", normalizer=None, indices=None, label_preproc=get_low_labels, tiler=None,
                 file_size=1, pred: bool = False, batch_size: int = 1, tile_size: int = 128, center_distance=None,
                 augment_prob: float = 0, offset=0):
        self.root_dir = d_path
        self.ext = ext
        self.file_size = file_size
        self.file_paths = glob.glob(os.path.join(d_path, "*." + ext))

        if len(self.file_paths) == 0:
            raise FileNotFoundError("no {} files in {}".format(ext, d_path))

        if indices is not None:
            self.file_paths = [self.file_paths[i] for i in indices]

        self.normalizer = normalizer
        self.label_preproc = label_preproc
        self.tiler = tiler
        self.pred = pred
        self.batch_size = batch_size
        self.tile_size = tile_size
        self.center_distance = center_distance
        self.augment_prob = augment_prob
        self.offset = offset

    def __len__(self):
        if self.ext in ["npz", "nc"]:
            return len(self.file_paths)

    def __getitem__(self, idx):
        if self.ext == "nc":
            if self.pred:
                # Prediction mode
                radiances, cloud_mask, labels = read_nc(self.file_paths[idx])
                tiles, locations = self.tiler(radiances)
                if self.normalizer is not None:
                    tiles = self.normalizer(tiles)
                if self.label_preproc is not None:
                    labels = labels[..., 0]
                return self.file_paths[idx], tiles, locations, cloud_mask, labels
            else:
                # On-the-fly tile generation
                tiles = None
                next_file = idx
                while tiles is None:
                    # If one nc file has no labeled pixels for the given tile size, another random nc file is used
                    radiances, cloud_mask, labels = read_nc(self.file_paths[next_file])
                    tiles, _ = sample_random_tiles_from_track(radiances, cloud_mask, labels, tile_size=self.tile_size,
                                                              batch_size=self.batch_size,
                                                              center_distance=self.center_distance, offset=self.offset)
                    next_file = np.random.randint(0, len(self.file_paths))
                radiances = np.zeros((self.batch_size, 13, self.tile_size, self.tile_size))
                labels = np.zeros((self.batch_size, self.tile_size, self.tile_size))
                cloud_mask = np.zeros((self.batch_size, self.tile_size, self.tile_size))
                for tile in range(self.batch_size):
                    clabels = tiles[2].data[tile].squeeze()[..., 0]  # take lowest clouds as GT
                    # clabels = get_most_frequent_label(tiles[2].data[tile])
                    cloud_mask[tile] = tiles[1].data[tile].squeeze()
                    # remove labels in non-cloud regions (cloud classes are from 0 to 7, pixels without label are -1)
                    # => add 1, multiply with cloud mask to set labels in non-cloud regions to 0 and substract 1 to get
                    # rid of cloud mask again
                    # clabels[clabels >= 0] += 1
                    # clabels *= cloud_mask[tile].astype(np.int8)
                    # clabels[clabels != -1] -= 1
                    labels[tile] = clabels
                    radiances[tile] = tiles[0].data[tile]
                if self.normalizer is not None:
                    radiances = self.normalizer(radiances)
                if self.augment_prob > 0:
                    for sample in range(len(radiances)):
                        if random.random() < self.augment_prob:
                            radiances[sample] = np.rot90(radiances[sample], axes=(1, 2))
                            labels[sample] = np.rot90(labels[sample])
                            cloud_mask[sample] = np.rot90(cloud_mask[sample])
                return torch.from_numpy(radiances), torch.from_numpy(labels), torch.from_numpy(cloud_mask)
        elif self.ext == "npz":
            radiances, labels = read_npz(self.file_paths[idx])
            if self.normalizer is not None:
                radiances = self.normalizer(radiances)
            if self.label_preproc is not None:
                labels = self.label_preproc(labels)
        return torch.from_numpy(radiances), torch.from_numpy(labels)

    def __str__(self):
        return 'CUMULO'
