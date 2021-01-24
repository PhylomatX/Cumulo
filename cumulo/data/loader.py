import glob
import numpy as np
import os
import netCDF4 as nc4
import torch
import h5py
from torch.utils.data import Dataset

radiances = ['ev_250_aggr1km_refsb_1', 'ev_250_aggr1km_refsb_2', 'ev_1km_emissive_29', 'ev_1km_emissive_33',
             'ev_1km_emissive_34', 'ev_1km_emissive_35', 'ev_1km_emissive_36', 'ev_1km_refsb_26', 'ev_1km_emissive_27',
             'ev_1km_emissive_20', 'ev_1km_emissive_21', 'ev_1km_emissive_22', 'ev_1km_emissive_23']
coordinates = ['latitude', 'longitude']
properties = ['cloud_water_path', 'cloud_optical_thickness', 'cloud_effective_radius', 'cloud_phase_optical_properties',
              'cloud_top_pressure', 'cloud_top_height', 'cloud_top_temperature', 'cloud_emissivity',
              'surface_temperature']
rois = 'cloud_mask'
labels = 'cloud_layer_type'


# ------------------------------------------------------------ CUMULO HELPERS

def get_class_occurrences(labels):
    """ 
    Takes in a numpy.ndarray of size (nb_instances, W, H, nb_layers=10) describing for each pixel the types of clouds identified at each of the 10 heights and returns a numpy.ndarray of size (nb_points, 8) counting the number of times one of the 8 type of clouds was spotted vertically over a whole instance.
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

    f_radiances = np.vstack([file.variables[name][:] for name in radiances])
    f_properties = np.vstack([file.variables[name][:] for name in properties])
    f_rois = file.variables[rois][:]
    f_labels = file.variables[labels][:]

    return f_radiances, f_properties, f_rois, f_labels


def read_npz(npz_file):
    file = np.load(npz_file)
    return file['radiances'], file['labels']


def get_mf_label(labels):
    """ See tile generation method. """
    return labels[..., 0]


def get_low_labels(labels):
    """ See tile generation method. """
    return labels[..., 1]


def get_low_labels_raw(labels):
    """ See tile generation method. """
    return labels[..., 2]


class CumuloDataset(Dataset):

    def __init__(self, d_path, ext="npz", normalizer=None, indices=None, label_preproc=get_low_labels, tiler=None, file_size=1):

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

    def __len__(self):
        if self.ext == "npz":
            return len(self.file_paths)
        elif self.ext == "h5":
            return len(self.file_paths) * self.file_size

    def __getitem__(self, idx):
        if self.ext == "nc":
            radiances, properties, rois, labels = read_nc(self.file_paths[idx])
            tiles, locations = self.tiler(radiances)
            return self.file_paths[idx], tiles, locations, rois, labels
        elif self.ext == "npz":
            radiances, labels = read_npz(self.file_paths[idx])
        elif self.ext == "h5":
            file_num = int(idx / self.file_size)
            file_ix = idx % self.file_size
            file = h5py.File(self.file_paths[file_num], "r+")
            radiances = np.array(file["/radiances"]).astype(np.float32)[file_ix]
            labels = np.array(file["/labels"]).astype(np.int8)[file_ix]

        if self.normalizer is not None:
            radiances = self.normalizer(radiances)

        if self.label_preproc is not None:
            labels = self.label_preproc(labels)

        return torch.from_numpy(radiances), torch.from_numpy(labels)

    def __str__(self):
        return 'CUMULO'


class TestDataset(Dataset):

    def __init__(self, d_path: str):
        self.root_dir = d_path
        self.file_paths = glob.glob(os.path.join(d_path, "*.npz"))

    def get_files(self):
        return self.file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        filename = self.file_paths[idx]
        radiances, labels = read_npz(filename)
        return radiances, labels[..., 0], self.file_paths[idx]


class TestDatasetTorch(Dataset):
    def __init__(self, d_path: str):
        self.dataset = TestDataset(d_path)

    def get_files(self):
        return self.dataset.get_files()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        rads, labs, file = self.dataset[idx]
        return torch.from_numpy(rads), torch.from_numpy(labs)
