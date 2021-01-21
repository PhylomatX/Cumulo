import os
import h5py
from glob import glob
import numpy as np
from cumulo.data.loader import read_npz


def npz2hdf5(npz_path: str, hdf5_path: str, size: int = 1000):
    files = glob(os.path.join(npz_path, '*.npz'))
    radiances, labels = read_npz(files[0])
    if not os.path.exists(hdf5_path):
        os.makedirs(hdf5_path)
    for i in range(int(len(files)/size)+1):
        t_radiances = np.zeros((size, *radiances.shape))
        t_labels = np.zeros((size, *labels.shape))
        for j in range(size):
            if j >= len(files):
                break
            radiances, labels = read_npz(files[i * size + j])
            t_radiances[j] = radiances
            t_labels[j] = labels
        store_hdf5(t_radiances, t_labels, os.path.join(hdf5_path, f's{size}_{i}.h5'))


def store_hdf5(radiances: np.ndarray, labels: np.ndarray, hdf5_path: str):
    file = h5py.File(hdf5_path, 'w')
    file.create_dataset('radiances', np.shape(radiances), h5py.h5t.IEEE_F32BE, data=radiances)
    file.create_dataset('labels', np.shape(labels), h5py.h5t.STD_I8BE, data=labels)
    file.close()


if __name__ == '__main__':
    npz_p = '/home/john/Projekte/BAP/Data/Cumulo/npz/'
    hdf5_p = '/home/john/Projekte/BAP/Data/Cumulo/hdf5/'
    npz2hdf5(npz_p, hdf5_p, 1000)
