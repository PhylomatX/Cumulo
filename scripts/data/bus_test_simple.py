import os
import glob
import numpy as np

if __name__ == '__main__':
    path = '/storage/group/dataset_mirrors/01_incoming/satellite/Cumulo/processed/npz_merged/'
    files = glob.glob(os.path.join(path, f"*.npz"))
    for file in files:
        print(file)
        data = np.load(file)
        radiances = data['radiances']
        labels = data['labels']
