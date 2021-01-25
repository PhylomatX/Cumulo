import os
import glob
import pickle as pkl
from tqdm import tqdm
from absl import app
from absl import flags
import numpy as np

flags.DEFINE_string('path', None, help='Location of files')
flags.DEFINE_string('ext', 'npz', help='Extension of files')
flags.DEFINE_string('out_path', None, help='Directory where merged tiles should be saved.')
flags.DEFINE_integer('iters', 20, help='Number of iterations')
flags.DEFINE_integer('fnum', None, help='Number of files')
flags.DEFINE_integer('stop', None, help='Step for debugging')
FLAGS = flags.FLAGS


def read_npz(npz_file):
    file = np.load(npz_file)
    return file['radiances'], file['labels']


class TestDataset:

    def __init__(self, d_path: str, ext: str = "npz"):
        self.root_dir = d_path
        self.file_paths = glob.glob(os.path.join(d_path, f"*.{ext}"))
        self.ext = ext

    def get_files(self):
        return self.file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        filename = self.file_paths[idx]
        if self.ext == "npz":
            radiances, labels = read_npz(filename)
        elif self.ext == "pkl":
            with open(filename, 'rb') as f:
                sample = pkl.load(f)
            radiances = sample[0]
            labels = sample[1]
        return radiances, labels[..., 0], filename


def bus_test(path: str, out_path: str):
    dataset = TestDataset(path, ext=FLAGS.ext)
    files = dataset.get_files()
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    with open(os.path.join(out_path, 'files.pkl'), 'wb') as f:
        pkl.dump(files, f)

    if FLAGS.fnum is None:
        fnum = len(dataset)
    else:
        fnum = FLAGS.fnum

    for j in range(FLAGS.iters):
        for i in tqdm(range(fnum)):
            if FLAGS.stop is not None and i == FLAGS.stop:
                import ipdb
                ipdb.set_trace()
            rads, labs, file = dataset[i]
            print(file)


def main(_):
    bus_test(FLAGS.path, FLAGS.out_path)


if __name__ == '__main__':
    app.run(main)
