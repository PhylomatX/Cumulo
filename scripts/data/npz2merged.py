import os
from glob import glob
import numpy as np
import random
from tqdm import tqdm
from cumulo.data.loader import read_npz
from absl import app
from absl import flags

flags.DEFINE_string('npz_path', None, help='Location of npz files')
flags.DEFINE_string('out_path', None, help='Directory where merged tiles should be saved.')
flags.DEFINE_integer('size', 16, help='Number of tiles which should get merged into 1 file')
flags.DEFINE_boolean('remove', False, help='Remove npz files after tiles have been merged.')
FLAGS = flags.FLAGS


def npz2merged(npz_path: str, out_path: str, size: int = 16, remove: bool = False):
    files = glob(os.path.join(npz_path, '*.npz'))
    radiances, labels = read_npz(files[0])
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    random.shuffle(files)
    for i in tqdm(range(int(len(files)/size))):
        t_radiances = np.zeros((size, *radiances.shape))
        t_labels = np.zeros((size, *labels.shape))
        for j in range(size):
            radiances, labels = read_npz(files[i * size + j])
            if remove:
                os.remove(files[i * size + j])
            t_radiances[j] = radiances
            t_labels[j] = labels
        np.savez(out_path + f'{i}_s{size}', radiances=t_radiances, labels=t_labels)


def main(_):
    npz2merged(FLAGS.npz_path, FLAGS.out_path, size=FLAGS.size, remove=FLAGS.remove)


if __name__ == '__main__':
    app.run(main)
