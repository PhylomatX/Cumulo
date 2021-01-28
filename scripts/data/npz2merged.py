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
flags.DEFINE_integer('nsize', 16, help='Number of files which should get merged into 1 file')
flags.DEFINE_boolean('remove', False, help='Remove npz files after tiles have been merged.')
flags.DEFINE_integer('csize', 1, help='Current size of files.')
FLAGS = flags.FLAGS


def npz2merged(npz_path: str, out_path: str, nsize: int = 16, csize: int = 1, remove: bool = False):
    files = glob(os.path.join(npz_path, '*.npz'))
    radiances, labels = read_npz(files[0])
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    random.shuffle(files)
    for i in tqdm(range(int(len(files)/nsize))):
        if csize > 1:
            idim = 1
        else:
            idim = 0
        t_radiances = np.zeros((nsize*csize, *radiances.shape[idim:]))
        t_labels = np.zeros((nsize*csize, *labels.shape[idim:]))
        for j in range(nsize):
            radiances, labels = read_npz(files[i * nsize + j])
            if remove:
                os.remove(files[i * nsize + j])
            t_radiances[j*csize:j*csize+csize] = radiances
            t_labels[j*csize:j*csize+csize] = labels
        np.savez(out_path + f'{i}_s{nsize}', radiances=t_radiances, labels=t_labels)


def main(_):
    npz2merged(FLAGS.npz_path, FLAGS.out_path, nsize=FLAGS.nsize, csize=FLAGS.csize, remove=FLAGS.remove)


if __name__ == '__main__':
    app.run(main)
