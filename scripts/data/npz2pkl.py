import os
from glob import glob
import pickle as pkl
from tqdm import tqdm
from cumulo.data.loader import read_npz
from absl import app
from absl import flags

flags.DEFINE_string('npz_path', None, help='Location of npz files')
flags.DEFINE_string('out_path', None, help='Directory where merged tiles should be saved.')
flags.DEFINE_boolean('remove', False, help='Remove npz files after tiles have been merged.')
FLAGS = flags.FLAGS


def npz2pkl(npz_path: str, out_path: str, remove: bool = False):
    files = glob(os.path.join(npz_path, '*.npz'))
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    for i in tqdm(range(len(files))):
        radiances, labels = read_npz(files[i])
        if remove:
            os.remove(files[i])
        with open(out_path + f'{i}.pkl', 'wb') as f:
            pkl.dump([radiances, labels], f)


def main(_):
    npz2pkl(FLAGS.npz_path, FLAGS.out_path, remove=FLAGS.remove)


if __name__ == '__main__':
    app.run(main)
