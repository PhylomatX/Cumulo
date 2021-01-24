import os
import pickle as pkl
from tqdm import tqdm
from cumulo.data.loader import TestDataset
from absl import app
from absl import flags

flags.DEFINE_string('npz_path', None, help='Location of npz files')
flags.DEFINE_string('out_path', None, help='Directory where merged tiles should be saved.')
flags.DEFINE_integer('iters', 20, help='Number of iterations')
flags.DEFINE_integer('fnum', None, help='Number of files')
FLAGS = flags.FLAGS


def bus_test(npz_path: str, out_path: str):
    dataset = TestDataset(npz_path)
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
            rads, labs, file = dataset[i]
            print(file)


def main(_):
    bus_test(FLAGS.npz_path, FLAGS.out_path)


if __name__ == '__main__':
    app.run(main)
