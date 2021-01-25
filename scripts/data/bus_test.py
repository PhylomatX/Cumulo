import os
import pickle as pkl
from tqdm import tqdm
from cumulo.data.loader import TestDataset
from absl import app
from absl import flags

flags.DEFINE_string('path', None, help='Location of files')
flags.DEFINE_string('ext', 'npz', help='Extension of files')
flags.DEFINE_string('out_path', None, help='Directory where merged tiles should be saved.')
flags.DEFINE_integer('iters', 20, help='Number of iterations')
flags.DEFINE_integer('fnum', None, help='Number of files')
flags.DEFINE_integer('stop', None, help='Step for debugging')
FLAGS = flags.FLAGS


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
