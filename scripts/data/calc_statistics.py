from absl import app
from absl import flags
import numpy as np
import os
from cumulo.data.loader import CumuloDataset
from cumulo.utils.utils import get_dataset_statistics

flags.DEFINE_string('npz_path', None, help='Directory where npz files are located.')
flags.DEFINE_integer('class_num', 9, help='Number of classes.')
flags.DEFINE_integer('tile_size', 128, help='Tile size.')

FLAGS = flags.FLAGS


def main(_):
    dataset = CumuloDataset(FLAGS.npz_path, ext="npz")
    weights, class_weights, m, s = get_dataset_statistics(dataset, FLAGS.class_num, batch_size=40, tile_size=FLAGS.tile_size, device='cpu')
    print(weights)
    print(class_weights)
    np.save(os.path.join(FLAGS.npz_path, "class-weights.npy"), class_weights)
    np.save(os.path.join(FLAGS.npz_path, "mean.npy"), m)
    np.save(os.path.join(FLAGS.npz_path, "std.npy"), s)


if __name__ == '__main__':
    app.run(main)
