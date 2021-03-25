from absl import app
from absl import flags
import numpy as np
import os
from cumulo.data.loader import CumuloDataset
from cumulo.utils.utils import get_dataset_statistics

flags.DEFINE_string('path', None, help='Directory where npz files are located.')
flags.DEFINE_integer('class_num', 9, help='Number of classes.')
flags.DEFINE_integer('tile_size', 128, help='Tile size.')
flags.DEFINE_integer('nb_tiles', None, help='Number of tiles which should be used for the statistics.')
flags.DEFINE_integer('dataset_bs', 64, help='')
flags.DEFINE_integer('center_distance', None, help='')
flags.DEFINE_string('filetype', 'nc', help='')

FLAGS = flags.FLAGS


def main(_):
    dataset = CumuloDataset(FLAGS.path, batch_size=FLAGS.dataset_bs, tile_size=FLAGS.tile_size)
    weights, class_weights, m, std = get_dataset_statistics(dataset, FLAGS.class_num, tile_size=FLAGS.tile_size, nb_samples=FLAGS.nb_tiles)
    print(weights)
    print(class_weights)
    print(m.squeeze())
    print(std.squeeze())
    np.save(os.path.join(FLAGS.path, "class-weights.npy"), class_weights)
    np.save(os.path.join(FLAGS.path, "mean.npy"), m)
    np.save(os.path.join(FLAGS.path, "std.npy"), std)


if __name__ == '__main__':
    app.run(main)
