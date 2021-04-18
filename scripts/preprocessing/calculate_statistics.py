from absl import app
from absl import flags
import numpy as np
import os
from cumulo.data.loader import CumuloDataset
from cumulo.utils.training import get_dataset_statistics

flags.DEFINE_string('path', None, help='Directory where npz files are located.')
flags.DEFINE_integer('class_num', 9, help='Number of classes.')
flags.DEFINE_integer('tile_size', 256, help='Tile size.')
flags.DEFINE_integer('sample_number', None, help='Number of nc files which should be used for the statistics.')
flags.DEFINE_integer('tile_number', 16, help='Number of tiles drawn from each nc file in the dataset')

FLAGS = flags.FLAGS


def main(_):
    dataset = CumuloDataset(FLAGS.path, batch_size=FLAGS.tile_number, tile_size=FLAGS.tile_size)
    weights, class_weights, m, std = get_dataset_statistics(dataset, FLAGS.class_num, tile_size=FLAGS.tile_size, sample_number=FLAGS.sample_number)
    print(weights)
    print(class_weights)
    print(m.squeeze())
    print(std.squeeze())
    np.save(os.path.join(FLAGS.path, "class-weights.npy"), class_weights)
    np.save(os.path.join(FLAGS.path, "mean.npy"), m)
    np.save(os.path.join(FLAGS.path, "std.npy"), std)


if __name__ == '__main__':
    app.run(main)
