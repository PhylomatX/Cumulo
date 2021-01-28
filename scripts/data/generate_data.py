import os
from absl import app
from absl import flags
import numpy as np

flags.DEFINE_string('path', None, help='Location where dataset should be generated')
flags.DEFINE_integer('size', None, help='Number of files')
FLAGS = flags.FLAGS


def main(_):
    for i in range(FLAGS.size):
        rads = np.random.random((16, 13, 128, 128))
        labs = np.random.random((16, 128, 128, 3))
        np.savez(os.path.join(FLAGS.path, f'{i}.npz'), radiances=rads, labels=labs)


if __name__ == '__main__':
    app.run(main)
