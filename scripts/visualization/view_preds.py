import os
import numpy as np
import matplotlib.pyplot as plt
from absl import app
from absl import flags

flags.DEFINE_string('p_path', None, help='Location of predictions')
FLAGS = flags.FLAGS


def main(_):
    files = os.listdir(FLAGS.p_path)
    for file in files:
        data = np.load(os.path.join(FLAGS.p_path, file))
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].imshow(data['prediction'])
        axs[1].imshow(data['labels'])
        plt.show()
        plt.close()


if __name__ == '__main__':
    app.run(main)
