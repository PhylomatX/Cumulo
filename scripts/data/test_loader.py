from absl import app
from absl import flags
import time
from cumulo.data.loader import CumuloDataset
import matplotlib.pyplot as plt

flags.DEFINE_string('path', None, help='Directory where nc files are located.')

FLAGS = flags.FLAGS


def main(_):
    dataset = CumuloDataset(FLAGS.path, ext="nc", batch_size=64, tile_size=128, center_distance=0)
    total_time = 0
    for sample in range(len(dataset)):
        start = time.time()
        rads, labs = dataset[sample]
        total_time += time.time() - start
        fig, axs = plt.subplots(1, 2, figsize=(15, 8))
        axs[0].imshow(rads[0][0])
        axs[1].imshow(labs[0])
        plt.show()
        plt.close()

    print(total_time / len(dataset))


if __name__ == '__main__':
    app.run(main)
