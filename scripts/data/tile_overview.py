import os
from absl import app
from absl import flags
from cumulo.data.loader import CumuloDataset
import matplotlib.pyplot as plt

flags.DEFINE_string('save_path', None, help='Directory where images should get saved.')
FLAGS = flags.FLAGS


def main(_):
    load_path = FLAGS.data_path
    dataset = CumuloDataset(load_path, ext="npz", label_preproc=None)

    for instance in dataset:
        filename, radiances, properties, rois, labels = instance

        tile_num = radiances.shape[0]
        size = radiances.shape[-1]

        if not os.path.exists(FLAGS.save_path + str(size)):
            os.makedirs(FLAGS.save_path + str(size))

        for tile in range(0, tile_num, 20):
            fig = plt.figure(figsize=(10, 10))
            gs = fig.add_gridspec(3, 3, hspace=0, wspace=0)
            axs = gs.subplots(sharex='col', sharey='row')

            axs[0, 0].imshow(labels[tile, 0, ..., 0])
            axs[0, 1].imshow(labels[tile, 0, ..., 1])
            axs[0, 2].imshow(rois[tile, 0, ...])
            for i in range(3):
                axs[1, i].imshow(radiances[tile, i+6, ...])
                axs[2, i].imshow(radiances[tile, i+9, ...])

            plt.savefig(FLAGS.save_path + str(size) + f'/{tile}.png')
            plt.close(fig)


if __name__ == '__main__':
    app.run(main)
