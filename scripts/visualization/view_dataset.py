from absl import app
from absl import flags
from cumulo.data.loader import CumuloDataset
import matplotlib.pyplot as plt

flags.DEFINE_string('npz_path', None, help='Directory where npz files are located.')
FLAGS = flags.FLAGS


def main(_):
    dataset = CumuloDataset(FLAGS.npz_path, ext="npz")
    print(len(dataset))

    for sample in range(len(dataset)):
        rads, labs = dataset[sample]
        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(2, 2, hspace=0, wspace=0)
        axs = gs.subplots(sharex='col', sharey='row')
        axs[0, 0].imshow(labs)
        axs[0, 1].imshow(rads[0])
        axs[1, 0].imshow(rads[1])
        axs[1, 1].imshow(rads[2])
        plt.show()
        plt.close()


if __name__ == '__main__':
    app.run(main)
