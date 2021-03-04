import os
import re
from absl import app
from absl import flags
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from cumulo.utils.utils import include_cloud_mask

flags.DEFINE_string('path', None, help='Directory where npz files are located.')
flags.DEFINE_string('type', 'train', help='Type of example (val or train)')
flags.DEFINE_list('sample_interval', [0, 6], help='Interval of files to show from each example.')
flags.DEFINE_list('epoch_interval', [0, 100], help='Interval of files to show from each example.')
flags.DEFINE_boolean('save', False, help='Save examples as images')
FLAGS = flags.FLAGS


def main(_):
    files = os.listdir(FLAGS.path)
    file_dict = {}
    epoch_interval = list(map(int, FLAGS.epoch_interval))
    sample_interval = list(map(int, FLAGS.sample_interval))
    for file in files:
        epoch = int(re.findall(r"(\d+)", file)[0])
        if epoch in range(epoch_interval[0], epoch_interval[1]) and FLAGS.type in file:
            file_dict[epoch] = file
    epochs = list(file_dict.keys())
    epochs.sort()
    for epoch in epochs:
        file = file_dict[epoch]
        data = np.load(os.path.join(FLAGS.path, file))
        labels = data['labels']
        outputs = data['predictions'] + 1
        # cloud_mask = data['cloud_mask']
        # merged = include_cloud_mask(labels, cloud_mask)
        merged = labels
        cmap = colors.ListedColormap(['k', 'whitesmoke', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                                      'tab:brown', 'tab:pink', 'dimgrey', 'tab:olive', 'tab:cyan'])
        for i in range(sample_interval[0], sample_interval[1]):
            if not FLAGS.save:
                fig, axs = plt.subplots(1, 2, figsize=(20, 10))
                axs[0].imshow(outputs[i], vmin=0, vmax=10, cmap=cmap)
                axs[1].imshow(merged[i]+1, vmin=0, vmax=10, cmap=cmap)
                axs[0].set_title(f'output - {file}_{i}')
                axs[1].set_title('label')
                plt.show()
            else:
                figsize = (15, 12)
                fig, axs = plt.subplots(1, 1, figsize=figsize)
                plt.title('Prediction')
                plt.imshow(outputs[i], vmin=0, vmax=10, cmap=cmap)
                plt.axis('off')
                plt.savefig(os.path.join(FLAGS.path, f'epoch_{epoch}_{FLAGS.type}_pred.png'))
                plt.close()
                fig, axs = plt.subplots(1, 1, figsize=figsize)
                plt.title('Ground Truth')
                plt.imshow(merged[i]+1, vmin=0, vmax=10, cmap=cmap)
                plt.axis('off')
                plt.savefig(os.path.join(FLAGS.path, f'epoch_{epoch}_{FLAGS.type}_gt.png'))
                plt.close()


if __name__ == '__main__':
    app.run(main)
