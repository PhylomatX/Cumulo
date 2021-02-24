import os
import re
from absl import app
from absl import flags
import numpy as np
import matplotlib.pyplot as plt

flags.DEFINE_string('path', None, help='Directory where npz files are located.')
flags.DEFINE_string('type', 'train', help='Type of example (val or train)')
flags.DEFINE_list('sample_interval', [0, 6], help='Interval of files to show from each example.')
flags.DEFINE_list('epoch_interval', [0, 100], help='Interval of files to show from each example.')
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
        outputs = data['outputs']
        cloud_mask = data['cloud_mask']
        for i in range(sample_interval[0], sample_interval[1]):
            fig, axs = plt.subplots(1, 3, figsize=(20, 10))
            axs[0].imshow(outputs[i])
            axs[1].imshow(labels[i])
            axs[2].imshow(cloud_mask[i])
            axs[0].set_title(f'output - {file}_{i}')
            axs[1].set_title('label')
            axs[2].set_title('cloud_mask')
            plt.show()


if __name__ == '__main__':
    app.run(main)
