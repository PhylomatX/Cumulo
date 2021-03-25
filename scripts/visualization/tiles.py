import os
import re
from absl import app
from absl import flags
from .utils import *


flags.DEFINE_string('path', None, help='Directory where npz files are located.')
flags.DEFINE_string('type', 'train', help='Type of example (val or train)')
flags.DEFINE_list('epoch_interval', [0, 100], help='Only training examples from epochs within this interval are selected')
flags.DEFINE_boolean('cloud_mask_as_binary', True, help='Make cloud mask binary')
flags.DEFINE_boolean('save', False, help='Save examples as images')
FLAGS = flags.FLAGS


def main(_):
    files = os.listdir(FLAGS.path)
    file_dict = {}
    epoch_interval = list(map(int, FLAGS.epoch_interval))
    for file in files:
        epoch = int(re.findall(r"(\d+)", file)[0])
        if epoch in range(epoch_interval[0], epoch_interval[1]) and FLAGS.type in file:
            file_dict[epoch] = file
    epochs = list(file_dict.keys())
    epochs.sort()
    for epoch in epochs:
        file = file_dict[epoch]
        data = np.load(os.path.join(FLAGS.path, file))

        prediction = prediction_from_outputs(data['outputs'])
        if FLAGS.continuous:
            prediction = prediction_to_continuous_rgb(prediction, FLAGS.cloud_mask_as_binary)
        else:
            prediction = prediction_to_discrete_rgb(prediction)

        ground_truth = labels_and_cloud_mask_to_rgb(data['labels'], data['cloud_mask'])

        if FLAGS.save:
            file = os.path.join(FLAGS.path, file)
            prediction_to_file(file, prediction, ground_truth, FLAGS.cloud_mask_as_binary)
        else:
            prediction_to_figure(prediction, ground_truth, FLAGS.cloud_mask_as_binary)
            plt.show()
            plt.close()


if __name__ == '__main__':
    app.run(main)
