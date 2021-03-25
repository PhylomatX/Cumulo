import os
from absl import app
from absl import flags
from .utils import *


flags.DEFINE_string('path', None, help='Location of predictions')
flags.DEFINE_boolean('save', False, help='Save predictions as images')
flags.DEFINE_boolean('continuous_colors', True, help='Use predictions for color weighting')
flags.DEFINE_boolean('cloud_mask_as_binary', True, help='Make cloud mask binary')
FLAGS = flags.FLAGS


def main(_):
    files = list(filter(lambda f: 'npz' in f, os.listdir(FLAGS.path)))
    for file in files:
        print(file)
        data = np.load(os.path.join(FLAGS.path, file))
        prediction = prediction_from_outputs(data['outputs'])
        if FLAGS.continuous_colors:
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
