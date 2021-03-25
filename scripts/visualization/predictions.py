import os
import numpy as np
from absl import app
from cumulo.utils.visualization import outputs_to_figure_or_file
from flags import FLAGS


def main(_):
    files = list(filter(lambda f: 'npz' in f, os.listdir(FLAGS.path)))
    for file in files:
        print(file)
        file = os.path.join(FLAGS.path, file)
        data = np.load(file)
        outputs_to_figure_or_file(data['prediction'], data['labels'], data['cloud_mask'], FLAGS.use_continuous_colors,
                                  FLAGS.cloud_mask_as_binary, FLAGS.to_file, file, FLAGS.label_dilation)


if __name__ == '__main__':
    app.run(main)
