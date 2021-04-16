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
        outputs_to_figure_or_file(data['outputs'], data['labels'], data['cloud_mask'], cloud_mask_as_binary=FLAGS.cloud_mask_as_binary,
                                  to_file=FLAGS.to_file, npz_file=file, label_dilation=FLAGS.label_dilation, no_cloud_mask_prediction=FLAGS.no_cloud_mask)


if __name__ == '__main__':
    app.run(main)
