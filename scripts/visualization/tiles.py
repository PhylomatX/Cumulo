import re
import os
import numpy as np
from absl import app
from .flags import FLAGS
from cumulo.utils.visualization import outputs_to_figure_or_file


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
        file = os.path.join(FLAGS.path, file)
        data = np.load(file)
        outputs_to_figure_or_file(data['outputs'], data['labels'], data['cloud_mask'], FLAGS.use_continuous_colors, FLAGS.cloud_mask_as_binary, FLAGS.to_file, )


if __name__ == '__main__':
    app.run(main)
