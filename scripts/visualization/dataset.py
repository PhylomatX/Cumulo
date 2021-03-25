import os
import imageio
import numpy as np
from absl import app
from absl import flags
from cumulo.utils.basics import read_nc
from cumulo.utils.visualization import labels_and_cloud_mask_to_rgb

flags.DEFINE_string('path', None, help='Directory where nc files are located.')
flags.DEFINE_integer('swath_number', 1, help='How many swaths should get visualized?')
FLAGS = flags.FLAGS


def main(_):
    files = list(filter(lambda f: 'nc' in f, os.listdir(FLAGS.path)))

    for file in range(FLAGS.swath_number):
        file = os.path.join(FLAGS.path, files[file])
        radiances, cloud_mask, labels = read_nc(file)
        ground_truth = labels_and_cloud_mask_to_rgb(labels, cloud_mask)
        imageio.imwrite(file.replace('.nc', f'_gt.png'), (ground_truth * 255).astype(np.uint8))
        for radiance in range(radiances.shape[0]):
            imageio.imwrite(file.replace('.nc', f'_rad{radiance}.png'), radiances[radiance])


if __name__ == '__main__':
    app.run(main)
