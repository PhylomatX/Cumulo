import os
from absl import app
from absl import flags
import numpy as np
from tqdm import tqdm
from cumulo.data.nc_loader import read_nc

flags.DEFINE_string('nc_path', None, help='Directory where nc files are located.')
FLAGS = flags.FLAGS


def check_for_stripe_pattern(radiances) -> bool:
    """ Checks if multiple columns / rows are the same as this is a common issue in the nc files. """
    erroneous = False
    for i in range(radiances.shape[2] - 1):
        if np.all(radiances[0][:, i] == radiances[0][:, i + 1]):
            erroneous = True
            break
    return erroneous


def main(_):
    files = os.listdir(FLAGS.nc_path)
    removed = 0
    for file in tqdm(files):
        filename = os.path.join(FLAGS.nc_path, file)
        radiances, properties, cloud_mask, labels = read_nc(filename)
        if check_for_stripe_pattern(radiances):
            os.remove(filename)
            removed += 1
            print(f'Removed {file}')
    print(f'{removed} nc files have been removed because of artefacts.')


if __name__ == '__main__':
    app.run(main)
