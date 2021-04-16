import os
from absl import app
from absl import flags
import numpy as np
import pickle as pkl
from tqdm import tqdm
from cumulo.utils.basics import read_nc

flags.DEFINE_string('nc_path', None, help='Directory where nc files are located.')
flags.DEFINE_string('removed_path', None, help='Directory where removed nc files should get saved.')
FLAGS = flags.FLAGS

MAX_WIDTH, MAX_HEIGHT = 1354, 2030


def check_size(radiances):
    """ Ensures that the radiances of all images have the same shape. """
    if radiances.shape != (13, 1354, 2030):
        print('Invalid size!')
        return True
    return False


def check_for_stripe_pattern(radiances) -> bool:
    """ Checks if multiple columns / rows are the same as this is a common issue in the nc files. """
    erroneous = False
    for i in range(radiances.shape[2] - 1):
        if np.all(radiances[0][:, i] == radiances[0][:, i + 1]):
            print('Artefacts!')
            erroneous = True
            break
    return erroneous


def main(_):
    files = os.listdir(FLAGS.nc_path)
    no_labels = []
    removed = 0
    for file in tqdm(files):
        filename = os.path.join(FLAGS.nc_path, file)
        try:
            radiances, cloud_mask, labels = read_nc(filename)
        except:
            print('Invalid file')
            continue
        if check_size(radiances) or check_for_stripe_pattern(radiances):
            os.rename(filename, filename.replace(FLAGS.nc_path, FLAGS.removed_path))
            removed += 1
        elif np.all(labels == -1):
            print('No labels!')
            no_labels.append(file)
    print(f'{removed} nc files have been removed because of artefacts.')
    with open(os.path.join(FLAGS.nc_path, 'no_labels.pkl'), 'wb') as f:
        pkl.dump(no_labels, f)


if __name__ == '__main__':
    app.run(main)
