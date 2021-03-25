import os
from absl import app
from absl import flags
import numpy as np
from tqdm import tqdm
from cumulo.utils.utils import read_nc, get_sampling_mask

flags.DEFINE_string('nc_path', None, help='Directory where nc files are located.')
flags.DEFINE_integer('size', 128, help='Tile size.')
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


def check_for_empty_labels(tile_size, labels):
    allowed_pixels = get_sampling_mask((MAX_WIDTH, MAX_HEIGHT), tile_size)
    label_mask = get_label_mask(labels)
    potential_pixels = allowed_pixels & label_mask
    potential_pixels_idx = np.array(list(zip(*np.where(potential_pixels == 1))))

    if len(potential_pixels_idx) == 0:
        print('No labels!')
        return True
    else:
        return False


def main(_):
    files = os.listdir(FLAGS.nc_path)
    artefacts = []
    no_labels = []
    clean = os.path.join(FLAGS.nc_path, 'clean')
    if not os.path.exists(clean):
        os.makedirs(clean)
    for file in tqdm(files):
        filename = os.path.join(FLAGS.nc_path, file)
        try:
            radiances, properties, cloud_mask, labels = read_nc(filename)
        except:
            print('Invalid file')
            continue
        if check_size(radiances) or check_for_stripe_pattern(radiances):
            os.rename(filename, filename.replace(FLAGS.nc_path, artefacts + '/'))
            removed += 1
        elif check_for_empty_labels(FLAGS.size, labels):
            os.rename(filename, filename.replace(FLAGS.nc_path, no_labels + '/'))
            removed += 1
        else:
            os.rename(filename, filename.replace(FLAGS.nc_path, clean + '/'))
    print(f'{removed} nc files have been removed because of artefacts.')


if __name__ == '__main__':
    app.run(main)
