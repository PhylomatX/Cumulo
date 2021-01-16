import glob
import os
from absl import app
from absl import flags
import numpy as np
from cumulo.data.nc_loader import read_nc
from cumulo.data.nc_tile_extractor import sample_random_tiles_from_track

flags.DEFINE_string('nc_path', None, help='The dataset directory.')
flags.DEFINE_string('npz_path', None, help='Directory where tiles should be saved.')
flags.DEFINE_integer('size', 3, help='Tile size.')
flags.DEFINE_integer('redundancy', 1, help='How many tiles should get extracted at each position?')
flags.DEFINE_boolean('remove', False, help='Remove nc files after tiles have been generated.')
FLAGS = flags.FLAGS


def get_most_frequent_label(labels):
    """ labels should be of size (nb_instances, ...).

        Returns the most frequent label for each whole instance.
    """

    labels = labels.squeeze()
    mask = np.any(labels != -1, axis=2)
    lpixels = labels[mask]

    rpixels = np.zeros(len(lpixels))

    for ix, lpixel in enumerate(lpixels):
        occ = np.zeros(8)
        uniques, counts = np.unique(lpixel, return_counts=True)
        for v, c in zip(uniques, counts):
            if v != -1:
                occ[v] = c
        rpixels[ix] = np.argmax(occ).astype(float)

    labels_flat = np.ones_like(labels[..., 0]) * -1
    labels_flat[mask] = rpixels

    return labels_flat


def include_cloud_mask(labels, cloud_mask):
    labels[labels >= 0] += 1
    return labels * cloud_mask


def main(_):
    nc_dir = FLAGS.nc_path
    save_dir = FLAGS.npz_path

    file_paths = glob.glob(os.path.join(nc_dir, "*.nc"))

    if len(file_paths) == 0:
        raise FileNotFoundError("no nc files in", nc_dir)

    for ix, filename in enumerate(file_paths):
        radiances, properties, cloud_mask, labels = read_nc(filename)

        name = os.path.basename(filename).replace(".nc", "") + f'_{FLAGS.size}'

        save_name = os.path.join(save_dir, name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        tiles, positions = sample_random_tiles_from_track(radiances, properties, cloud_mask, labels,
                                                          tile_size=FLAGS.size, redundancy=FLAGS.redundancy)

        if tiles is None:
            continue

        for tile in range(tiles[0].shape[0]):
            labels = tiles[3].data[tile].squeeze()
            cloud_mask = tiles[2].data[tile].squeeze()
            mf_labels = include_cloud_mask(get_most_frequent_label(labels), cloud_mask)
            low_labels = include_cloud_mask(labels[..., 0], cloud_mask)
            low_labels_raw = labels[..., 0]

            np.savez(save_name + '_' + str(tile), radiances=tiles[0].data[tile], labels=np.dstack((mf_labels, low_labels, low_labels_raw)))
            # np.savez(save_name + '_' + str(tile), radiances=tiles[0].data[tile], properties=tiles[1].data[tile],
            #          cloud_mask=tiles[2].data[tile], labels=tiles[3].data[tile], location=positions[tile])

        if FLAGS.remove:
            os.remove(filename)


if __name__ == '__main__':
    app.run(main)
