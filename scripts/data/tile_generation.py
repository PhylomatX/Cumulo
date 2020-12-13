import glob
import os
from absl import app
from absl import flags
import numpy as np
from cumulo.data.nc_loader import read_nc
from cumulo.data.nc_tile_extractor import get_label_mask, sample_labelled_and_unlabelled_tiles, extract_cloudy_labelled_tiles

flags.DEFINE_string('nc_path', None, help='The dataset directory.')
flags.DEFINE_string('npz_path', None, help='Directory where tiles should be saved.')
flags.DEFINE_integer('size', 3, help='Tile size.')
FLAGS = flags.FLAGS


def main(_):
    nc_dir = FLAGS.nc_path
    save_dir = FLAGS.npz_path

    # for dr in [os.path.join(save_dir, "label"), os.path.join(save_dir, "unlabel")]:
    #     if not os.path.exists(dr):
    #         os.makedirs(dr)

    file_paths = glob.glob(os.path.join(nc_dir, "*.nc"))

    if len(file_paths) == 0:
        raise FileNotFoundError("no nc files in", nc_dir)

    for ix, filename in enumerate(file_paths):
        radiances, properties, cloud_mask, labels = read_nc(filename)
        label_mask = get_label_mask(labels)

        # labelled_tiles, unlabelled_tiles, labelled_positions, unlabelled_positions = sample_labelled_and_unlabelled_tiles(
        #     (radiances, properties, cloud_mask, labels), cloud_mask[0], label_mask[0], tile_size=FLAGS.size)

        labelled_tiles, labelled_positions = extract_cloudy_labelled_tiles((radiances, properties, cloud_mask, labels),
                                                                           cloud_mask[0], label_mask[0], tile_size=FLAGS.size)

        if labelled_tiles is None:
            continue

        name = os.path.basename(filename).replace(".nc", "") + f'_{FLAGS.size}'

        print(labelled_tiles[0].shape)

        save_name = os.path.join(save_dir, name)
        np.savez_compressed(save_name, radiances=labelled_tiles[0].data, properties=labelled_tiles[1].data,
                            cloud_mask=labelled_tiles[2].data, labels=labelled_tiles[3].data,
                            location=labelled_positions)

        # save_name = os.path.join(save_dir, "unlabel", name)
        # np.savez_compressed(save_name, radiances=unlabelled_tiles[0].data, properties=unlabelled_tiles[1].data,
        #                     cloud_mask=unlabelled_tiles[2].data, labels=unlabelled_tiles[3].data,
        #                     location=unlabelled_positions)


if __name__ == '__main__':
    app.run(main)
