import numpy as np


def divide_into_tiles(tile_size, offset, radiances):
    img_width = radiances.shape[1]
    img_height = radiances.shape[2]

    output_size = tile_size - 2 * offset
    nb_outputs_row = (img_width - 2 * offset) // output_size
    nb_outputs_col = (img_height - 2 * offset) // output_size

    radiances = []
    locations = []

    # --- gather tiles from within swath ---
    for i in range(nb_outputs_row):
        for j in range(nb_outputs_col):
            radiances.append(radiances[:, i * output_size: 2 * offset + (i + 1) * output_size, j * output_size: 2 * offset + (j + 1) * output_size])
            locations.append(((offset + i * output_size, offset + (i + 1) * output_size),
                              (offset + j * output_size, offset + (j + 1) * output_size)))

    # --- gather tiles from bottom row ---
    for i in range(nb_outputs_row):
        radiances.append(radiances[:, i * output_size: 2 * offset + (i + 1) * output_size, img_height - tile_size:img_height])
        locations.append(((offset + i * output_size, offset + (i + 1) * output_size),
                          (offset + img_height - tile_size, img_height - offset)))

    # --- gather tiles from most right column ---
    for j in range(nb_outputs_col):
        radiances.append(radiances[:, img_width - tile_size:img_width, j * output_size: 2 * offset + (j + 1) * output_size])
        locations.append(((offset + img_width - tile_size, img_width - offset),
                          (offset + j * output_size, offset + (j + 1) * output_size)))

    # --- gather tile from lower right corner ---
    radiances.append(radiances[:, img_width - tile_size:img_width, img_height - tile_size:img_height])
    locations.append(((offset + img_width - tile_size, img_width - offset),
                      (offset + img_height - tile_size, img_height - offset)))

    radiances = np.stack(radiances)
    locations = np.stack(locations)

    return radiances, locations


def write_confusion_matrix(cm: np.array, names: list) -> str:
    txt = f"{'':<15}"
    for name in names:
        txt += f"{name:<15}"
    txt += '\n'
    for ix, name in enumerate(names):
        txt += f"{name:<15}"
        for num in cm[ix]:
            txt += f"{num:<15}"
        txt += '\n'
    return txt


def get_target_names(gtl: np.ndarray, hcl: np.ndarray, targets: list) -> list:
    """ Extracts the names of the labels which appear in gtl and hcl. """
    targets = np.array(targets)
    total = np.unique(np.concatenate((gtl, hcl), axis=0)).astype(int)
    return list(targets[total])