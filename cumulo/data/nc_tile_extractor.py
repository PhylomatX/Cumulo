import numpy as np
import random

MAX_WIDTH, MAX_HEIGHT = 1354, 2030


# -------------------------------------------------------------------------------------------------- UTILS

def get_tile_offsets(tile_size):
    offset = tile_size // 2
    offset_2 = offset

    if not tile_size % 2:
        offset_2 -= 1

    return offset, offset_2


def get_sampling_mask(mask_shape=(MAX_HEIGHT, MAX_WIDTH), tile_size=3):
    """
    Returns a mask of allowed centers for the tiles to be sampled. Excludes all points within
    a tile_size offset from the border regions.
    """
    mask = np.ones(mask_shape, dtype=np.uint8)

    # must not sample tile centers in the borders, so that tiles keep to required shape
    mask[:, :tile_size] = 0
    mask[:, -tile_size:] = 0
    mask[:tile_size, :] = 0
    mask[-tile_size:, :] = 0

    return mask


def get_label_mask(labels):
    """ given the class occurences channels over 10 layers, returns a 2d array marking as 1 the labelled pixels and as 0 the unlabelled ones."""

    label_mask = np.sum(~labels.mask, 3) > 0

    return label_mask


def get_unlabel_mask(label_mask, tile_size=3):
    """returns inverse of label mask, with all pixels around a labelled one eroded."""

    offset, offset_2 = get_tile_offsets(tile_size)

    unlabel_mask = (~label_mask).copy()

    labelled_idx = np.where(label_mask)

    for center_w, center_h in zip(*labelled_idx):
        w1 = center_w - offset
        w2 = center_w + offset_2 + 1
        h1 = center_h - offset
        h2 = center_h + offset_2 + 1

        unlabel_mask[w1:w2, h1:h2] = False

    return unlabel_mask


# -------------------------------------------------------------------------------------------------- SAMPLERS

def sample_cloudy_unlabelled_tiles(swath_tuple, cloud_mask, label_mask, number_of_tiles, tile_size=3):
    """
    :param swath_tuple: 
    :param cloud_mask: 2d array of zise (w, h) marking the cloudy pixels 
    :param label_mask: 2d array of zise (w, h) marking the labelled pixels 
    :param number_of_tiles: the number of tiles to sample. It is reset to maximal number of tiles that can be sampled, if bigger
    :param tile_size: size of the tile selected from within the image
    :return: a 4-d array (nb_tiles, nb_channels, w, h) of sampled tiles; and a list of tuples ((w1, w2), (h1, h2)) with the relative positions of the sampled tiles withing the swath
    The script will use a cloud_mask channel to mask away all non-cloudy data and a label_mask channel to mask away all labelled data. The script will then randomly select a number of tiles (:param number of tiles) from the cloudy areas that are unlabelled.
    """

    # mask out borders not to sample outside the swath
    allowed_pixels = get_sampling_mask((MAX_WIDTH, MAX_HEIGHT), tile_size)

    # mask out labelled pixels and pixels around them
    unlabel_mask = get_unlabel_mask(label_mask)

    # combine the three masks, tile centers will be sampled from the cloudy and unlabelled pixels that are not in the borders of the swath
    unlabelled_pixels = np.logical_and.reduce([allowed_pixels, cloud_mask, unlabel_mask])
    unlabelled_pixels_idx = np.where(unlabelled_pixels == 1)
    unlabelled_pixels_idx = list(zip(*unlabelled_pixels_idx))

    number_of_tiles = min(number_of_tiles, len(unlabelled_pixels_idx))

    # sample without replacement
    tile_centers_idx = np.random.choice(np.arange(len(unlabelled_pixels_idx)), number_of_tiles, False)
    unlabelled_pixels_idx = np.array(unlabelled_pixels_idx)
    tile_centers = unlabelled_pixels_idx[tile_centers_idx]

    # compute distances from tile center of tile upper left and lower right corners
    offset, offset_2 = get_tile_offsets(tile_size)

    positions, tiles = [], [[] for _ in swath_tuple]
    for center in tile_centers:
        center_w, center_h = center

        w1 = center_w - offset
        w2 = center_w + offset_2 + 1
        h1 = center_h - offset
        h2 = center_h + offset_2 + 1

        tile_position = ((w1, w2), (h1, h2))

        positions.append(tile_position)
        for i, a in enumerate(swath_tuple):
            tiles[i].append(a[:, w1:w2, h1:h2])

    positions = np.stack(positions)
    for i, t in enumerate(tiles):
        tiles[i] = np.stack(t)

    return tiles, positions


def extract_cloudy_labelled_tiles(swath_tuple, cloud_mask, label_mask, tile_size=3):
    """
    :param swath_tuple: input numpy array from MODIS of size (nb_channels, w, h)
    :param cloud_mask: 2d array of zise (w, h) marking the cloudy pixels 
    :param label_mask: 2d array of zise (w, h) marking the labelled pixels 
    :param tile_size: the size of the channels
    :return: a 4-d array (nb_tiles, nb_channels, w, h) of sampled tiles; and a list of tuples ((w1, w2), (h1, h2)) with the relative positions of the extracted tiles withing the swath
    The script will use a cloud_mask channel to mask away all non-cloudy data and a label_mask channel to mask away all unlabelled data. The script will then select all tiles from the cloudy areas that are labelled.
    """

    # mask not to sample outside the swath
    allowed_pixels = get_sampling_mask((MAX_WIDTH, MAX_HEIGHT), tile_size)

    # combine the three masks, tile centers will be sampled from the cloudy and labelled pixels that are not in the borders of the swath
    labelled_pixels = allowed_pixels & cloud_mask & label_mask
    labelled_pixels_idx = np.where(labelled_pixels == 1)
    labelled_pixels_idx = list(zip(*labelled_pixels_idx))

    if len(labelled_pixels_idx) == 0:
        return None, None

    offset, offset_2 = get_tile_offsets(tile_size)

    positions, tiles = [], [[] for _ in swath_tuple]
    for center in labelled_pixels_idx:
        center_w, center_h = center

        w1 = center_w - offset
        w2 = center_w + offset_2 + 1
        h1 = center_h - offset
        h2 = center_h + offset_2 + 1

        tile_position = ((w1, w2), (h1, h2))

        positions.append(tile_position)
        for i, a in enumerate(swath_tuple):
            tiles[i].append(a[:, w1:w2, h1:h2])

    positions = np.stack(positions)
    for i, t in enumerate(tiles):
        tiles[i] = np.stack(t)

    return tiles, positions


def sample_random_tiles_from_track(radiances, cloud_mask, labels, tile_size=128, verbose=False, properties=None,
                                   batch_size=None, center_distance=None):
    # sampling tiles from border region is not allowed
    allowed_pixels = get_sampling_mask((MAX_WIDTH, MAX_HEIGHT), tile_size)
    # get pixels along satellite track
    label_mask = get_label_mask(labels)
    potential_pixels = allowed_pixels & label_mask
    potential_pixels_idx = np.array(list(zip(*np.where(potential_pixels[0] == 1))))

    if len(potential_pixels_idx) == 0:
        return None, None

    if center_distance is None:
        center_distance = tile_size

    if center_distance > 0:
        # filter equidistant labeled pixels
        last_center = None
        not_used = 0
        mask = np.ones(len(potential_pixels_idx)).astype(np.bool)
        for ix, center in enumerate(potential_pixels_idx):
            if last_center is None:
                last_center = center
            else:
                if np.sqrt(np.sum((center - last_center)**2)) < center_distance:
                    not_used += 1
                    mask[ix] = False
                else:
                    last_center = center
        if verbose:
            print(f'Used: {len(potential_pixels_idx)-not_used}. Total: {len(potential_pixels_idx)}')
        potential_pixels_idx = potential_pixels_idx[mask]

    potential_pixels_idx_orig = potential_pixels_idx

    if batch_size is not None:
        if len(potential_pixels_idx) > batch_size:
            idcs = np.random.choice(np.arange(len(potential_pixels_idx)), batch_size, replace=False)
            potential_pixels_idx = potential_pixels_idx[idcs]
        else:
            while len(potential_pixels_idx) < batch_size:
                potential_pixels_idx = np.vstack((potential_pixels_idx, potential_pixels_idx_orig))
            idcs = np.random.choice(np.arange(len(potential_pixels_idx)), batch_size, replace=False)
            potential_pixels_idx = potential_pixels_idx[idcs]

    random_offsets = np.random.randint(-(tile_size // 2) + 1, (tile_size // 2) - 1, potential_pixels_idx.shape)
    potential_pixels_idx += random_offsets
    offset, offset_2 = get_tile_offsets(tile_size)
    if properties is not None:
        swath_tuple = (radiances, properties, cloud_mask, labels)
    else:
        swath_tuple = (radiances, cloud_mask, labels)
    positions, tiles = [], [[] for _ in swath_tuple]
    for center in potential_pixels_idx:
        ll = center - offset
        ur = center + offset_2 + 1
        tile_position = ((ll[0], ur[0]), (ll[1], ur[1]))

        positions.append(tile_position)
        for i, a in enumerate(swath_tuple):
            tiles[i].append(a[:, ll[0]:ur[0], ll[1]:ur[1]])

    positions = np.stack(positions)
    for i, t in enumerate(tiles):
        tiles[i] = np.stack(t)

    return tiles, positions


def sample_labelled_and_unlabelled_tiles(swath_tuple, cloud_mask, label_mask, tile_size=3):
    """
    :param swath_tuple: tuple of numpy arrays of size (C, H, W, ...) to be tiled coherently
    :param tile_size: size of tile (default 3)
    :param cloud_mask: mask where cloudy
    :param label_mask: mask where labels are available 
    :return: nested list of labelled tiles, unlabelled tiles, labelled tile positions, unlabelled tile positions
    Samples the same amount of labelled and unlabelled tiles from the cloudy data.
    """

    labelled_tiles, labelled_positions = extract_cloudy_labelled_tiles(swath_tuple, cloud_mask, label_mask, tile_size)
    if labelled_tiles is None:
        return None, None, None, None

    number_of_labels = len(labelled_tiles[0])

    unlabelled_tiles, unlabelled_positions = sample_cloudy_unlabelled_tiles(swath_tuple, cloud_mask, label_mask,
                                                                            number_of_labels, tile_size)

    return labelled_tiles, unlabelled_tiles, labelled_positions, unlabelled_positions
