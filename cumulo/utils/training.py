from tqdm import tqdm
import numpy as np

MAX_WIDTH, MAX_HEIGHT = 1354, 2030


class GlobalNormalizer(object):

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        return (image - self.mean) / self.std


class LocalNormalizer(object):

    def __call__(self, image):
        means = np.mean(image, axis=(1, 2)).reshape((image.shape[0], 1, 1))
        stds = np.std(image, axis=(1, 2)).reshape((image.shape[0], 1, 1))
        image = (image - means) / stds
        return image


def get_sampling_mask(mask_shape, tile_size):
    mask = np.ones(mask_shape, dtype=np.uint8)
    mask[:, :tile_size] = 0
    mask[:, -tile_size:] = 0
    mask[:tile_size, :] = 0
    mask[-tile_size:, :] = 0
    return mask


def sample_n_tiles_with_labels(radiances, cloud_mask, labels, n, tile_size, valid_convolution_offset, filter_cloudy_labels):
    """
    Randomly draws n tiles with labels from the given swath. If n > number of labeled pixels, multiple tiles are extracted around
    each pixel. However, all tiles differ due to random shifting.

    Args:
        radiances: radiances from the swath.
        cloud_mask: cloud mask from the swath.
        labels: labels from the swath.
        n: number of tiles.
        tile_size: tile extensions (e.g. 256 x 256).
        valid_convolution_offset: the size offset between network inputs and outputs (due to valid convolutions).
        filter_cloudy_labels: extract tiles only from labels which are inside clouds.
    """
    allowed_mask = get_sampling_mask((MAX_WIDTH, MAX_HEIGHT), tile_size)
    labelled_mask = labels != -1
    if filter_cloudy_labels:
        potential_pixels = allowed_mask & labelled_mask & cloud_mask
    else:
        potential_pixels = allowed_mask & labelled_mask
    potential_pixels = np.array(list(zip(*np.where(potential_pixels == 1))))
    potential_pixels_cache = potential_pixels.copy()

    if len(potential_pixels) == 0:
        return None, None, None

    if len(potential_pixels) > n:
        idcs = np.random.choice(np.arange(len(potential_pixels)), n, replace=False)
        potential_pixels = potential_pixels[idcs]
    else:
        # --- if there are not enough labeled pixels, multiple tiles are extracted around single pixels.
        # This normally doesn't happen as most nc files contain more than 128 labeled pixels and normal
        # batch sizes for U-Nets are not larger than 32. ---
        while len(potential_pixels) < n:
            potential_pixels = np.vstack((potential_pixels, potential_pixels_cache))
        idcs = np.random.choice(np.arange(len(potential_pixels)), n, replace=False)
        potential_pixels = potential_pixels[idcs]

    if tile_size > 3:
        # --- shift tiles randomly to avoid overfitting, but ensure that labels are within network output (in case of valid convolutions) ---
        random_offsets = np.random.randint(-(tile_size // 2) + valid_convolution_offset + 1, (tile_size // 2) - valid_convolution_offset - 1, potential_pixels.shape)
        potential_pixels += random_offsets

    swath_tuple = (radiances, cloud_mask, labels)
    tiles = [[] for _ in swath_tuple]
    for pixel in potential_pixels:
        ll = pixel - tile_size // 2
        ur = pixel + tile_size // 2
        if tile_size % 2 != 0:
            ur += 1
        for ix, variable in enumerate(swath_tuple):
            if ix == 0:
                tiles[ix].append(variable[:, ll[0]:ur[0], ll[1]:ur[1]])  # radiances have shape (13, height, width)
            else:
                tiles[ix].append(variable[ll[0]:ur[0], ll[1]:ur[1]])  # labels and cloud mask have shape (height, width)
    return tuple(map(np.stack, tiles))


def get_dataset_statistics(dataset, class_number, tile_size, sample_number=None):
    """
    Calculates the statistics (class weights and channel-wise mean and variance) of the dataset.

    Args:
        dataset: DataSet which returns either batched or single tiles (e.g. the CumuloDataset)
        class_number: number of classes in the dataset.
        tile_size: tile extends (e.g. 256 for 256 x 256).
        sample_number: number of tiles to use for the statistics calculation.
    """
    weights = np.zeros(class_number)
    mean = np.zeros(13)
    variance = np.zeros(13)
    if sample_number is None:
        sample_number = len(dataset)
    tile_number = sample_number
    radiances, labels, _ = dataset[0]
    if len(radiances.shape) == 4:
        # calculate statistics when dataset returns batched tiles
        tile_number *= radiances.shape[0]

    batch_size = 1
    for sample in tqdm(range(sample_number)):
        radiances, labels, _ = dataset[sample]
        if len(radiances.shape) == 4:
            batch_size = radiances.shape[0]
        for i in range(batch_size):
            radiances_ = radiances[i].numpy()
            labels_ = labels[i].numpy()
            weights += np.histogram(labels_, bins=range(class_number + 1), normed=False)[0]
            mean += np.mean(radiances_, axis=(1, 2))

    mean /= tile_number
    mean = mean.reshape((13, 1, 1))

    for sample in tqdm(range(sample_number)):
        radiances, labels, _ = dataset[sample]
        for i in range(batch_size):
            radiances_ = radiances[i].numpy()
            variance += np.sum((radiances_ - mean)**2, (1, 2))

    variance /= tile_number * tile_size ** 2
    std = np.sqrt(variance)
    std = std.reshape((13, 1, 1))
    weights = weights / np.sum(weights)
    weights_div = 1 / (np.log(1.02 + weights))
    return weights, weights_div, mean, std
