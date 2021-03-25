from tqdm import tqdm
import netCDF4 as nc4
import numpy as np

radiances_nc = ['ev_250_aggr1km_refsb_1', 'ev_250_aggr1km_refsb_2', 'ev_1km_emissive_29', 'ev_1km_emissive_33',
                'ev_1km_emissive_34', 'ev_1km_emissive_35', 'ev_1km_emissive_36', 'ev_1km_refsb_26',
                'ev_1km_emissive_27', 'ev_1km_emissive_20', 'ev_1km_emissive_21', 'ev_1km_emissive_22', 'ev_1km_emissive_23']
cloud_mask_nc = 'cloud_mask'
labels_nc = 'cloud_layer_type'

MAX_WIDTH, MAX_HEIGHT = 1354, 2030


def read_nc(nc_file):
    """return masked arrays, with masks indicating the invalid values"""
    file = nc4.Dataset(nc_file, 'r', format='NETCDF4')
    radiances = np.vstack([file.variables[name][:] for name in radiances_nc])
    cloud_mask = file.variables[cloud_mask_nc][:]
    labels = file.variables[labels_nc][:]
    labels = labels.data[0]
    labels = labels[..., 0]  # take lowest clouds as GT
    file.close()
    return radiances.data, cloud_mask.data[0], labels


def include_cloud_mask(labels, cloud_mask):
    labels = labels.copy()
    labels[labels >= 0] += 1
    return labels * cloud_mask


def get_dataset_statistics(dataset, nb_classes, tile_size, nb_samples=None):
    weights = np.zeros(nb_classes)
    m = np.zeros(13)
    s = np.zeros(13)
    if nb_samples is None:
        nb_samples = len(dataset)
    nb_tiles = nb_samples
    rads, labels, _ = dataset[0]
    if len(rads.shape) == 4:
        nb_tiles *= rads.shape[0]

    batch_size = 1
    for sample in tqdm(range(nb_samples)):
        rads, labels, _ = dataset[sample]
        if len(rads.shape) == 4:
            batch_size = rads.shape[0]
        for i in range(batch_size):
            crads = rads[i].numpy()
            clabels = labels[i].numpy()
            weights += np.histogram(clabels, bins=range(nb_classes + 1), normed=False)[0]
            m += np.mean(crads, axis=(1, 2))
        print(f"Sample: {sample}")
        print(f"weights: {weights / np.sum(weights)}")
        print(f"mean: {m / (sample * rads.shape[0])}")

    m /= nb_tiles
    m = m.reshape((13, 1, 1))

    for sample in tqdm(range(nb_samples)):
        rads, labels = dataset[sample]
        for i in range(batch_size):
            crads = rads[i].numpy()
            s += np.sum((crads - m)**2, (1, 2))
        print(f"weights: {weights / np.sum(weights)}")
        print(f"mean: {m}")

    s /= nb_tiles * tile_size ** 2
    std = np.sqrt(s)
    std = std.reshape((13, 1, 1))
    weights = weights / np.sum(weights)
    weights_div = 1 / (np.log(1.02 + weights))
    return weights, weights_div, m, std


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


def get_sampling_mask(mask_shape, tile_size):
    """
    Returns a mask of allowed centers for the tiles to be sampled. Excludes all points within
    a tile_size offset from the border regions.
    """
    mask = np.ones(mask_shape, dtype=np.uint8)
    mask[:, :tile_size] = 0
    mask[:, -tile_size:] = 0
    mask[:tile_size, :] = 0
    mask[-tile_size:, :] = 0
    return mask


def sample_n_tiles_with_labels(radiances, cloud_mask, labels, n, tile_size=128, valid_convolution_offset=0, filter_cloudy_labels=True):
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
        while len(potential_pixels) < n:
            potential_pixels = np.vstack((potential_pixels, potential_pixels_cache))
        idcs = np.random.choice(np.arange(len(potential_pixels)), n, replace=False)
        potential_pixels = potential_pixels[idcs]

    # shift tiles randomly to avoid overfitting, but ensure that labels are within network output (in case of valid convolutions)
    random_offsets = np.random.randint(-(tile_size // 2) + 1 + valid_convolution_offset, (tile_size // 2) - 1 - valid_convolution_offset, potential_pixels.shape)
    potential_pixels += random_offsets

    swath_tuple = (radiances, cloud_mask, labels)
    tiles = [[] for _ in swath_tuple]
    for pixel in potential_pixels:
        ll = pixel - tile_size // 2
        ur = pixel + tile_size // 2
        for ix, variable in enumerate(swath_tuple):
            if ix == 0:
                tiles[ix].append(variable[:, ll[0]:ur[0], ll[1]:ur[1]])  # radiances have shape (13, height, width)
            else:
                tiles[ix].append(variable[ll[0]:ur[0], ll[1]:ur[1]])  # labels and cloud mask have shape (height, width)
    return tuple(map(np.stack, tiles))
