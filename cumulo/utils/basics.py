import netCDF4 as nc4
import numpy as np
from scipy import special as ss

radiances_nc = ['ev_250_aggr1km_refsb_1', 'ev_250_aggr1km_refsb_2', 'ev_1km_emissive_29', 'ev_1km_emissive_33',
                'ev_1km_emissive_34', 'ev_1km_emissive_35', 'ev_1km_emissive_36', 'ev_1km_refsb_26',
                'ev_1km_emissive_27', 'ev_1km_emissive_20', 'ev_1km_emissive_21', 'ev_1km_emissive_22', 'ev_1km_emissive_23']
cloud_mask_nc = 'cloud_mask'
labels_nc = 'cloud_layer_type'


def read_nc(nc_file, filter_most_freqent=False):
    file = nc4.Dataset(nc_file, 'r', format='NETCDF4')
    radiances = np.vstack([file.variables[name][:] for name in radiances_nc])
    cloud_mask = file.variables[cloud_mask_nc][:]
    labels = file.variables[labels_nc][:]
    labels = labels.data[0]
    if filter_most_freqent:
        labels = get_most_frequent_label(labels)  # take most frequent clouds of each pixel as GT
    else:
        labels = labels[..., 0]  # take lowest clouds as GT
    file.close()
    return radiances.data, cloud_mask.data[0], labels


def get_most_frequent_label(labels):
    labels = labels
    mask = np.any(labels != -1, axis=2)
    labeled_pixels = labels[mask]
    result_pixels = np.zeros(len(labeled_pixels))

    for ix, labeled_pixel in enumerate(labeled_pixels):
        occ = np.zeros(8)
        uniques, counts = np.unique(labeled_pixel, return_counts=True)
        for v, c in zip(uniques, counts):
            if v != -1:
                occ[v] = c
        result_pixels[ix] = np.argmax(occ)

    labels_flat = np.ones_like(labels[..., 0]) * -1
    labels_flat[mask] = result_pixels

    return labels_flat


def include_cloud_mask(labels, cloud_mask):
    labels = labels.copy()
    labels[labels >= 0] += 1
    return labels * cloud_mask


def probabilities_from_outputs(outputs):
    outputs[0, ...] = ss.expit(outputs[0, ...])  # first channel was trained for cloud mask
    outputs[1:9, ...] = ss.softmax(outputs[1:9, ...], axis=0)  # next 8 channels were trained for cloud classes
    return outputs
