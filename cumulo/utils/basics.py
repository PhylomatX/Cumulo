import netCDF4 as nc4
import numpy as np

radiances_nc = ['ev_250_aggr1km_refsb_1', 'ev_250_aggr1km_refsb_2', 'ev_1km_emissive_29', 'ev_1km_emissive_33',
                'ev_1km_emissive_34', 'ev_1km_emissive_35', 'ev_1km_emissive_36', 'ev_1km_refsb_26',
                'ev_1km_emissive_27', 'ev_1km_emissive_20', 'ev_1km_emissive_21', 'ev_1km_emissive_22', 'ev_1km_emissive_23']
cloud_mask_nc = 'cloud_mask'
labels_nc = 'cloud_layer_type'


def read_nc(nc_file):
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
