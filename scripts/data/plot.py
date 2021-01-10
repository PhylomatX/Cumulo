import os
import numpy as np
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from cumulo.data.nc_loader import read_nc_full

if __name__ == '__main__':
    path = os.path.expanduser('~/Projekte/BAP/Data/Cumulo/nc/A2008.001.1250.nc')
    data = read_nc_full(path)

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()

    x = data['longitude']
    y = data['latitude']

    ax.scatter(x[0:40, 0:40], y[0:40, 0:40], c=data['cloud_mask'][0:40, 0:40])
    print(np.unique(data['cloud_mask'][0:40, 0:40], return_counts=True))
    plt.show()
