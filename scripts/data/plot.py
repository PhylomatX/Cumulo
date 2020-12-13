import os
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

    ax.scatter(x, y, c=data['ev_250_aggr1km_refsb_1'])

    plt.show()
