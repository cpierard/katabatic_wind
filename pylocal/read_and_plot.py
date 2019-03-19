
import h5py
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy.ma as ma #for the mask
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as colors
import scipy.ndimage as ndimage

def extraer_datos(nombre_h5):

    with h5py.File(nombre_h5, 'r') as hdf:
        base_items = list(hdf.items())
        print(base_items, '\n')
        tasks = hdf.get('tasks')
        scales = hdf.get('scales')
        #scales_items = list(scales.items())
        #print(scales_items)
        tasks_items = list(tasks.items())
        print(tasks_items)

        T = np.array(tasks.get('T'))
        ρ = np.array(tasks.get('ρ'))
        u = np.array(tasks.get('u'))
        v = np.array(tasks.get('v'))
        ω = np.array(tasks.get('vort'))
        p = np.array(tasks.get('p'))
        t = np.array(scales.get('sim_time'))
        T_x = np.array(tasks.get('T_x'))
        T_y = np.array(tasks.get('T_y'))

    return T, ρ, t, u, v, ω, p, T_x, T_y

T_dat , ρ_dat, t_dat, u_dat, v_dat, ω_dat, p_dat, T_x_dat, T_y_dat = extraer_datos('../katabatic_wind_2/katabatic_wind_2_s1.h5')

plot_field = u_dat #np.sqrt(T_x_dat**2 + T_y_dat**2)
v_max = plot_field.max() #1500
v_min = plot_field.min()


nx = (524, 303)
dx, dy = 1.07/nx[0], 0.56/nx[1]
x, y = np.mgrid[slice(0, 1.07, dx), slice(0, 0.56, dy)]

k = 1

for i in range(0,40):

    if t_dat[i] < 100:
        time_str = str(t_dat[i])[:4]
    elif t_dat[i] >= 100:
        time_str = str(t_dat[i])[:5]
    elif t_dat[i] >= 1000:
        time_str = str(t_dat[i])[:6]

    data = ndimage.rotate(plot_field[i,:,:], -21, cval = np.nan)
    data = ma.masked_where(np.isnan(data), data)

    fig = plt.figure(figsize=((15,8.6)))
    ax = fig.add_subplot(111)
    p = ax.pcolormesh(x, y, data, cmap='RdBu_r', norm= colors.Normalize(vmin=v_min,vmax=v_max))
    ax.axis('off')
    plt.text(0.62, 0.056, 'Time = ' + time_str, fontsize=14)
    cbaxes = fig.add_axes([0.15, 0.2, 0.4, 0.02])
    fig.suptitle(r'u[$ m \ s^{-1}$]', fontsize = 14,  x = 0.35, y = 0.28)
    cbar = plt.colorbar(p, cax=cbaxes, orientation = 'horizontal')

    h = '%03d'%k
    plt.savefig('../snapshots/s_3/s3_kat'+ h)
    k += 1
    plt.close(fig)

print('**DONE**')
