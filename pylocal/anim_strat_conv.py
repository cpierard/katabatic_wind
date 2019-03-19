import h5py
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
#from IPython import display
#from matplotlib import animation
#import matplotlib.colors as colors
#from mpl_toolkits.axes_grid1 import make_axes_locatable

#Función para extraer datos del archivo hdf5

dx, dy = 1/512., 0.2/64.

x, y = np.mgrid[slice(0, 1, dx), slice(0, 0.2, dy)]

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
        #s = np.array(tasks.get('s'))
        t = np.array(scales.get('sim_time'))
        #NN = np.array(tasks.get('NN'))
        print('Exportando...')

    return T, ρ, t

#Abajo tienes que poner el nombre del archivo hdf5 en donde guardaste los datos.

T_dat , ρ_dat, t_dat, = extraer_datos('katabatic_water/katabatic_water_s1/katabatic_water_s1_p0.h5')

k = 1

print(T_dat.shape)

for i in range(0,40):

    plt.figure(figsize=((15,3)))
    print('Gráfica: ', i)
    p = plt.pcolormesh(x, y, T_dat[i,:,:], cmap='rainbow')
    plt.colorbar(p)
    plt.title(str(t_dat[0]))
    plt.ylim(0,0.2)

    h = '%03d'%k
    plt.savefig('snapshots/s_1/s1_kat'+ h)
    k += 1
    plt.close(fig)

print('**DONE**')
