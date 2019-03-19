import numpy as np
from mpi4py import MPI
from dedalus import public as de
from dedalus.extras import flow_tools
import time

import logging
logger = logging.getLogger(__name__)

# Definig the domain

Lx, Ly = (1, 0.2)
nx, ny = (512, 128)

# Constants
g = 9.8 #  m/s^2
κ = 1.3e-7 #thermal difussivity m^2/s
angle = 21

ρ0 = 1.2312  #characteristic density of air [kg/m^3]
P0 = 100000 #Pa
T_b = 5.0 # Bottom temperature [ºC]
T_top = 10.0 # Temperature at the top [ºC]

#Adimensional numbers
Reynolds = 10000 # Reynolds number
Peclet = 1e6 # Peclet number

#Definig the basis
x_basis = de.Fourier('x', nx, interval=(0, Lx))
y_basis = de.Chebyshev('y', ny, interval=(0, Ly))
domain = de.Domain([x_basis, y_basis], grid_dtype=np.float64)

#Setting type of solver

problem = de.IVP(domain, variables=['p', 'u', 'v', 'ρ', 'T', 'uy', 'vy', 'Ty']) #Initial Value Problem

problem.meta['p', 'T', 'u', 'v', 'ρ']['y']['dirichlet'] = True #Variables to solve

#Definig the parameters of the problem

#problem.parameters['P'] = (Rayleigh * Prandtl)**(-1/2)
problem.parameters['R'] = 287 #Pa m^3/(kg K)
problem.parameters['P0'] = P0
problem.parameters['κ'] = κ
problem.parameters['ρ0'] = ρ0
problem.parameters['g'] = g
problem.parameters['T_b'] = T_b
problem.parameters['T_top'] = T_top
problem.parameters['Re'] = Reynolds
problem.parameters['Pe'] = Peclet
problem.parameters['ang_y'] = np.cos(2*np.pi*21/360)
problem.parameters['ang_x'] = - np.sin(2*np.pi*21/360)
#Equations

problem.add_equation("dx(u) + vy = 0") #Mass conservation
problem.add_equation("dt(u) - (dx(dx(u)) + dy(uy))/Re + dx(p)/ρ0 = -(u*dx(u) + v*uy) - g*(ρ - ρ0)*ang_x/ρ0") #Navier-Stokes x
problem.add_equation("dt(v) - (dx(dx(v)) + dy(vy))/Re + dy(p)/ρ0 = -(u*dx(v) + v*vy) - g*(ρ - ρ0)*ang_y/ρ0") #Navier-Stokes y
problem.add_equation("ρ = P0/((T+273) * R)") #ideal gas equation.
problem.add_equation("dt(T) - 1/Pe*(dx(dx(T)) + dy(Ty)) = - u*dx(T) - v*Ty") #Energy conservation

#Defining the auxiliary variables
problem.add_equation("Ty - dy(T) = 0")
problem.add_equation("uy - dy(u) = 0")
problem.add_equation("vy - dy(v) = 0")

#Definig Boundary Conditions
problem.add_bc("left(T) = T_b")
problem.add_bc("right(T) = T_top")

problem.add_bc("left(u) = 0")
problem.add_bc("right(u) = 0")

problem.add_bc("left(v) = 0")
problem.add_bc("right(v) = 0", condition="(nx != 0)")

problem.add_bc("right(p) = 0", condition="(nx == 0)")

#Defining the solver used
solver = problem.build_solver(de.timesteppers.RK222) #Runge-Kutta order 2


# ## Initial conditions
x = domain.grid(0)
y = domain.grid(1)
T = solver.state['T']
Ty = solver.state['Ty']
ρ = solver.state['ρ']

yb, yt = y_basis.interval
x = domain.grid(0,scales=domain.dealias)
y = domain.grid(1,scales=domain.dealias)
xm, ym = np.meshgrid(x,y)

#Computing initial profile for density (maybe not necesary)
T_i = np.zeros((512,64))+ 10
#_i[0:20, :] = np.zeros((20,64)) + 5
T['g'] = T_i
ρ['g'] = P0/((T['g']+273) * 287)
# Initial timestep
dt = 0.001
# Integration parameters
solver.stop_sim_time = 20
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

# Analysis
snapshots = solver.evaluator.add_file_handler('../katabatic_stable_ic', sim_dt=0.25, max_writes=1200)
snapshots.add_system(solver.state)
snapshots.add_task("dx(v)- uy", name='vort')
snapshots.add_task("dx(T)", name='T_x')
snapshots.add_task("dy(T)", name='T_y')

# CFL
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=5, safety=0.1, max_change=1.5, min_change=0.5, max_dt=0.02, threshold=0.005)
CFL.add_velocities(('u', 'v'))

#Solver
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok:
        dt = CFL.compute_dt()
        dt = solver.step(dt)
        if (solver.iteration-1) % 10 == 0:
            # Update plot of scalar field
            logger.info('Iteration: %i, Time: %e, dt: %e' %(solver.iteration, solver.sim_time, dt))

except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
