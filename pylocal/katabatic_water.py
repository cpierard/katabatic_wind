import numpy as np
from mpi4py import MPI
from dedalus import public as de
from dedalus.extras import flow_tools
import time

import logging
logger = logging.getLogger(__name__)

# Definig the domain

Lx, Ly = (1, 0.2)
nx, ny = (512, 64)

# Some constants (Taken form D. Leconet Article see Readme.md)

ν = 1.8e-6 # m^2/s viscocidad cinemática
k = 2e-5 # s^-1
g = 9.8 #  m/s^2
κ = 1.3e-7 #thermal difussivity m^2/s
α = 0.15 # kg/m^3ºC thermal expansion
β = 0.78 # kg/m^3%0 salinity contraction
angle = 21



# Initial stratification profile (see notebook ...)

ρ0 = 1006.75  #Initial density mean [kg/m^3]
#s_top = 0. #Salinity at the top [parts per thousand]
#s_bot = 12.5 #salinity at the bottom [parts per thousand]
#s0 = 12.5

#z_int = 0.08 # Convection height [m]

T0 = 10.0 # Mean of the initial temperature on the profile [ºC]
T_b = 5.0 # Bottom temperature [ºC]
T_air = 10. # Air Temperature [ºC]
T_top = 10.0 # Temperature at the top [ºC]

#Adimensional numbers
L_conv = 0.35 # Don't recall this guy
Reynolds = 100 # Reynolds number
Schmidt = 1000 # Schmidt number
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
#problem.parameters['R'] = (Rayleigh / Prandtl)**(-1/2)
#problem.parameters['F'] = F = 1
problem.parameters['ν'] = ν
problem.parameters['κ'] = κ
problem.parameters['T_air'] = T_air
problem.parameters['k'] = k
problem.parameters['ρ0'] = ρ0
problem.parameters['g'] = 9.8
problem.parameters['α'] = α
problem.parameters['β'] = β
problem.parameters['T_0'] = T0 #4.0 ºC
problem.parameters['T_b'] = T_b
problem.parameters['T_top'] = T_top
problem.parameters['Lx'] = Lx
problem.parameters['Re'] = Reynolds
problem.parameters['Sc'] = Schmidt
problem.parameters['Pe'] = Peclet
problem.parameters['ang_y'] = np.cos(2*np.pi*21/360)
problem.parameters['ang_x'] = np.sin(2*np.pi*21/360)
#Equations

problem.add_equation("dx(u) + vy = 0") #Mass conservation
problem.add_equation("dt(u) - ν*(dx(dx(u)) + dy(uy)) + dx(p)/ρ0 = -(u*dx(u) + v*uy) + g*(ρ - ρ0)*ang_x/ρ0") #Navier-Stokes x
problem.add_equation("dt(v) - ν*(dx(dx(v)) + dy(vy)) + dy(p)/ρ0 = -(u*dx(v) + v*vy) - g*(ρ - ρ0)*ang_y/ρ0") #Navier-Stokes y
problem.add_equation("ρ = ρ0 - α*(T - T_0)") #Equation of state of water #change it for ideal gas equation.
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

#Definig the function for the initial stratification profile (notebook ...)
#def perfil_arriba(x):
#    return -s_bot/(Ly - z_int)*x -(-s_bot/(Ly - z_int))*Ly

yb, yt = y_basis.interval

x = domain.grid(0,scales=domain.dealias)
y = domain.grid(1,scales=domain.dealias)
xm, ym = np.meshgrid(x,y)

#Computing initial profile for density (maybe not necesary)
T_i = np.zeros((512,32))+ 10
T_i[180:200, :20] = np.ones((20,20))*5
T_i[140:160, :20] = np.ones((20,20))*5
T['g'] = T_i
ρ['g'] = ρ0 - α*(T['g'] - T0)

# Initial timestep
dt = 0.02
# Integration parameters
solver.stop_sim_time = 30
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf

# Analysis
snapshots = solver.evaluator.add_file_handler('../katabatic_water_2', sim_dt=0.15, max_writes=1200) #Name of file, simulation time step, and max number of steps for file
snapshots.add_system(solver.state)
#snapshots.add_task("integ(s,'x')/Lx", name='s profile') #Computes the mean of salinity per layer
#snapshots.add_task("integ(T,'x')/Lx", name='T profile') #Computes the mean of Temperature per layer
#snapshots.add_task("integ(ρ,'x')/Lx", name='ρ profile') #Computes the mean of density per layer
#snapshots.add_task("-g/ρ0*dy(ρ)", name='NN') #Computes Brunt-Vaisala for the domain
snapshots.add_task("dx(v)-dy(u)", name='vort')

# CFL
#CFL = flow_tools.CFL(solver, initial_dt = dt, max_change = 0.5)
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=5, safety=0.1, max_change=1.5, min_change=0.5, max_dt=0.02, threshold=0.01)
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
