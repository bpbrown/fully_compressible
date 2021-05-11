"""
Dedalus script for 2D compressible convection in a polytrope,
with specified number of density scale heights of stratification.

Usage:
    FC_poly.py [options]

Options:
    --Rayleigh=<Rayleigh>                Rayleigh number [default: 1e4]
    --Prandtl=<Prandtl>                  Prandtl number = nu/kappa [default: 1]
    --n_rho=<n_rho>                      Density scale heights across unstable layer [default: 3]
    --epsilon=<epsilon>                  The level of superadiabaticity of our polytrope background [default: 1e-4]
    --gamma=<gamma>                      Gamma of ideal gas (cp/cv) [default: 5/3]
    --aspect=<aspect_ratio>              Physical aspect ratio of the atmosphere [default: 4]

    --nz=<nz>                            vertical z (chebyshev) resolution [default: 128]
    --nx=<nx>                            Horizontal x (Fourier) resolution; if not set, nx=4*nz

    --run_time=<run_time>                Run time, in hours [default: 23.5]
    --run_time_buoy=<run_time_buoy>      Run time, in buoyancy times
    --run_time_iter=<run_time_iter>      Run time, number of iterations; if not set, n_iter=np.inf

    --label=<label>                      Additional label for run output directory
"""

import numpy as np
import dedalus.public as de
from dedalus.core import arithmetic, problems, solvers
import dedalus_sphere
from mpi4py import MPI
import time
rank = MPI.COMM_WORLD.rank

from docopt import docopt
args = docopt(__doc__)
from fractions import Fraction

from dedalus.tools.parsing import split_equation
from dedalus.extras.flow_tools import GlobalArrayReducer

import sys
import os
data_dir = sys.argv[0].split('.py')[0]
data_dir += "_nrho{}_Ra{}_Pr{}".format(args['--n_rho'], args['--Rayleigh'], args['--Prandtl'])
data_dir += "_eps{}_a{}".format(args['--epsilon'], args['--aspect'])

from dedalus.tools.config import config
config['logging']['filename'] = os.path.join(data_dir,'logs/dedalus_log')
config['logging']['file_level'] = 'DEBUG'
import logging
logger = logging.getLogger(__name__)

#Resolution
nz = int(args['--nz'])
nx = args['--nx']
if nx is not None:
    nx = int(nx)
else:
    nx = int(nz*float(args['--aspect']))

run_time_buoy = args['--run_time_buoy']
if run_time_buoy != None:
    run_time_buoy = float(run_time_buoy)

run_time_iter = args['--run_time_iter']
if run_time_iter != None:
    run_time_iter = int(float(run_time_iter))
else:
    run_time_iter = np.inf

Ra = Rayleigh = float(args['--Rayleigh']),
Pr = Prandtl = float(args['--Prandtl'])
γ  = float(Fraction(args['--gamma']))

# wrong height
Lx, Lz = (float(args['--aspect']), 1)
print(nx, Lx)

c = de.coords.CartesianCoordinates('x', 'z')
d = de.distributor.Distributor((c,))
xb = de.basis.ComplexFourier(c.coords[0], size=nx, bounds=(0, Lx))
zb = de.basis.ChebyshevT(c.coords[1], size=nz, bounds=(0, Lz))
x = xb.local_grid(1)
z = zb.local_grid(1)

# Fields
T = de.field.Field(name='p', dist=d, bases=(xb,zb), dtype=np.float64)
lnρ = de.field.Field(name='b', dist=d, bases=(xb,zb), dtype=np.float64)
u = de.field.Field(name='u', dist=d, bases=(xb,zb), dtype=np.float64, tensorsig=(c,))

# Taus
zb1 = de.basis.ChebyshevU(c.coords[1], size=nz, bounds=(0, Lz), alpha0=0)
τT1 = de.field.Field(name='t1', dist=d, bases=(xb,), dtype=np.float64)
τT2 = de.field.Field(name='t2', dist=d, bases=(xb,), dtype=np.float64)
τu1 = de.field.Field(name='t3', dist=d, bases=(xb,), dtype=np.float64, tensorsig=(c,))
τu2 = de.field.Field(name='t4', dist=d, bases=(xb,), dtype=np.float64, tensorsig=(c,))
P1 = de.field.Field(name='P1', dist=d, bases=(zb1,), dtype=np.float64)
if rank == 0:
    P1['c'][0,-1] = 1

# Parameters and operators
div = lambda A: de.operators.Divergence(A, index=0)
lap = lambda A: de.operators.Laplacian(A, c)
grad = lambda A: de.operators.Gradient(A, c)
curl = lambda A: de.operators.Curl(A)
dot = lambda A, B: arithmetic.DotProduct(A, B)
cross = lambda A, B: arithmetic.CrossProduct(A, B)
trace = lambda A: de.operators.Trace(A)
trans = lambda A: de.operators.TransposeComponents(A)
dt = lambda A: de.operators.TimeDerivative(A)
exp = lambda A: de.operators.UnaryGridFunctionField(np.exp, A)

dx = lambda A: de.operators.Differentiate(A, c.coords[0])
dz = lambda A: de.operators.Differentiate(A, c.coords[1])
P2 = dz(P1).evaluate()

ex = de.field.Field(name='ez', dist=d, bases=(zb,), dtype=np.float64, tensorsig=(c,))
ez = de.field.Field(name='ez', dist=d, bases=(zb,), dtype=np.float64, tensorsig=(c,))
ex['g'][0] = 1
ez['g'][1] = 1

T0 = de.field.Field(dist=d, bases=(zb,), dtype=np.float64)
grad_T0 = de.field.Field(dist=d, bases=(zb,), dtype=np.float64, tensorsig=(c,))
lnρ0 = de.field.Field(dist=d, bases=(zb,), dtype=np.float64)
grad_lnρ0 = de.field.Field(dist=d, bases=(zb,), dtype=np.float64, tensorsig=(c,))
ρ0_inv = de.field.Field(dist=d, bases=(zb,), dtype=np.float64)
grad_T0.set_scales(zb.dealias)
T0.set_scales(zb.dealias)
grad_lnρ0.set_scales(zb.dealias)

m = 1.5-1e-2
Tc = 1
lnρc = 1
grad_T0['g'][-1] = -1/(m+1)
T0['g'] = Tc + grad_T0['g'][-1]*z
grad_lnρ0['g'] = m*grad_T0['g']/T0['g']
lnρ0['g'] = lnρc + grad_lnρ0['g'][-1]*z
ρ0_inv['g'] = np.exp(-lnρ0['g'])

# stress-free bcs
#u_perp_inner = radial(angular(e(r=r_inner)))
#u_perp_outer = radial(angular(e(r=r_outer)))

e = grad(u) + trans(grad(u))
e.store_last = True

viscous_terms = div(e) - 2/3*grad(div(u))
trace_e = trace(e)
trace_e.store_last = True
Phi = 0.5*trace(dot(e, e)) - 1/3*(trace_e*trace_e)

Mach2 = ε = 1e-2
P = 0.01

def eq_eval(eq_str):
    return [eval(expr) for expr in split_equation(eq_str)]
# Υ = ln(ρ), θ = ln(h)
problem = problems.IVP([Υ, u, T, τu1, τu2, τT1, τT2])
problem.add_equation((dt(Υ) + div(u) + dot(u, grad(Υ0)) - P1*dot(ez,τu2), -dot(u, grad(Υ))))
problem.add_equation((dt(u) + grad(T) + T*grad(ln_rho0) + T0*grad(Υ) - ρ0_inv*viscous_terms - P1*τu1 - P2*τu2,
                      -dot(u,grad(u)) - T*grad(Υ) + ρ0_inv*(exp(-Υ)-1)*viscous_terms))
problem.add_equation((dt(T) + dot(u, grad(T0)) + (γ-1)*T0*div(u) + P1*τT1 + P2*τT2,
                      -dot(u,grad(T)) - (γ-1)*T*div(u) + ρ0_inv*(γ-1)*(exp(-Υ)-1)*lap(T)))
problem.add_equation((T(z=0), 0))
problem.add_equation((u(z=0), 0))
problem.add_equation((T(z=Lz), 0))
problem.add_equation((u(z=Lz), 0))
print("Problem built")

solver = solvers.InitialValueSolver(problem, de.timesteppers.RK222)
solver.stop_iteration = run_time_iter

cfl_cadence = 1
cfl_threshold = 0.1
cfl_safety_factor = 0.4
max_Δt = Δt = 0.1

CFL = flow_tools.CFL(solver, initial_dt=Δt, cadence=cfl_cadence, safety=cfl_safety_factor,
                     max_change=1.5, min_change=0.5, max_dt=max_Δt, threshold=cfl_threshold)
CFL.add_velocities(('u', 'w'))

good_solution = True
while solver.ok and good_solution:
    Δt = CFL.compute_dt()
    # advance
    solver.step(Δt)
    log_string = 'Iteration: {:5d}, Time: {:8.3e}, dt: {:8.3e}, '.format(solver.iteration, solver.sim_time, Δt)
    logger.info(log_string)
