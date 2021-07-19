"""
Dedalus script for 2D compressible convection in a polytrope,
with specified number of density scale heights of stratification.

Usage:
    FC_poly.py [options]

Options:
    --Rayleigh=<Rayleigh>                Rayleigh number [default: 1e4]
    --Prandtl=<Prandtl>                  Prandtl number = nu/kappa [default: 1]
    --n_rho=<n_rho>                      Density scale heights across unstable layer [default: 3]
    --n_h=<n_h>                          Enthalpy scale heights [default: 2]
    --epsilon=<epsilon>                  The level of superadiabaticity of our polytrope background [default: 1e-4]
    --m=<m>                              Polytopic index of our polytrope
    --gamma=<gamma>                      Gamma of ideal gas (cp/cv) [default: 5/3]
    --aspect=<aspect_ratio>              Physical aspect ratio of the atmosphere [default: 4]

    --nh=<nh>                            Enthalpy scale heights to use [default: 2]

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
from mpi4py import MPI
from dedalus.tools.parallel import Sync
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
data_dir += "_nh{}_Ra{}_Pr{}".format(args['--n_h'], args['--Rayleigh'], args['--Prandtl'])
data_dir += "_eps{}_a{}".format(args['--epsilon'], args['--aspect'])

from dedalus.tools.config import config
config['logging']['filename'] = os.path.join(data_dir,'logs/dedalus_log')
config['logging']['file_level'] = 'DEBUG'
import logging
logger = logging.getLogger(__name__)
with Sync() as sync:
    if sync.comm.rank == 0:
        if not os.path.exists('{:s}/'.format(data_dir)):
            os.mkdir('{:s}/'.format(data_dir))
        logdir = os.path.join(data_dir,'logs')
        if not os.path.exists(logdir):
            os.mkdir(logdir)


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

m_ad = 1/(γ-1)
if args['--m']:
    m = float(args['--m'])
else:
    m = m_ad - float(args['--epsilon'])

ε = m_ad - m

cP = γ/(γ-1)

# this assumes h_bot=1, grad_φ = (γ-1)/γ (or L=Hρ)
h_bot = 1
h_slope = -1/(1+m)

n_h = float(args['--n_h'])
Lz = -1/h_slope*(1-np.exp(-n_h))
Lx = float(args['--aspect'])*Lz
print((nz, Lz), (nx, Lx))

c = de.coords.CartesianCoordinates('x', 'z')
d = de.distributor.Distributor((c,))
xb = de.basis.RealFourier(c.coords[0], size=nx, bounds=(0, Lx))
zb = de.basis.ChebyshevT(c.coords[1], size=nz, bounds=(0, Lz))
x = xb.local_grid(1)
z = zb.local_grid(1)

# Fields
θ = de.field.Field(name='θ', dist=d, bases=(xb,zb), dtype=np.float64)
Υ = de.field.Field(name='Υ', dist=d, bases=(xb,zb), dtype=np.float64)
s = de.field.Field(name='s', dist=d, bases=(xb,zb), dtype=np.float64)
u = de.field.Field(name='u', dist=d, bases=(xb,zb), dtype=np.float64, tensorsig=(c,))

# Taus
zb1 = de.basis.ChebyshevU(c.coords[1], size=nz, bounds=(0, Lz), alpha0=0)
τs1 = de.field.Field(name='τs1', dist=d, bases=(xb,), dtype=np.float64)
τs2 = de.field.Field(name='τs2', dist=d, bases=(xb,), dtype=np.float64)
τu1 = de.field.Field(name='τu1', dist=d, bases=(xb,), dtype=np.float64, tensorsig=(c,))
τu2 = de.field.Field(name='τu2', dist=d, bases=(xb,), dtype=np.float64, tensorsig=(c,))
P1 = de.field.Field(name='P1', dist=d, bases=(zb1,), dtype=np.float64)
if rank == 0:
    P1['c'][0,-1] = 1
dz = lambda A: de.operators.Differentiate(A, c.coords[1])
P2 = dz(P1).evaluate()

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
exp = lambda A: de.operators.UnaryGridFunction(np.exp, A)
log = lambda A: de.operators.UnaryGridFunction(np.log, A)
integ = lambda A, C : de.operators.Integrate(A, C)

o = de.field.Field(name='s', dist=d, bases=(xb,zb), dtype=np.float64)
o['g'] = 1
logger.warning(integ(integ(o,'x'),'z').evaluate()['g'])


ex = de.field.Field(name='ex', dist=d, bases=(zb,), dtype=np.float64, tensorsig=(c,))
ez = de.field.Field(name='ez', dist=d, bases=(zb,), dtype=np.float64, tensorsig=(c,))
ex['g'][0] = 1
ez['g'][1] = 1

h0 = de.field.Field(name='h0', dist=d, bases=(zb,), dtype=np.float64)

h0['g'] = h_bot+h_slope*z
θ0 = log(h0).evaluate()
Υ0 = (m*θ0).evaluate()
s0 = (1/γ*θ0 - (γ-1)/γ*Υ0).evaluate()

# stress-free bcs
#u_perp_inner = radial(angular(e(r=r_inner)))
#u_perp_outer = radial(angular(e(r=r_outer)))

e = grad(u) + trans(grad(u))
e.store_last = True

viscous_terms = div(e) - 2/3*grad(div(u))
trace_e = trace(e)
trace_e.store_last = True
Phi = 0.5*trace(dot(e, e)) - 1/3*(trace_e*trace_e)

Ma2 = ε
Pr = 1

μ = 0.001
κ = μ*cP/Pr # Mihalas & Mihalas eq (28.3)

# Υ = ln(ρ), θ = ln(h)
problem = problems.IVP([Υ, u, s, θ, τu1, τu2, τs1, τs2])
problem.add_equation((dt(Υ) + div(u) + dot(u, grad(Υ0)) - P1*dot(ez,τu2), -dot(u, grad(Υ))))
problem.add_equation((dt(u) + Ma2*(γ/(γ-1)*grad(h0*θ) - h0*grad(s) - h0*θ*grad(s0)) \
                      -μ*exp(-Υ0)*viscous_terms + P1*τu1 + P2*τu2,
                      -dot(u,grad(u)) + Ma2*(γ/(γ-1)*grad(h0*(exp(θ)-1-θ)) + h0*(exp(θ)-1)*grad(s) + h0*(exp(θ)-1-θ)*grad(s0)))) #\
#                      + μ*exp(-Υ0)*(exp(-Υ)-1)*viscous_terms)) # nonlinear density effects on viscosity
problem.add_equation((dt(s) + dot(u,grad(s0)) - κ*exp(-Υ0)*lap(θ) + P1*τs1 + P2*τs2,
                      -dot(u,grad(s)) + κ*exp(-Υ0-Υ)*dot(grad(θ),grad(θ)))) # need VH and nonlinear density effects on diffusion
problem.add_equation((θ - (γ-1)*Υ - γ*s, 0)) #EOS, cP absorbed into s.
problem.add_equation((θ(z=0), 0))
problem.add_equation((u(z=0), 0))
problem.add_equation((θ(z=Lz), 0))
problem.add_equation((u(z=Lz), 0))
logger.info("Problem built")

# initial conditions
noise = de.field.Field(name='noise', dist=d, bases=(xb, zb), dtype=np.float64)
noise['g'] = np.random.randn(*noise['g'].shape)
amp = 1e-4*Ma2
s['g'] = amp*noise['g']*np.sin(np.pi*z/Lz)

solver = solvers.InitialValueSolver(problem, de.timesteppers.SBDF2)
solver.stop_iteration = run_time_iter

cfl_cadence = 1
cfl_threshold = 0.1
cfl_safety_factor = 0.4
max_Δt = Δt = 0.1

# CFL = flow_tools.CFL(solver, initial_dt=Δt, cadence=cfl_cadence, safety=cfl_safety_factor,
#                      max_change=1.5, min_change=0.5, max_dt=max_Δt, threshold=cfl_threshold)
# CFL.add_velocities(('u', 'w'))

Δt = 1e-2

KE = 0.5*exp(Υ0+Υ)*dot(u,u)
IE = h0*exp(θ)*(s+s0)

reducer = GlobalArrayReducer(d.comm_cart)

dz = Lz/nz
dx = Lx/nx
safety = 0.4
def compute_dt(dt_old, threshold=0.1, dt_max=1e-2):
  local_freq = np.abs(u['g'][1]/dz) + np.abs(u['g'][0]/dx)
  global_freq = reducer.global_max(local_freq)
  if global_freq == 0.:
      dt = np.inf
  else:
      dt = 1 / global_freq
  dt *= safety
  if dt > dt_max: dt = dt_max
  if dt < dt_old*(1+threshold) and dt > dt_old*(1-threshold): dt = dt_old
  return dt

# need integration weights
def vol_avg(q):
    Q = integ(integ(q,'x'),'z').evaluate()['g']
    return reducer.reduce_scalar(Q, MPI.SUM)

def L_inf(q):
    if q['g'].size == 0:
        Q = 0
    else:
        Q = np.max(np.abs(q['g']))
    return reducer.reduce_scalar(Q, MPI.MAX)

slice_output = solver.evaluator.add_file_handler(data_dir+'/snapshots',sim_dt=1,max_writes=10)
slice_output.add_task(s+s0, name='s+s0')
slice_output.add_task(s, name='s')
#slice_output.add_task(dot(curl(u),curl(u)), name='enstrophy')

report_cadence = 10
print(vol_avg(o))
good_solution = True
while solver.ok and good_solution:
    # advance
    solver.step(Δt)
    if solver.iteration % report_cadence == 0:
        KE_avg = vol_avg(KE.evaluate())
        IE_avg = vol_avg(IE.evaluate())

        log_string = 'Iteration: {:5d}, Time: {:8.3e}, dt: {:8.3e}, KE: {:.2g}, IE: {:.2g}'.format(solver.iteration, solver.sim_time, Δt, KE_avg, IE_avg)
        log_string += ' |τs| ({:.2g} {:.2g} {:.2g} {:.2g})'.format(L_inf(τu1), L_inf(τu2), L_inf(τs1), L_inf(τs2))
        logger.info(log_string)
    Δt = compute_dt(Δt)
    good_solution = np.isfinite(Δt)
