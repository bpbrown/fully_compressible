"""
Dedalus script for 2D compressible convection in a polytrope,
with specified number of density scale heights of stratification.

Usage:
    FC_poly.py [options]

Options:
    --Rayleigh=<Rayleigh>                Rayleigh number (not used) [default: 1e4]
    --mu=<mu>                            Viscosity [default: 0.0015]
    --Prandtl=<Prandtl>                  Prandtl number = nu/kappa [default: 1]
    --n_rho=<n_rho>                      Density scale heights across unstable layer (not used) [default: 0.5]
    --n_h=<n_h>                          Enthalpy scale heights [default: 0.5]
    --epsilon=<epsilon>                  The level of superadiabaticity of our polytrope background [default: 0.5]
    --m=<m>                              Polytopic index of our polytrope
    --gamma=<gamma>                      Gamma of ideal gas (cp/cv) [default: 5/3]
    --aspect=<aspect_ratio>              Physical aspect ratio of the atmosphere [default: 4]

    --safety=<safety>                    CFL safety factor
    --SBDF2                              Use SBDF2

    --nz=<nz>                            vertical z (chebyshev) resolution [default: 64]
    --nx=<nx>                            Horizontal x (Fourier) resolution; if not set, nx=4*nz

    --run_time=<run_time>                Run time, in hours [default: 23.5]
    --run_time_buoy=<run_time_buoy>      Run time, in buoyancy times
    --run_time_iter=<run_time_iter>      Run time, number of iterations; if not set, n_iter=np.inf

    --ncc_cutoff=<ncc_cutoff>            Amplitude cutoff for NCCs [default: 1e-8]

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
from dedalus.extras import flow_tools

import sys
import os
import pathlib
import h5py

ncc_cutoff = float(args['--ncc_cutoff'])

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
    strat_label = 'm{}'.format(args['--m'])
else:
    m = m_ad - float(args['--epsilon'])
    strat_label = 'eps{}'.format(args['--epsilon'])
ε = m_ad - m

cP = γ/(γ-1)

data_dir = sys.argv[0].split('.py')[0]
#data_dir += "_nh{}_Ra{}_Pr{}".format(args['--n_h'], args['--Rayleigh'], args['--Prandtl'])
data_dir += "_nh{}_μ{}_Pr{}".format(args['--n_h'], args['--mu'], args['--Prandtl'])
data_dir += "_{}_a{}".format(strat_label, args['--aspect'])
data_dir += "_nz{:d}".format(nz)
if args['--label']:
    data_dir += '_{:s}'.format(args['--label'])

from dedalus.tools.config import config
config['logging']['filename'] = os.path.join(data_dir,'logs/dedalus_log')
config['logging']['file_level'] = 'DEBUG'
import logging
logger = logging.getLogger(__name__)

logger.info(args)
logger.info("saving data in: {}".format(data_dir))

with Sync() as sync:
    if sync.comm.rank == 0:
        if not os.path.exists('{:s}/'.format(data_dir)):
            os.mkdir('{:s}/'.format(data_dir))
        logdir = os.path.join(data_dir,'logs')
        if not os.path.exists(logdir):
            os.mkdir(logdir)



# this assumes h_bot=1, grad_φ = (γ-1)/γ (or L=Hρ)
h_bot = 1
h_slope = -1/(1+m)

n_h = float(args['--n_h'])
Lz = -1/h_slope*(1-np.exp(-n_h))
Lx = float(args['--aspect'])*Lz

dealias = 2
c = de.coords.CartesianCoordinates('x', 'z')
d = de.distributor.Distributor((c,))
xb = de.basis.RealFourier(c.coords[0], size=nx, bounds=(0, Lx), dealias=dealias)
zb = de.basis.ChebyshevT(c.coords[1], size=nz, bounds=(0, Lz), dealias=dealias)
x = xb.local_grid(1)
z = zb.local_grid(1)

# Fields
θ = de.field.Field(name='θ', dist=d, bases=(xb,zb), dtype=np.float64)
Υ = de.field.Field(name='Υ', dist=d, bases=(xb,zb), dtype=np.float64)
s = de.field.Field(name='s', dist=d, bases=(xb,zb), dtype=np.float64)
u = de.field.Field(name='u', dist=d, bases=(xb,zb), dtype=np.float64, tensorsig=(c,))

# Taus
#zb1 = de.basis.ChebyshevU(c.coords[1], size=nz, bounds=(0, Lz))
#zb1 = de.basis.ChebyshevT(c.coords[1], size=nz, bounds=(0, Lz))
zbr = zb.clone_with(a=zb.a+1, b=zb.b+1)
zb1 = zb.clone_with(a=zb.a+2, b=zb.b+2)
#zb1 = de.basis.ChebyshevV(c.coords[1], size=nz, bounds=(0, Lz))
τs1 = de.field.Field(name='τs1', dist=d, bases=(xb,), dtype=np.float64)
τs2 = de.field.Field(name='τs2', dist=d, bases=(xb,), dtype=np.float64)
τu1 = de.field.Field(name='τu1', dist=d, bases=(xb,), dtype=np.float64, tensorsig=(c,))
τu2 = de.field.Field(name='τu2', dist=d, bases=(xb,), dtype=np.float64, tensorsig=(c,))
P1 = de.field.Field(name='P1', dist=d, bases=(zb1,), dtype=np.float64)
P2 = de.field.Field(name='P2', dist=d, bases=(zb1,), dtype=np.float64)
if rank == 0:
    P1['c'][0,-1] = 1
    P2['c'][0,-2] = 1
#dz = lambda A: de.operators.Differentiate(A, c.coords[1])
#P2 = dz(P1).evaluate()

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
sqrt = lambda A: de.operators.UnaryGridFunction(np.sqrt, A)
integ = lambda A, C : de.operators.Integrate(A, C)
Coeff = de.operators.Coeff
Conv = de.operators.Convert

ex = de.field.Field(name='ex', dist=d, bases=(zb,), dtype=np.float64, tensorsig=(c,))
ez = de.field.Field(name='ez', dist=d, bases=(zb,), dtype=np.float64, tensorsig=(c,))
ex['g'][0] = 1
ez['g'][1] = 1

h0 = de.field.Field(name='h0', dist=d, bases=(zb,), dtype=np.float64)

h0['g'] = h_bot+h_slope*z
θ0 = log(h0).evaluate()
Υ0 = (m*θ0).evaluate()
s0 = (1/γ*θ0 - (γ-1)/γ*Υ0).evaluate()
ρ0_inv = exp(-Υ0).evaluate()

for f in [h0, θ0, Υ0, s0]:
    logger.info("{:} ranges from {:.2g}--{:.2g}".format(f, np.min(f['g']), np.max(f['g'])))


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

μ = float(args['--mu'])
κ = μ*cP/Pr # Mihalas & Mihalas eq (28.3)
s_bot = s0(z=0).evaluate()['g']
s_top = s0(z=Lz).evaluate()['g']

Ra_bot = (1/(μ*κ*cP)*exp(Υ0)(z=0)).evaluate()['g']
Ra_top = (1/(μ*κ*cP)*exp(Υ0)(z=Lz)).evaluate()['g']

if rank ==0:
    logger.info("Ra(z=0)   = {:.2g}".format(Ra_bot[0][0]))
    logger.info("Ra(z={:.1f}) = {:.2g}".format(Lz, Ra_top[0][0]))
    logger.info("Δs = {:.2g}".format(s_bot[0][0]-s_top[0][0]))

scale = de.field.Field(name='scale', dist=d, bases=(zb,), dtype=np.float64)
scale.require_scales(dealias)
scale['g'] = h0['g']

for ncc in [grad(Υ0), grad(h0), h0, exp(-Υ0), grad(s0)]:
    logger.info('scaled {:} has  {:} terms'.format(ncc,(np.abs((scale*ncc).evaluate()['c'])>ncc_cutoff).sum()))

# Υ = ln(ρ), θ = ln(h)
problem = problems.IVP([Υ, u, s, θ, τu1, τu2, τs1, τs2])
problem.add_equation((scale*(dt(Υ) + div(u) + dot(u, grad(Υ0)) - P1*dot(ez,τu2)),
                      Coeff(Conv(scale*(-dot(u, grad(Υ))) ,zbr)) ))
# check signs of terms in next equation for grad(h) terms...
problem.add_equation((scale*(dt(u) + Ma2*cP*(grad(h0*θ)) \
                      - Ma2*cP*(h0*grad(s) + h0*grad(s0)*θ) \
                      - μ*ρ0_inv*viscous_terms \
                      + P1*τu1 + μ*P2*τu2),
                      Coeff(Conv(
                      scale*(-dot(u,grad(u)) \
                                - Ma2*cP*(grad(h0*(exp(θ)-1-θ))) \
                                + Ma2*cP*(h0*(exp(θ)-1)*grad(s) + h0*grad(s0)*(exp(θ)-1-θ)) ),zbr)) )) # \
#                      + μ*exp(-Υ0)*(exp(-Υ)-1)*viscous_terms))) # nonlinear density effects on viscosity
problem.add_equation((scale*(dt(s) + dot(u,grad(s0)) - κ*ρ0_inv*lap(θ) + P1*τs1 + κ*ρ0_inv*P2*τs2),
                      Coeff(Conv(
                      scale*(-dot(u,grad(s)) + κ*dot(grad(θ),grad(θ)) ),zbr)) )) # need VH and nonlinear density effects on diffusion
                      #  κ*exp(-Υ0)*(exp(-Υ)-1)*lap(θ) + κ*exp(-Υ0-Υ)*dot(grad(θ),grad(θ)))
problem.add_equation((θ - (γ-1)*Υ - γ*s, 0)) #EOS, cP absorbed into s.
problem.add_equation((θ(z=0), 0))
#problem.add_equation((s(z=0), 0))
problem.add_equation((u(z=0), 0))
problem.add_equation((θ(z=Lz), 0))
#problem.add_equation((s(z=Lz), 0))
problem.add_equation((u(z=Lz), 0))
logger.info("Problem built")

# initial conditions
rng = np.random.default_rng(seed=42+rank)
noise = de.field.Field(name='noise', dist=d, bases=(xb, zb), dtype=np.float64)
noise['g'] = 2*rng.random(noise['g'].shape)-1 # -1--1 uniform distribution
noise.require_scales(0.25)
noise['g']
noise.require_scales(1)

amp = 1e-4*Ma2
s['g'] = amp*noise['g']*np.sin(np.pi*z/Lz)
Υ['g'] = -γ/(γ-1)*s['g']
θ['g'] = γ*s['g'] + (γ-1)*Υ['g']
for f in [s,Υ,θ]:
    logger.info("{}: {:.2g}--{:.2g}".format(f, np.min(f['g']), np.max(f['g'])))

if args['--SBDF2']:
    ts = de.timesteppers.SBDF2
    cfl_safety_factor = 0.3
else:
    ts = de.timesteppers.RK443
    cfl_safety_factor = 0.4
if args['--safety']:
    cfl_safety_factor = float(args['--safety'])

solver = solvers.InitialValueSolver(problem, ts, ncc_cutoff=ncc_cutoff)
solver.stop_iteration = run_time_iter

cfl_cadence = 1
cfl_threshold = 0.1
max_Δt = Δt = 10

dt = 1
cfl = flow_tools.CFL(solver, dt, safety=cfl_safety_factor, cadence=cfl_cadence, threshold=cfl_threshold,
                     max_change=1.5, min_change=0.5, max_dt=max_Δt)
cfl.add_velocity(u)

KE = 0.5*exp(Υ0+Υ)*dot(u,u)
IE = cP*Ma2*h0*exp(θ)*(s+s0)
Re = exp(Υ0+Υ)*sqrt(dot(u,u))/μ

slice_output = solver.evaluator.add_file_handler(data_dir+'/snapshots',sim_dt=0.125,max_writes=20)
slice_output.add_task(s+s0, name='s+s0')
slice_output.add_task(s, name='s')
slice_output.add_task(θ, name='θ')
#slice_output.add_task(dot(curl(u),curl(u)), name='enstrophy')

if rank == 0:
    scalar_file = pathlib.Path('{:s}/scalar_output.h5'.format(data_dir)).absolute()
    if os.path.exists(str(scalar_file)):
        scalar_file.unlink()
    scalar_f = h5py.File('{:s}'.format(str(scalar_file)), 'a')
    parameter_group = scalar_f.create_group('parameters')
    parameter_group['μ'] = μ
    parameter_group['Prandtl'] = Prandtl
    parameter_group['n_h'] = n_h
    parameter_group['nx'] = nx
    parameter_group['nz'] = nz

    scale_group = scalar_f.create_group('scales')
    scale_group.create_dataset(name='sim_time', shape=(0,), maxshape=(None,), dtype=np.float64)
    task_group = scalar_f.create_group('tasks')
    scalar_keys = ['KE', 'IE', 'Re', 'Re_max', 'τ_u1', 'τ_u2', 'τ_s1', 'τ_s2']
    for key in scalar_keys:
        task_group.create_dataset(name=key, shape=(0,), maxshape=(None,), dtype=np.float64)
    scalar_index = 0
    scalar_f.close()
    scalar_data ={}


report_cadence = 10
good_solution = True
KE_avg = 0

flow = flow_tools.GlobalFlowProperty(solver, cadence=report_cadence)
flow.add_property(Re, name='Re')
flow.add_property(KE, name='KE')
flow.add_property(IE, name='IE')
flow.add_property(τu1, name='τu1')
flow.add_property(τu2, name='τu2')
flow.add_property(τs1, name='τs1')
flow.add_property(τs2, name='τs2')


while solver.ok and good_solution:
    # advance
    solver.step(Δt)
    if solver.iteration % report_cadence == 0:
        KE_avg = flow.grid_average('KE')
        IE_avg = flow.grid_average('IE')
        Re_avg = flow.grid_average('Re')
        Re_max = flow.max('Re')
        τu1_max = flow.max('τu1')
        τu2_max = flow.max('τu2')
        τs1_max = flow.max('τs1')
        τs2_max = flow.max('τs2')
        log_string = 'Iteration: {:5d}, Time: {:8.3e}, dt: {:5.1e}, KE: {:.2g}, IE: {:.2g}, Re: {:.2g} ({:.2g})'.format(solver.iteration, solver.sim_time, Δt, KE_avg, IE_avg, Re_avg, Re_max)
        log_string += ' |τs| ({:.2g} {:.2g} {:.2g} {:.2g})'.format(τu1_max, τu2_max, τs1_max, τs2_max)
        logger.info(log_string)
        if rank == 0:
            scalar_data['KE'] = KE_avg
            scalar_data['IE'] = IE_avg
            scalar_data['Re'] = Re_avg
            scalar_data['Re_max'] = Re_max
            scalar_data['τ_u1'] = τu1_max
            scalar_data['τ_u2'] = τu2_max
            scalar_data['τ_s1'] = τs1_max
            scalar_data['τ_s2'] = τs2_max

            scalar_f = h5py.File('{:s}'.format(str(scalar_file)), 'a')
            scalar_f['scales/sim_time'].resize(scalar_index+1, axis=0)
            scalar_f['scales/sim_time'][scalar_index] = solver.sim_time
            for key in scalar_data:
                scalar_f['tasks/'+key].resize(scalar_index+1, axis=0)
                scalar_f['tasks/'+key][scalar_index] = scalar_data[key]
            scalar_index += 1
            scalar_f.close()
    Δt = cfl.compute_dt()
    good_solution = np.isfinite(Δt)*np.isfinite(KE_avg)
if not good_solution:
    logger.info("simulation terminated with good_solution = {}".format(good_solution))
    logger.info("Δt = {}".format(Δt))
    logger.info("KE = {}".format(KE_avg))
    logger.info("τu = {}".format((τu1_max,τu2_max,τs1_max,τs2_max)))
