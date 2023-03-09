"""
Dedalus script for 2D compressible convection in a polytrope,
with specified number of density scale heights of stratification.

Usage:
    FC_poly.py [options]

Options:
    --mu=<mu>                            Viscosity [default: 0.07]
    --Prandtl=<Prandtl>                  Prandtl number = nu/kappa [default: 1]
    --n_h=<n_h>                          Enthalpy scale heights [default: 0.5]
    --epsilon=<epsilon>                  The level of superadiabaticity of our polytrope background [default: 0.5]
    --m=<m>                              Polytopic index of our polytrope
    --gamma=<gamma>                      Gamma of ideal gas (cp/cv) [default: 5/3]
    --aspect=<aspect_ratio>              Physical aspect ratio of the atmosphere [default: 4]

    --safety=<safety>                    CFL safety factor
    --SBDF2                              Use SBDF2
    --max_dt=<max_dt>                    Largest timestep; also sets initial dt [default: 1]

    --nz=<nz>                            vertical z (chebyshev) resolution [default: 64]
    --nx=<nx>                            Horizontal x (Fourier) resolution; if not set, nx=4*nz

    --run_time=<run_time>                Run time, in hours [default: 23.5]
    --run_time_buoy=<run_time_buoy>      Run time, in buoyancy times
    --run_time_iter=<run_time_iter>      Run time, number of iterations; if not set, n_iter=np.inf

    --ncc_cutoff=<ncc_cutoff>            Amplitude cutoff for NCCs [default: 1e-8]

    --whole_sun

    --label=<label>                      Additional label for run output directory
"""

import numpy as np
from mpi4py import MPI
rank = MPI.COMM_WORLD.rank
from dedalus.tools.parallel import Sync

from docopt import docopt
args = docopt(__doc__)
from fractions import Fraction

import sys
import os
import pathlib
import h5py

import logging
logger = logging.getLogger(__name__)

dlog = logging.getLogger('evaluator')
dlog.setLevel(logging.WARNING)

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
data_dir += "_nh{}_μ{}_Pr{}".format(args['--n_h'], args['--mu'], args['--Prandtl'])
data_dir += "_{}_a{}".format(strat_label, args['--aspect'])
data_dir += "_nz{:d}_nx{:d}".format(nz,nx)
if args['--whole_sun']:
    data_dir += '_wholesun'
if args['--label']:
    data_dir += '_{:s}'.format(args['--label'])

import dedalus.tools.logging as dedalus_logging
dedalus_logging.add_file_handler(data_dir+'/logs/dedalus_log', 'DEBUG')

import dedalus.public as de
from dedalus.extras import flow_tools

logger.info(args)
logger.info("saving data in: {}".format(data_dir))

if args['--whole_sun']:
    theta = 10
    h_bot = theta+1
    h_slope = -theta
    Lz = 1
else:
    # this assumes h_bot=1, grad_φ = (γ-1)/γ (or L=Hρ)
    h_bot = 1
    h_slope = -1/(1+m)
    grad_φ = (γ-1)/γ
    n_h = float(args['--n_h'])
    Lz = -1/h_slope*(1-np.exp(-n_h))

Lx = float(args['--aspect'])*Lz

dealias = 2
c = de.CartesianCoordinates('x', 'y', 'z')
d = de.Distributor(c, dtype=np.float64)
xb = de.RealFourier(c.coords[0], size=nx, bounds=(0, Lx), dealias=dealias)
zb = de.ChebyshevT(c.coords[2], size=nz, bounds=(0, Lz), dealias=dealias)
b = (xb, zb)
x = xb.local_grid(1)
z = zb.local_grid(1)

# Fields
θ = d.Field(name='θ', bases=b)
Υ = d.Field(name='Υ', bases=b)
s = d.Field(name='s', bases=b)
u = d.VectorField(c, name='u', bases=b)

# Taus
zb1 = zb.clone_with(a=zb.a+1, b=zb.b+1)
zb2 = zb.clone_with(a=zb.a+2, b=zb.b+2)
lift1 = lambda A, n: de.Lift(A, zb1, n)
lift = lambda A, n: de.Lift(A, zb2, n)
τ_s1 = d.Field(name='τs1', bases=xb)
τ_s2 = d.Field(name='τs2', bases=xb)
τ_u1 = d.VectorField(c, name='τu1', bases=(xb,))
τ_u2 = d.VectorField(c, name='τu2', bases=(xb,))

# Parameters and operators
ddt = lambda A: de.TimeDerivative(A)
div = lambda A: de.Divergence(A, index=0)
lap = lambda A: de.Laplacian(A, c)
grad = lambda A: de.Gradient(A, c)
curl = lambda A: de.Curl(A)
trace = lambda A: de.Trace(A)
trans = lambda A: de.TransposeComponents(A)

integ = lambda A: de.Integrate(de.Integrate(A, 'x'), 'z')
avg = lambda A: integ(A)/(Lx*Lz)
x_avg = lambda A: de.Integrate(A, 'x')/(Lx)

ex, ey, ez = c.unit_vector_fields(d)

h0 = d.Field(name='h0', bases=zb)
h0['g'] = h_bot+h_slope*z

θ0 = np.log(h0).evaluate()
θ0.name = 'θ0'
Υ0 = (m*(θ0)).evaluate() # normalize to zero at bottom
Υ0.name = 'Υ0'
s0 = (1/γ*θ0 - (γ-1)/γ*Υ0).evaluate()
s0.name = 's0'
ρ0 = np.exp(Υ0).evaluate()
ρ0.name = 'ρ0'
grad_s0 = grad(s0).evaluate()
grad_θ0 = grad(θ0).evaluate()
grad_h0 = grad(h0).evaluate()
grad_Υ0 = grad(Υ0).evaluate()

h0_g = de.Grid(h0).evaluate()
h0_grad_s0_g = de.Grid(h0*grad(s0)).evaluate()

ρ0_g = de.Grid(ρ0).evaluate()
ρ0_h0_g = de.Grid(ρ0*h0).evaluate()
ρ0_grad_h0_g = de.Grid(ρ0*grad(h0)).evaluate()
ρ0_h0_grad_s0_g = de.Grid(ρ0*h0*grad(s0)).evaluate()

# it's a polytrope, so the zero state is in thermal equilibrium.
# θ0_RHS = dist.Field(name='θ0_RHS', bases=b)
# θ0.change_scales(1)
# θ0_RHS.require_grid_space()
# if θ0['g'].size > 0:
#     θ0_RHS['g'] = θ0['g']


# stress-free bcs
e = grad(u) + trans(grad(u))

viscous_terms = div(e) - 2/3*grad(div(u))
trace_e = trace(e)
Phi = 0.5*trace(e@e) - 1/3*(trace_e*trace_e)

Ma2 = 1 #ε
Pr = 1

scrR = float(args['--mu'])
scrP = scrR*Prandtl # Mihalas & Mihalas eq (28.3)

s_bot = s0(z=0).evaluate()['g']
s_top = s0(z=Lz).evaluate()['g']

delta_s = s_bot-s_top
# g = m+1
# pre = g*(delta_s)*Lz**3
# Ra_bot = pre*(1/(μ*κ*cP)*np.exp(2*Υ0)(z=0)).evaluate()['g']
# Ra_mid = pre*(1/(μ*κ*cP)*np.exp(2*Υ0)(z=Lz/2)).evaluate()['g']
# Ra_top = pre*(1/(μ*κ*cP)*np.exp(2*Υ0)(z=Lz)).evaluate()['g']

Υ_bot = Υ0(z=0).evaluate()['g']
Υ_top = Υ0(z=Lz).evaluate()['g']

θ_bot = θ0(z=0).evaluate()['g']
θ_top = θ0(z=Lz).evaluate()['g']

if rank ==0:
    # logger.info("Ra(z=0)   = {:.2g}".format(Ra_bot[0][0]))
    # logger.info("Ra(z={:.1f}) = {:.2g}".format(Lz/2, Ra_mid[0][0]))
    # logger.info("Ra(z={:.1f}) = {:.2g}".format(Lz, Ra_top[0][0]))
    logger.info("Δs = {:.2g} ({:.2g} to {:.2g})".format(s_bot[0,0,0]-s_top[0,0,0],s_bot[0,0,0],s_top[0,0,0]))
    logger.info("Δθ = {:.2g} ({:.2g} to {:.2g})".format(θ_bot[0,0,0]-θ_top[0,0,0],θ_bot[0,0,0],θ_top[0,0,0]))
    logger.info("ΔΥ = {:.2g} ({:.2g} to {:.2g})".format(Υ_bot[0,0,0]-Υ_top[0,0,0],Υ_bot[0,0,0],Υ_top[0,0,0]))


logger.info("NCC expansions:")
for ncc in [ρ0, ρ0*grad_h0, ρ0*h0, ρ0*h0*grad_s0, h0*grad_θ0, h0*grad_Υ0]:
    logger.info("{}: {}".format(ncc.evaluate(), np.where(np.abs(ncc.evaluate()['c']) >= ncc_cutoff)[0].shape))

# Υ = ln(ρ), θ = ln(h)
problem = de.IVP([Υ, u, s, θ, τ_u1, τ_u2, τ_s1, τ_s2])
# check signs of terms in next equation for grad(h) terms...
problem.add_equation((ρ0*ddt(u)
                      + ρ0*grad_h0*θ
                      + ρ0*h0*grad(θ)
                      - ρ0*h0*grad(s)
                      - ρ0*h0*grad_s0*θ
                      - scrR*viscous_terms
                      + lift(τ_u1,-1) + lift(τ_u2,-2),
                      -ρ0_g*(u@grad(u))
                      -ρ0_grad_h0_g*(np.expm1(θ)-θ)
                      -ρ0_h0_g*np.expm1(θ)*grad(θ)
                      +ρ0_h0_g*np.expm1(θ)*grad(s)
                      +ρ0_h0_grad_s0_g*(np.expm1(θ)-θ) ))
problem.add_equation((h0*(ddt(Υ) + div(u) + u@grad_Υ0) + 1/scrR*lift(τ_u2,-1)@ez,
                      -h0_g*(u@grad(Υ)) ))
problem.add_equation((h0*ρ0*(ddt(s))
                      - h0*scrP*(lap(θ) + 2*grad_θ0@grad(θ))
                      + lift(τ_s1,-1) + lift(τ_s2,-2),
                      - ρ0_h0_g*(u@grad(s))
                      + h0_g*scrP*grad(θ)@grad(θ) ))
                      #+ Ek/Co2*0.5*h0_inv_g*Phi )) # figure this one out
problem.add_equation((θ - (γ-1)*Υ - γ*s, 0)) #EOS, cP absorbed into s.
# boundary conditions
problem.add_equation((θ(z=0), 0))
problem.add_equation((ez@u(z=0), 0))
problem.add_equation((ez@(ex@e(z=0)), 0))
problem.add_equation((ez@(ey@e(z=0)), 0))
problem.add_equation((θ(z=Lz), 0))
problem.add_equation((ez@u(z=Lz), 0))
problem.add_equation((ez@(ex@e(z=Lz)), 0))
problem.add_equation((ez@(ey@e(z=Lz)), 0))
logger.info("Problem built")

# initial conditions
amp = 1e-4*Ma2

zb, zt = zb.bounds
noise = d.Field(name='noise', bases=b)
noise.fill_random('g', seed=42, distribution='normal', scale=amp) # Random noise
noise.low_pass_filter(scales=0.25)

s['g'] = noise['g']*np.sin(np.pi*z/Lz)
Υ['g'] = -γ/(γ-1)*s['g']
θ['g'] = γ*s['g'] + (γ-1)*Υ['g']

if args['--SBDF2']:
    ts = de.SBDF2
    cfl_safety_factor = 0.3
else:
    ts = de.RK443
    cfl_safety_factor = 0.4
if args['--safety']:
    cfl_safety_factor = float(args['--safety'])

solver = problem.build_solver(ts)
solver.stop_iteration = run_time_iter

Δt = max_Δt = float(args['--max_dt'])
cfl = flow_tools.CFL(solver, Δt, safety=cfl_safety_factor, cadence=1, threshold=0.1,
                     max_change=1.5, min_change=0.5, max_dt=max_Δt)
cfl.add_velocity(u)

ρ = np.exp(Υ0+Υ).evaluate()
h = h0*np.exp(θ).evaluate()
KE = 0.5*ρ*u@u
IE = cP*Ma2*h*(s+s0)
Re = (ρ/scrR)*np.sqrt(u@u)
ω = curl(u)
κ = 1/scrP

slice_dt = 0.5/np.sqrt(ε)

slice_output = solver.evaluator.add_file_handler(data_dir+'/slices',sim_dt=slice_dt,max_writes=20)
slice_output.add_task(s+s0, name='s+s0')
slice_output.add_task(s, name='s')
slice_output.add_task(θ, name='θ')
slice_output.add_task(ω, name='vorticity')
slice_output.add_task(ω**2, name='enstrophy')
slice_output.add_task(x_avg(-κ*grad(h)@ez/cP), name='F_κ')
slice_output.add_task(x_avg(0.5*ρ*u@ez*u@u), name='F_KE')
slice_output.add_task(x_avg(u@ez*h), name='F_h')
slice_output.add_task(grad(s0), name='grad_s0')
slice_output.add_task(x_avg(grad(s0+s)), name='grad_s')
slice_output.add_task(s0, name='s0(z)')
slice_output.add_task(x_avg(s0+s), name='s(z)')


traces = solver.evaluator.add_file_handler(data_dir+'/traces', sim_dt=0.1, max_writes=None)
traces.add_task(avg(0.5*ρ*u@u), name='KE')
traces.add_task(avg(IE), name='IE')
traces.add_task(avg(Re), name='Re')
traces.add_task(avg(ω**2), name='enstrophy')
traces.add_task(x_avg(np.sqrt(τ_u1@τ_u1)), name='τ_u1')
traces.add_task(x_avg(np.sqrt(τ_u2@τ_u2)), name='τ_u2')
traces.add_task(x_avg(np.sqrt(τ_s1**2)), name='τ_s1')
traces.add_task(x_avg(np.sqrt(τ_s2**2)), name='τ_s2')

report_cadence = 10
good_solution = True

flow = flow_tools.GlobalFlowProperty(solver, cadence=report_cadence)
flow.add_property(Re, name='Re')
flow.add_property(KE, name='KE')
flow.add_property(IE, name='IE')
flow.add_property(τ_u1, name='τ_u1')
flow.add_property(τ_u2, name='τ_u2')
flow.add_property(τ_s1, name='τ_s1')
flow.add_property(τ_s2, name='τ_s2')

KE_avg = 0
while solver.proceed and good_solution:
    # advance
    solver.step(Δt)
    if solver.iteration % report_cadence == 0:
        KE_avg = flow.grid_average('KE')
        IE_avg = flow.grid_average('IE')
        Re_avg = flow.grid_average('Re')
        Re_max = flow.max('Re')
        τu1_max = flow.max('τ_u1')
        τu2_max = flow.max('τ_u2')
        τs1_max = flow.max('τ_s1')
        τs2_max = flow.max('τ_s2')
        τ_max = np.max([τu1_max,τu2_max,τs1_max,τs2_max])
        log_string = 'Iteration: {:5d}, Time: {:8.3e} ({:.1e}), dt: {:5.1e}'.format(solver.iteration, solver.sim_time, solver.sim_time*scrR, Δt)
        log_string += ', KE: {:.2g}, IE: {:.2g}, Re: {:.2g} ({:.2g})'.format(KE_avg, IE_avg, Re_avg, Re_max)
        log_string += ', τ: {:.2g}'.format(τ_max)
        logger.info(log_string)
    Δt = cfl.compute_timestep()
    good_solution = np.isfinite(Δt)*np.isfinite(KE_avg)

if not good_solution:
    logger.info("simulation terminated with good_solution = {}".format(good_solution))
    logger.info("Δt = {}".format(Δt))
    logger.info("KE = {}".format(KE_avg))
    logger.info("τu = {}".format((τu1_max,τu2_max,τs1_max,τs2_max)))

solver.log_stats()
logger.debug("mode-stages/DOF = {}".format(solver.total_modes/(nx*nz)))
