"""
Dedalus script for 2D compressible convection in a polytrope,
with specified number of density scale heights of stratification.

Usage:
    FC_poly.py [options]

Options:
    --R=<R>                              Reynolds number [default: 1e2]
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
    --nx=<nx>                            Horizontal x (Fourier) resolution; if not set, nx=aspect*nz

    --run_time=<run_time>                Run time, in hours [default: 23.5]
    --run_time_buoy=<run_time_buoy>      Run time, in buoyancy times
    --run_time_iter=<run_time_iter>      Run time, number of iterations; if not set, n_iter=np.inf

    --ncc_cutoff=<ncc_cutoff>            Amplitude cutoff for NCCs [default: 1e-8]

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

R = float(args['--R'])
R_inv = scrR = 1/R
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

Ma2 = ε
scrM = 1/Ma2
s_c_over_c_P = scrS = 1 # s_c/c_P = 1

logger.info("Ma2 = {:.3g}, R = {:.3g}, R_inv = {:.3g}, Pr = {:.3g}, γ = {:.3g}".format(Ma2, R, R_inv, Pr, γ))

data_dir = sys.argv[0].split('.py')[0]
data_dir += "_nh{}_R{}_Pr{}".format(args['--n_h'], args['--R'], args['--Prandtl'])
data_dir += "_{}_a{}".format(strat_label, args['--aspect'])
data_dir += "_nz{:d}_nx{:d}".format(nz,nx)
if args['--label']:
    data_dir += '_{:s}'.format(args['--label'])

from dedalus.tools.config import config
config['logging']['filename'] = os.path.join(data_dir,'logs/dedalus_log')
config['logging']['file_level'] = 'DEBUG'

with Sync() as sync:
    if sync.comm.rank == 0:
        if not os.path.exists('{:s}/'.format(data_dir)):
            os.mkdir('{:s}/'.format(data_dir))
        logdir = os.path.join(data_dir,'logs')
        if not os.path.exists(logdir):
            os.mkdir(logdir)

import dedalus.public as de
from dedalus.extras import flow_tools

logger.info(args)
logger.info("saving data in: {}".format(data_dir))


# this assumes h_bot=1, grad_φ = (γ-1)/γ (or L=Hρ)
h_bot = 1
h_slope = -1/(1+m)
grad_φ = (γ-1)/γ

n_h = float(args['--n_h'])
Lz = -1/h_slope*(1-np.exp(-n_h))
Lx = float(args['--aspect'])*Lz

dealias = 2
c = de.CartesianCoordinates('x', 'z')
d = de.Distributor(c, dtype=np.float64)
xb = de.RealFourier(c.coords[0], size=nx, bounds=(0, Lx), dealias=dealias)
zb = de.ChebyshevT(c.coords[1], size=nz, bounds=(0, Lz), dealias=dealias)
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
τ_s1 = d.Field(name='τ_s1', bases=xb)
τ_s2 = d.Field(name='τ_s2', bases=xb)
τ_u1 = d.VectorField(c, name='τ_u1', bases=(xb,))
τ_u2 = d.VectorField(c, name='τ_u2', bases=(xb,))

# Parameters and operators
div = lambda A: de.Divergence(A, index=0)
lap = lambda A: de.Laplacian(A, c)
grad = lambda A: de.Gradient(A, c)
#curl = lambda A: de.operators.Curl(A)
dot = lambda A, B: de.DotProduct(A, B)
cross = lambda A, B: de.CrossProduct(A, B)
trace = lambda A: de.Trace(A)
trans = lambda A: de.TransposeComponents(A)
dt = lambda A: de.TimeDerivative(A)

integ = lambda A: de.Integrate(de.Integrate(A, 'x'), 'z')
avg = lambda A: integ(A)/(Lx*Lz)
#x_avg = lambda A: de.Integrate(A, 'x')/(Lx)
x_avg = lambda A: de.Integrate(A, 'x')/(Lx)

from dedalus.core.operators import Skew
skew = lambda A: Skew(A)


ex = d.VectorField(c, name='ex')
ez = d.VectorField(c, name='ez')
ex['g'][0] = 1
ez['g'][1] = 1

# stress-free bcs
e = grad(u) + trans(grad(u))
e.store_last = True

viscous_terms = div(e) - 2/3*grad(div(u))
trace_e = trace(e)
trace_e.store_last = True
Phi = 0.5*trace(dot(e, e)) - 1/3*(trace_e*trace_e)


h0 = d.Field(name='h0', bases=zb)
h0['g'] = h_bot + z*h_slope #(Lz+1)-z

θ0 = np.log(h0).evaluate()
Υ0 = (m_ad*θ0).evaluate()
Υ0.name = 'Υ0'
ρ0 = np.exp(Υ0).evaluate()
ρ0.name = 'ρ0'
ρ0_inv = np.exp(-Υ0).evaluate()
ρ0_inv.name = '1/ρ0'
grad_h0 = grad(h0).evaluate()
grad_θ0 = grad(θ0).evaluate()
grad_Υ0 = grad(Υ0).evaluate()

h0_g = de.Grid(h0).evaluate()
h0_inv_g = de.Grid(1/h0).evaluate()
grad_h0_g = de.Grid(grad(h0)).evaluate()
ρ0_g = de.Grid(ρ0).evaluate()

ρ0_grad_h0_g = de.Grid(ρ0*grad(h0)).evaluate()
ρ0_h0_g = de.Grid(ρ0*h0).evaluate()

source = d.Field(name='source', bases=b)
source.change_scales(dealias)
source['g'] = (ε*R_inv/Pr/h0).evaluate()['g']
source_g = de.Grid(source).evaluate()

Υ_bot = Υ0(z=0).evaluate()['g']
Υ_top = Υ0(z=Lz).evaluate()['g']

θ_bot = θ0(z=0).evaluate()['g']
θ_top = θ0(z=Lz).evaluate()['g']

if rank ==0:
    logger.info("Δθ = {:.2g} ({:.2g} to {:.2g})".format(θ_bot[0][0]-θ_top[0][0],θ_bot[0][0],θ_top[0][0]))
    logger.info("ΔΥ = {:.2g} ({:.2g} to {:.2g})".format(Υ_bot[0][0]-Υ_top[0][0],Υ_bot[0][0],Υ_top[0][0]))

verbose = False
if verbose:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(nrows=2)

logger.info("NCC expansions:")
for ncc in [h0, ρ0, ρ0*grad(h0), ρ0*h0, ρ0*grad(θ0), h0*grad(Υ0)]:
    logger.info("{}: {}".format(ncc.evaluate(), np.where(np.abs(ncc.evaluate()['c']) >= ncc_cutoff)[0].shape))
    if verbose:
        ncc = ncc.evaluate()
        ncc.change_scales(1)
        if ncc['g'].ndim == 3:
            i = (1, 0, slice(None))
        else:
            i = (0, slice(None))
        ax[0].plot(z[0,:], ncc['g'][i])
        ax[1].plot(z[0,:], np.abs(ncc['g'][i]), label=ncc.name)
if verbose:
    ax[1].set_yscale('log')
    ax[1].legend()
    fig.savefig('structure.pdf')

# Υ = ln(ρ), θ = ln(h)
problem = de.IVP([u, Υ, θ, s, τ_u1, τ_u2, τ_s1, τ_s2])
problem.add_equation((ρ0*(dt(u) + 1/Ma2*(h0*grad(θ) + grad_h0*θ)
                      - 1/Ma2*s_c_over_c_P*h0*grad(s))
                      - R_inv*viscous_terms
                      + lift(τ_u1,-1) + lift(τ_u2,-2),
                      -ρ0_g*dot(u,grad(u))
                      -1/Ma2*ρ0_grad_h0_g*(np.expm1(θ)-θ)
                      -1/Ma2*ρ0_h0_g*np.expm1(θ)*grad(θ)
                      +1/Ma2*scrS*ρ0_h0_g*np.expm1(θ)*grad(s)
                      ))
problem.add_equation((h0*(dt(Υ) + div(u) + dot(u, grad_Υ0)), # + dot(lift(τ_u2,-1),ez),
                      -h0_g*dot(u, grad(Υ)) ))
problem.add_equation((θ - (γ-1)*Υ - s_c_over_c_P*γ*s, 0)) #EOS, s_c/cP = scrS
#TO-DO:
#consider adding back in diffusive & source nonlinearities
problem.add_equation((ρ0*s_c_over_c_P*dt(s)
                      - R_inv/Pr*(lap(θ)+2*dot(grad_θ0,grad(θ)))
                      + lift(τ_s1,-1) + lift(τ_s2,-2),
                      - ρ0_g*s_c_over_c_P*dot(u,grad(s))
                      + R_inv/Pr*dot(grad(θ),grad(θ))
                      + R_inv*Ma2*h0_inv_g*Phi  # + R_inv*Ma2*0.5*h0_inv_g*Phi
                      + source_g ))
# problem.add_equation((dot(ez,grad(θ))(z=0), 0))
# problem.add_equation((dot(ez,u)(z=0), 0))
problem.add_equation((u(z=0),0))
#problem.add_equation((dot(ez, dot(ex,e))(z=0), 0))
problem.add_equation((θ(z=0), 0))
problem.add_equation((θ(z=Lz), 0))
problem.add_equation((u(z=Lz),0))
# problem.add_equation((dot(ez,u)(z=Lz), 0))
# problem.add_equation((dot(ez, dot(ex,e))(z=Lz), 0))
logger.info("Problem built")

# initial conditions
amp = 1e-4

zb, zt = zb.bounds
noise = d.Field(name='noise', bases=b)
noise.fill_random('g', seed=42, distribution='normal', scale=amp) # Random noise
noise.low_pass_filter(scales=0.25)

s['g'] = noise['g']*np.cos(np.pi/2*z/Lz)
# pressure balanced ICs
Υ['g'] = -scrS*γ/(γ-1)*s['g']
θ['g'] = scrS*γ*s['g'] + (γ-1)*Υ['g'] # this should evaluate to zero

if args['--SBDF2']:
    ts = de.SBDF2
    cfl_safety_factor = 0.2
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

s0 = d.Field(name='s0')
s0['g'] = 0

ρ = ρ0*np.exp(Υ).evaluate()
h = h0*np.exp(θ).evaluate()
KE = 0.5*ρ*dot(u,u)
IE = 1/Ma2*ρ*h
PE = -1/Ma2*ρ*h*(s+s0)
Re = (ρ*R)*np.sqrt(dot(u,u))
ω = -div(skew(u))
KE.store_last = True
PE.store_last = True
IE.store_last = True
Re.store_last = True
ω.store_last = True

slice_dt = 0.1
trace_dt = 0.1

slice_output = solver.evaluator.add_file_handler(data_dir+'/slices',sim_dt=slice_dt,max_writes=20)
slice_output.add_task(s+s0, name='s+s0')
slice_output.add_task(s, name='s')
slice_output.add_task(θ, name='θ')
slice_output.add_task(ω, name='vorticity')
slice_output.add_task(ω**2, name='enstrophy')
# horizontal averages
slice_output.add_task(x_avg(s), name='s(z)')
slice_output.add_task(x_avg(-R_inv/Pr*dot(grad(h-h0),ez)), name='F_κ(z)')
slice_output.add_task(x_avg(0.5*ρ*dot(u,ez)*dot(u,u)), name='F_KE(z)')
slice_output.add_task(x_avg(dot(u,ez)*ρ*h), name='F_h(z)')
slice_output.add_task(x_avg(-dot(u,ez)*ρ*h*s), name='F_PE(z)')
#slice_output.add_task(x_avg(-h*source*z), name='F_source') # only valid for constant source
slice_output.add_task(x_avg(-h*source), name='Q_source(z)')

Ma_ad2 = Ma2*dot(u,u)*cP/(γ*h)
traces = solver.evaluator.add_file_handler(data_dir+'/traces', sim_dt=trace_dt, max_writes=np.inf)
traces.add_task(avg(0.5*ρ*dot(u,u)), name='KE')
traces.add_task(avg(PE), name='PE')
traces.add_task(avg(IE), name='IE')
traces.add_task(avg(Re), name='Re')
traces.add_task(avg(ω**2), name='enstrophy')
traces.add_task(np.sqrt(avg(Ma_ad2)), name='Ma_ad')
traces.add_task(x_avg(np.sqrt(dot(τ_u1,τ_u1))), name='τ_u1')
traces.add_task(x_avg(np.sqrt(dot(τ_u2,τ_u2))), name='τ_u2')
traces.add_task(x_avg(np.sqrt(τ_s1**2)), name='τ_s1')
traces.add_task(x_avg(np.sqrt(τ_s2**2)), name='τ_s2')

report_cadence = 10
good_solution = True

flow = flow_tools.GlobalFlowProperty(solver, cadence=report_cadence)
flow.add_property(Re, name='Re')
flow.add_property(KE, name='KE')
flow.add_property(IE, name='IE')
flow.add_property(np.sqrt(avg(Ma_ad2)), name='Ma_ad')
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
        Ma_ad_avg = flow.grid_average('Ma_ad')
        Re_avg = flow.grid_average('Re')
        Re_max = flow.max('Re')
        τu1_max = flow.max('τ_u1')
        τu2_max = flow.max('τ_u2')
        τs1_max = flow.max('τ_s1')
        τs2_max = flow.max('τ_s2')
        τ_max = np.max([τu1_max,τu2_max,τs1_max,τs2_max])
        log_string = 'Iteration: {:5d}, Time: {:8.3e}, dt: {:5.1e}'.format(solver.iteration, solver.sim_time, Δt)
        log_string += ', KE: {:.2g}, Ma: {:.2g}, IE: {:.2g}, Re: {:.2g} ({:.2g})'.format(KE_avg, Ma_ad_avg, IE_avg, Re_avg, Re_max)
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
