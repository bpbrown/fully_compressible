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

    --nz=<nz>                            vertical z (chebyshev) resolution [default: 64]

    --R_min=<R_min>         Min value of scrR [default: 0.1]
    --R_max=<R_max>         Max value of scrR [default: 1]
    --R_num=<R_num>         How many values of scrR to search [default: 11]

    --single_solve          If set, solve at single mu value.  Otherwise, find the critical value.
    --skip_optimization     If set, don't do the optimization loop


    --tol=<tol>             Tolerance for opitimization loop [default: 1e-5]
    --eigs=<eigs>           Target number of eigenvalues to search for [default: 20]

    --ncc_cutoff=<ncc_cutoff>            Amplitude cutoff for NCCs [default: 1e-8]

    --whole_sun

    --no_slip

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

Pr = Prandtl = float(args['--Prandtl'])
γ  = float(Fraction(args['--gamma']))

no_slip = args['--no_slip']

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
data_dir += "_{}".format(strat_label)
data_dir += "_nz{:d}".format(nz)
if args['--whole_sun']:
    data_dir += '_wholesun'
if no_slip:
    data_dir += '_no-slip'
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

dealias = 2
c = de.CartesianCoordinates('x', 'y', 'z')
d = de.Distributor(c, dtype=np.complex128)
zb = de.ChebyshevT(c.coords[2], size=nz, bounds=(0, Lz), dealias=dealias)
b = (zb)
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
τ_s1 = d.Field(name='τs1')
τ_s2 = d.Field(name='τs2')
τ_u1 = d.VectorField(c, name='τu1')
τ_u2 = d.VectorField(c, name='τu2')

# Parameters and operators
grad0 = lambda A: de.Gradient(A, c)

kx = d.Field(name='kx')
dx = lambda A: 1j*kx*A
div = lambda A: de.Divergence(A, index=0) + dx(A@ex)
lap = lambda A: de.Laplacian(A, c) + dx(dx(A))
grad = lambda A: de.Gradient(A, c) + dx(A)*ex
curl = lambda A: de.Curl(A) - dx(A@ez)*ey + dx(A@ey)*ez
trace = lambda A: de.Trace(A)
trans = lambda A: de.TransposeComponents(A)

integ = lambda A: de.Integrate(de.Integrate(A, 'x'), 'z')
avg = lambda A: integ(A)/(Lx*Lz)
x_avg = lambda A: de.Integrate(A, 'x')/(Lx)

ex, ey, ez = c.unit_vector_fields(d)

h0 = d.Field(name='h0', bases=zb)
h0['g'] = h_bot+h_slope*z

θ0 = np.log(h0).evaluate()
Υ0 = (m*(θ0)).evaluate() # normalize to zero at bottom
s0 = (1/γ*θ0 - (γ-1)/γ*Υ0).evaluate()
ρ0 = np.exp(Υ0).evaluate()
grad_s0 = grad0(s0).evaluate()
grad_θ0 = grad0(θ0).evaluate()
grad_h0 = grad0(h0).evaluate()
grad_Υ0 = grad0(Υ0).evaluate()

h0_g = de.Grid(h0).evaluate()
h0_grad_s0_g = de.Grid(h0*grad0(s0)).evaluate()

ρ0_g = de.Grid(ρ0).evaluate()
ρ0_h0_g = de.Grid(ρ0*h0).evaluate()
ρ0_grad_h0_g = de.Grid(ρ0*grad0(h0)).evaluate()
ρ0_h0_grad_s0_g = de.Grid(ρ0*h0*grad0(s0)).evaluate()

# stress-free bcs
e = grad(u) + trans(grad(u))

viscous_terms = div(e) - 2/3*grad(div(u))
trace_e = trace(e)
Phi = 0.5*trace(e@e) - 1/3*(trace_e*trace_e)

Pr = 1

scrR = d.Field(name='scrR')
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

omega = d.Field(name='omega')
ddt = lambda A: omega*A


# Υ = ln(ρ), θ = ln(h)
problem = de.EVP([Υ, u, s, θ, τ_u1, τ_u2, τ_s1, τ_s2], eigenvalue=omega)
problem.add_equation((ρ0*ddt(u)
                      + ρ0*grad_h0*θ
                      + ρ0*h0*grad(θ)
                      - ρ0*h0*grad(s)
                      - ρ0*h0*grad_s0*θ
                      - scrR*viscous_terms
                      + lift(τ_u1,-1) + lift(τ_u2,-2),
                      0 ))
problem.add_equation((h0*(ddt(Υ) + div(u) + u@grad_Υ0) + 1/scrR*lift(τ_u2,-1)@ez,
                      0 ))
problem.add_equation((h0*ρ0*(ddt(s) + u@grad_s0)
                      - h0*scrP*(lap(θ) + 2*grad_θ0@grad(θ))
                      + lift(τ_s1,-1) + lift(τ_s2,-2),
                      0 ))
problem.add_equation((θ - (γ-1)*Υ - γ*s, 0)) #EOS, cP absorbed into s.
# boundary conditions
problem.add_equation((θ(z=0), 0))
if no_slip:
    problem.add_equation((u(z=0), 0))
else:
    problem.add_equation((ez@u(z=0), 0))
    problem.add_equation((ez@(ex@e(z=0)), 0))
    problem.add_equation((ez@(ey@e(z=0)), 0))
problem.add_equation((θ(z=Lz), 0))
if no_slip:
    problem.add_equation((u(z=Lz), 0))
else:
    problem.add_equation((ez@u(z=Lz), 0))
    problem.add_equation((ez@(ex@e(z=Lz)), 0))
    problem.add_equation((ez@(ey@e(z=Lz)), 0))
logger.info("Problem built")


solver = problem.build_solver(ncc_cutoff=ncc_cutoff)

N_eigs = int(float(args['--eigs']))


def compute_eigenvalues(kx_i, scrR_i, target):
    if not np.isscalar(kx_i):
        kx_i = kx_i[0]
    scrR['g'] = scrR_i
    kx['g'] = kx_i
    logger.info('kx = {:.3g}, R = {:.6g}'.format(kx_i, scrR_i))
    solver.solve_sparse(solver.subproblems[0], N=N_eigs, target=target, rebuild_matrices=True)
    i_evals = np.argsort(solver.eigenvalues.real)
    evals = solver.eigenvalues[i_evals]
    #evals /= np.sqrt(current_Ra)
    return(evals)

def peak_growth_rate(*args):
    evals = compute_eigenvalues(*args)
    peak_eval = evals[-1]
    # flip sign so minimize finds maximum
    return -1*peak_eval.real

for system  in ['matplotlib', 'subsystems']:
    dlog = logging.getLogger(system)
    dlog.setLevel(logging.WARNING)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()

kxs = np.geomspace(1,10, num=21)

target = 0 + 1j*0
scrR_i = float(args['--mu'])
if args['--single_solve']:
    peak_evals = []
    for kx_i in kxs:
        evals = compute_eigenvalues(kx_i, scrR_i, target)
        ax.scatter(evals.real, evals.imag, alpha=0.3)
        peak_evals.append(evals[-1])
    ax.axhline(y=0, linestyle='dashed', color='xkcd:dark grey', alpha=0.5, zorder=0)
    ax.axvline(x=0, linestyle='dashed', color='xkcd:dark grey', alpha=0.5, zorder=0)
    ax.set_xlabel(r'$\omega_R$')
    ax.set_ylabel(r'$\omega_I$')
    ax.scatter(target.real, target.imag, marker='x', label='target',  color='xkcd:dark green', alpha=0.2, zorder=1)
    ax.legend()
    fig_filename = 'eigenspectrum'
    fig.savefig(data_dir+'/'+fig_filename+'.png', dpi=300)

    peak_evals = np.array(peak_evals)
    i_max = np.argmax(peak_evals.real)
    print('peak growing mode: ω_r = {:.3g}, ω_i = {:.3g}, kx_i = {:.1e}'.format(peak_evals[i_max].real, peak_evals[i_max].imag, kxs[i_max]))
    fig, ax = plt.subplots()
    ax.scatter(kxs, peak_evals.real, alpha=0.5, label=r'$\omega_R$')
    ax.scatter(kxs, peak_evals.imag, alpha=0.5, label=r'$\omega_I$')
    ax.legend()
    ax.set_xscale('log')
    fig_filename = 'peak_omega_kx'
    fig.savefig(data_dir+'/'+fig_filename+'.png', dpi=300)
else:
    import scipy.optimize as sciop
    bounds = sciop.Bounds(lb=np.min(kxs), ub=np.max(kxs))
    peaks = {'σ':[], 'k':[], 'scrR':[]}
    discrete_peaks = {'σ':[], 'k':[], 'scrR':[]}

    fig, ax = plt.subplots()

    scrR_set = np.geomspace(float(args['--R_max']), float(args['--R_min']), num=int(float(args['--R_num'])))
    for scrR_i in scrR_set:
        target = 0 + 1j*0
        growth_rates = []
        for kx_i in kxs:
            evals = compute_eigenvalues(kx_i, scrR_i, target=target)
            growth_rates.append(evals[-1])
        growth_rates = np.array(growth_rates)
        i_peak = np.argmax(growth_rates)
        kx_i = kxs[i_peak]
        target = growth_rates[i_peak]
        discrete_peaks['σ'].append(target)
        discrete_peaks['k'].append(kx_i)
        discrete_peaks['scrR'].append(scrR_i)
        p = ax.scatter(kxs, growth_rates.real, alpha=0.5, label='{:.2g}'.format(scrR_i))

        optimize = not args['--skip_optimization']
        if optimize:
            target = 0 + 1j*0
            result = sciop.minimize(peak_growth_rate, kx_i, args=(scrR_i, target), method='Nelder-Mead', bounds=bounds, tol=1e-3)

            # obtain full complex rate
            peak_evals = compute_eigenvalues(kx_i, scrR_i, target)
            σ = peak_evals[-1]
            ax.scatter(kx_i, σ.real, c='xkcd:dark grey', marker='x', alpha=0.5)

            logger.info('peak search: start at scrR = {:.4g}, kx = {:.4g}, found σ_max = {:.2g},{:.2g}i, kx = {:.4g}'.format(scrR_i, kx_i, σ.real, σ.imag, result.x[0]))
            peaks['σ'].append(σ)
            peaks['k'].append(result.x[0])
            peaks['scrR'].append(scrR_i)

    if not optimize:
        peaks=discrete_peaks
    for key in peaks:
        peaks[key] = np.array(peaks[key])

    ax.plot(peaks['k'], peaks['σ'].real, linestyle='dotted', color='xkcd:dark grey')
    ax.axhline(y=0, color='xkcd:dark grey', linestyle='dashed')
    from scipy.interpolate import interp1d
    f_σR_scrR = interp1d(peaks['σ'].real, peaks['scrR'])
    f_σ_scrR = interp1d(peaks['scrR'], peaks['σ'])
    f_k_scrR = interp1d(peaks['scrR'], peaks['k'])

    print(peaks)
    print(discrete_peaks)
    crit_scrR = f_σR_scrR(0)
    crit_k = f_k_scrR(crit_scrR)
    crit_σ = f_σ_scrR(crit_scrR)

    logger.info('Critical point, based on interpolation:')
    logger.info('scrR = {:}, k = {:}'.format(crit_scrR, crit_k))
    logger.info('σ = {:}, {:}i'.format(crit_σ.real, crit_σ.imag))

    ax.scatter(crit_k, crit_σ.real, marker='s', label=r'$\mathcal{R}$'+': {:.3g}'.format(crit_scrR))

    ax.legend()
    ax.set_xscale('log')
    fig_filename = 'critical_curve'
    fig.savefig(data_dir+'/'+fig_filename+'.png', dpi=300)
