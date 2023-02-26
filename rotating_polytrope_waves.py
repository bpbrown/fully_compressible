"""
Dedalus script for properties of ideal 2D compressible waves in a rotating polytrope, with specified number of density scale heights of stratification.

Designed for testing of Hindman & Jain 2023

Returns critical R, k_c

Usage:
    rotating_polytrope_waves.py [options]

Options:
    --Prandtl=<Prandtl>                  Prandtl number = nu/kappa [default: 1]
    --n_h=<n_h>                          Enthalpy scale heights [default: 0.5]
    --epsilon=<epsilon>                  The level of superadiabaticity of our polytrope background [default: 0.5]
    --m=<m>                              Polytopic index of our polytrope
    --gamma=<gamma>                      Gamma of ideal gas (cp/cv) [default: 5/3]

    --no_slip                            Use no-slip boundary conditions

    --nz=<nz>                            vertical z (chebyshev) resolution [default: 64]

    --ncc_cutoff=<ncc_cutoff>            Amplitude cutoff for NCCs [default: 1e-8]

    --nkx=<nkx>       Number of kxs to solve at [default: 10]
    --target=<targ>   Target value for sparse eigenvalue search [default: 0]
    --eigs=<eigs>     Target number of eigenvalues to search for [default: 20]

    --dense           Solve densely for all eigenvalues (slow)

    --label=<label>                      Additional label for run output directory
"""

from mpi4py import MPI
import numpy as np
import sys
import os

from docopt import docopt
args = docopt(__doc__)
from fractions import Fraction

from structure import heated_polytrope
ncc_cutoff = float(args['--ncc_cutoff'])

# sparse eigenvalue search parameters
N_evals = int(float(args['--eigs']))
target = float(args['--target'])
nkx = int(float(args['--nkx']))

#Resolution
nz = int(args['--nz'])

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
#scrM = 1/Ma2
s_c_over_c_P = scrS = 1 # s_c/c_P = 1

no_slip = args['--no_slip']

import logging
logger = logging.getLogger(__name__)
dlog = logging.getLogger('evaluator')
dlog.setLevel(logging.WARNING)
dlog = logging.getLogger('matplotlib')
dlog.setLevel(logging.WARNING)

import dedalus.public as de
from dedalus.extras import flow_tools

logger.info("Ma2 = {:.3g}, Pr = {:.3g}, γ = {:.3g}, ε={:.3g}".format(Ma2, Pr, γ, ε))

# this assumes h_bot=1, grad_φ = (γ-1)/γ (or L=Hρ)
h_bot = 1
# generally, h_slope = -1/(1+m)
# start in an adibatic state, heat from there
h_slope = -1/(1+m_ad)
grad_φ = (γ-1)/γ

n_h = float(args['--n_h'])
Lz = -1/h_slope*(1-np.exp(-n_h))
print(n_h, Lz, h_slope)

dealias = 2
c = de.CartesianCoordinates('x', 'y', 'z')
d = de.Distributor(c, dtype=np.complex128)
zb = de.ChebyshevT(c.coords[-1], size=nz, bounds=(0, Lz), dealias=dealias)
b = zb
z = zb.local_grid(1)
zd = zb.local_grid(dealias)

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
τ_h1 = d.VectorField(c, name='τ_h1')
τ_s1 = d.Field(name='τ_s1')
τ_s2 = d.Field(name='τ_s2')
τ_u1 = d.VectorField(c, name='τ_u1')
τ_u2 = d.VectorField(c, name='τ_u2')

ex, ey, ez = c.unit_vector_fields(d)

kx = d.Field(name='kx')

# Parameters and operators
dx = lambda A: 1j*kx*A
div = lambda A: de.Divergence(A, index=0) + dx(A@ex)
lap = lambda A: de.Laplacian(A, c) + dx(dx(A))
grad = lambda A: de.Gradient(A, c) + dx(A)*ex
grad0 = lambda A: de.Gradient(A, c)
trace = lambda A: de.Trace(A)
trans = lambda A: de.TransposeComponents(A)
cross = lambda A, B: de.CrossProduct(A, B)

# stress-free bcs
e = grad(u) + trans(grad(u))

viscous_terms = div(e) - 2/3*grad(div(u))
trace_e = trace(e)

structure = heated_polytrope(nz, γ, ε, n_h)
h0 = d.Field(name='h0', bases=zb)
θ0 = d.Field(name='θ0', bases=zb)
Υ0 = d.Field(name='Υ0', bases=zb)
s0 = d.Field(name='s0', bases=zb)
h0['g'] = structure['h']['g']
θ0['g'] = structure['θ']['g']
Υ0['g'] = structure['Υ']['g']
s0['g'] = structure['s']['g']
logger.info(structure)

Ω = (1e-1*ey).evaluate()
omega = d.Field(name='omega')
ρ0 = np.exp(Υ0).evaluate()

logger.info("NCC expansions:")
grad_h0 = grad0(h0).evaluate()
grad_s0 = grad0(s0).evaluate()
grad_Υ0 = grad0(Υ0).evaluate()
for ncc in [ρ0, ρ0*h0, ρ0*grad_h0, h0*grad_Υ0, ρ0*grad_s0]:
    logger.info("{}: {}".format(ncc.evaluate(), np.where(np.abs(ncc.evaluate()['c']) >= ncc_cutoff)[0].shape))
logger.info("density scaleheights: {:.2g}".format((Υ0(z=0)-Υ0(z=Lz)).evaluate()['g'][0,0,0]))


# Υ = ln(ρ), θ = ln(h)
dt = lambda A: omega*A
problem = de.EVP([u, Υ, θ, s, τ_u1, τ_u2], eigenvalue=omega)
problem.add_equation((ρ0*dt(u) + ρ0*(1/Ma2*(h0*grad(θ) + grad_h0*θ)
                      - 1/Ma2*scrS*h0*(grad(s) + θ*grad_s0) )
                      + ρ0*2*cross(Ω, u)
                      + lift(τ_u1,-1) + lift(τ_u2,-2),
                      0 ))
problem.add_equation((h0*(div(u) + u@grad_Υ0) + lift(τ_u2,-1)@ez,
                      0 ))
problem.add_equation((θ - (γ-1)*Υ - scrS*γ*s, 0)) #EOS, s_c/cP = scrS
#TO-DO:
#consider adding back in diffusive & source nonlinearities
problem.add_equation((ρ0*dt(s) + ρ0*u@grad(s0),
                      0 ))
problem.add_equation((ez@u(z=0), 0))
problem.add_equation((ez@u(z=Lz), 0))
problem.add_equation((ey@u(z=0), 0))
problem.add_equation((ey@u(z=Lz), 0))
#problem.add_equation((ez@grad(θ)(z=0), 0))
problem.add_equation((θ(z=0), 0))
problem.add_equation((θ(z=Lz), 0))
logger.info("Problem built")

solver = problem.build_solver()

def compute_spectrum(kx_i):
    kx['g'] = kx_i
    if args['--dense']:
        solver.solve_dense(solver.subproblems[0], rebuild_matrices=True)
        # i_finite = np.isfinite(solver.eigenvalues)
        # solver.eigenvalues = solver.eigenvalues[i_finite]
    else:
        solver.solve_sparse(solver.subproblems[0], N=N_evals, target=target, rebuild_matrices=True)
        i_evals = np.argsort(solver.eigenvalues.real)
        evals = solver.eigenvalues[i_evals]
        evals = evals[np.isfinite(evals)]
    return evals

kxs = np.geomspace(1, 1e2, num=nkx)

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
fig_dis, ax_dis = plt.subplots(nrows=2, figsize=[8,4], sharex=True)
for kx_i in kxs:
    evals = compute_spectrum(kx_i)
    ax.scatter(evals.real, evals.imag, alpha=0.5, zorder=5)
    kx_i_g = kx_i*np.ones_like(evals.real)
    ax_dis[0].scatter(kx_i_g, evals.real, alpha=0.5, zorder=5)
    ax_dis[1].scatter(kx_i_g, evals.imag, alpha=0.5, zorder=5)

ax.axhline(y=0, linestyle='dashed', color='xkcd:dark grey', alpha=0.5, zorder=0)
ax.axvline(x=0, linestyle='dashed', color='xkcd:dark grey', alpha=0.5, zorder=0)
ax.set_title('k = {:}, '.format(kx)+r'$N_z = '+'{:d}'.format(nz)+r'$')
ax.set_xlabel(r'$\omega_R$')
ax.set_ylabel(r'$\omega_I$')
#ax.scatter(target.real, target.imag, marker='x', label='target',  color='xkcd:dark green', alpha=0.2, zorder=1)
ax.legend()
fig_filename = 'eigenspectrum_nz{:d}'.format(nz)
fig.savefig(fig_filename+'.png', dpi=300)

ax_dis[0].axhline(y=0, linestyle='dashed', color='xkcd:dark grey', alpha=0.5, zorder=0)
ax_dis[0].axvline(x=0, linestyle='dashed', color='xkcd:dark grey', alpha=0.5, zorder=0)
ax_dis[1].axhline(y=0, linestyle='dashed', color='xkcd:dark grey', alpha=0.5, zorder=0)
ax_dis[1].axvline(x=0, linestyle='dashed', color='xkcd:dark grey', alpha=0.5, zorder=0)
ax_dis[1].set_xlabel(r'$k_x$')
ylim = ax_dis[0].get_ylim()
ax_dis[0].set_ylim(0, ylim[-1])
ax_dis[0].set_ylabel(r'$\omega_R$')
ax_dis[1].set_ylabel(r'$\omega_I$')
fig_filename = 'dispersion_nz{:d}'.format(nz)
fig_dis.savefig(fig_filename+'.png', dpi=300)
