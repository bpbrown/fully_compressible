"""
Dedalus script for 2D compressible convection in a polytrope,
with specified number of density scale heights of stratification.

Returns critical R, k_c

Usage:
    FC_poly_Q_onset.py [options]

Options:
    --kx=<kx>                            Horizontal mode [default: 3]

    --Prandtl=<Prandtl>                  Prandtl number = nu/kappa [default: 1]
    --n_h=<n_h>                          Enthalpy scale heights [default: 0.5]
    --epsilon=<epsilon>                  The level of superadiabaticity of our polytrope background [default: 0.5]
    --m=<m>                              Polytopic index of our polytrope
    --gamma=<gamma>                      Gamma of ideal gas (cp/cv) [default: 5/3]

    --no_slip                            Use no-slip boundary conditions


    --nz=<nz>                            vertical z (chebyshev) resolution [default: 64]

    --ncc_cutoff=<ncc_cutoff>            Amplitude cutoff for NCCs [default: 1e-8]

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

#Resolution
nz = int(args['--nz'])
kx = float(args['--kx'])

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
c = de.CartesianCoordinates('x','z')
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

ex, ez = c.unit_vector_fields(d)

# Parameters and operators
dx = lambda A: 1j*kx*A
div = lambda A: de.Divergence(A, index=0) + dx(A@ex)
lap = lambda A: de.Laplacian(A, c) + dx(dx(A))
grad = lambda A: de.Gradient(A, c) + dx(A)*ex
trace = lambda A: de.Trace(A)
trans = lambda A: de.TransposeComponents(A)
dt = lambda A: de.TimeDerivative(A)

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

R_inv = d.Field(name='R_inv')
ρ0 = np.exp(Υ0).evaluate()
# Υ = ln(ρ), θ = ln(h)
problem = de.EVP([u, Υ, θ, s, τ_u1, τ_u2, τ_s1, τ_s2], eigenvalue=R_inv)
problem.add_equation((ρ0*(1/Ma2*(h0*grad(θ) + grad(h0)*θ)
                      - 1/Ma2*s_c_over_c_P*h0*(grad(s) + θ*grad(s0)) )
                      - R_inv*viscous_terms
                      + lift(τ_u1,-1) + lift(τ_u2,-2),
                      0 ))
problem.add_equation((h0*(div(u) + u@grad(Υ0)) + R_inv*lift(τ_u2,-1)@ez,
                      0 ))
problem.add_equation((θ - (γ-1)*Υ - s_c_over_c_P*γ*s, 0)) #EOS, s_c/cP = scrS
#TO-DO:
#consider adding back in diffusive & source nonlinearities
problem.add_equation((ρ0*u@grad(s0)
                      - R_inv/Pr*(lap(θ)+2*grad(θ0)@grad(θ))
                      + lift(τ_s1,-1) + lift(τ_s2,-2),
                      0 ))
if no_slip:
    problem.add_equation((u(z=0), 0))
    problem.add_equation((u(z=Lz), 0))
else:
    problem.add_equation((ez@u(z=0), 0))
    problem.add_equation((ez@(ex@e(z=0)), 0))
    problem.add_equation((ez@u(z=Lz), 0))
    problem.add_equation((ez@(ex@e(z=Lz)), 0))
problem.add_equation((ez@grad(θ)(z=0), 0))
problem.add_equation((θ(z=Lz), 0))
logger.info("Problem built")

solver = problem.build_solver()
solver.solve_dense(solver.subproblems[0])
i_evals = np.argsort(solver.eigenvalues.real)
evals = solver.eigenvalues[i_evals]
evals = evals[np.isfinite(evals)]
print(evals)
