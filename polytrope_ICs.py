"""
Computes hydrostatic equilbrium in polytropes for an enthalphy/entropy formulation.

Usage:
    polytrope_ICs.py [options]

Options:
    --n_rho=<n_rho>                      Density scale heights across unstable layer [default: 3]
    --epsilon=<epsilon>                  The level of superadiabaticity of our polytrope background [default: 0]
    --m=<m>                              Polytopic index m; optional (defaults to 1/(gamma-1)-epsilon if not specified)
    --gamma=<gamma>                      Gamma of ideal gas (cp/cv) [default: 5/3]

    --nh=nh                              Number of enthalpy scale heights; supersedes L [default: 2]
    --L=<L>                              Domain size in Hρ units at base of zone [default: 1]

    --top                                Normalize from the top of the domain
    --IC=<IC>                            Initial guess [default: isothermal]
    --nz=<nz>                            vertical z (chebyshev) resolution [default: 128]
    --iter=<iter>                        maximum iterations [default: 20]
"""

import numpy as np
import dedalus.public as de
from dedalus.core import arithmetic, problems, solvers
from mpi4py import MPI
import time
rank = MPI.COMM_WORLD.rank

import matplotlib.pyplot as plt

from docopt import docopt
args = docopt(__doc__)
from fractions import Fraction

from dedalus.tools.parsing import split_equation
from dedalus.extras.flow_tools import GlobalArrayReducer

import logging
logger = logging.getLogger(__name__)

#Resolution
nz = int(args['--nz'])

γ  = float(Fraction(args['--gamma']))

if args['--m']:
    m = float(args['--m'])
else:
    m = 1/(γ-1) - float(args['--epsilon'])
logger.info("m = {:}".format(m))

Lz = float(args['--L'])
norm_top = args['--top']
if args['--nh']:
    nh = float(args['--nh'])
    if norm_top:
        Lz = (m+1)*(np.exp(nh)-1)
    else:
        Lz = (m+1)*(1-np.exp(-nh))
    logger.info("to hit nh = {:.2g}, using Lz = {:.5g}".format(nh, Lz))
print("predicted enthalpy slope: {:}".format(-1/(1+m)))
if norm_top:
    h_bot = 1+Lz/(m+1)
else:
    h_bot = 1
h_top = h_bot - Lz/(m+1)
print("predicted enthalpy bot,top: {:.2g}, {:.2g}".format(h_bot,h_top))
print("predicted nh: {:.2g}".format(np.log(h_bot)-np.log(h_top)))
print("predicted nρ: {:.2g}".format(m*np.log(h_bot)-np.log(h_top)))
if h_top <= 0:
    logger.warning("h_top <= 0 (h_top = {:.2g})".format(h_top))

dealias = 2
c = de.coords.CartesianCoordinates('z')
d = de.distributor.Distributor((c,))
zb = de.basis.ChebyshevT(c.coords[0], size=nz, bounds=(0, Lz), dealias=dealias)
z = zb.local_grid(dealias)

# Parameters and operators
grad = lambda A: de.operators.Gradient(A, c)
lap = lambda A: de.operators.Laplacian(A, c)
dot = lambda A, B: arithmetic.DotProduct(A, B)
exp = lambda A: de.operators.UnaryGridFunction(np.exp, A)
ez = de.field.Field(name='ez', dist=d, bases=(zb,), dtype=np.float64, tensorsig=(c,))
ez['g'][0] = 1

Υ = de.field.Field(name='Υ', dist=d, bases=(zb,), dtype=np.float64)
θ = de.field.Field(name='θ', dist=d, bases=(zb,), dtype=np.float64)
S = de.field.Field(name='S', dist=d, bases=(zb,), dtype=np.float64)
# Taus
τθ = de.field.Field(name='τθ', dist=d, tensorsig=(c,), dtype=np.float64)
zb1 = de.basis.ChebyshevU(c.coords[0], size=nz, bounds=(0, Lz), alpha0=0, dealias=dealias)
P1 = de.field.Field(name='P1', dist=d, bases=(zb1,), dtype=np.float64)
if rank == 0:
    P1['c'][-1] = 1

grad_φ = (γ-1)/γ  # if L = (RT/g)(z=0) = Hρ(z=0) for an equivalent isothermal atmosphere, then grad_φ = 1/Cp

HS_problem = problems.NLBVP([Υ, θ, S, τθ])
HS_problem.add_equation((grad(θ) - grad(S) + τθ*P1 , -1*exp(-θ)*grad_φ*ez))
HS_problem.add_equation((Υ - m*θ, 0))
HS_problem.add_equation((S - (1/γ*θ - (γ-1)/γ*Υ), 0))
if norm_top:
    HS_problem.add_equation((θ(z=Lz),0))
else:
    HS_problem.add_equation((θ(z=0),0))

ncc_cutoff = 1e-14
tolerance = 1e-8

solver = solvers.NonlinearBoundaryValueSolver(HS_problem, ncc_cutoff=ncc_cutoff)

# Initial guess
for f in [Υ, θ, S]:
    f.require_scales(dealias)
IC = args['--IC']
if IC == 'isothermal':
    # isothermal ICs
    logger.info("using isothermal initial guess")
    θ['g'] = 0
elif IC == 'exact':
    # adiabatic polytrope ICs
    logger.info("using exact polytrope initial guess")
    θ['g'] = np.log(h_bot-1/(m+1)*z)
    Υ['g'] = m*θ['g']
    S['g'] = 1/γ*θ['g']-(γ-1)/γ*Υ['g']
else:
    # adiabatic polytrope ICs
    logger.info("using adiabatic near-polytrope initial guess")
    θ['g'] = np.log(h_bot-1/(m+1)*z)
    Υ['g'] = 1/(γ-1)*θ['g']
    S['g'] = 1/γ*θ['g']-(γ-1)/γ*Υ['g']

fig, ax = plt.subplots()

# Iterations
def error(perts):
    return np.sum([np.sum(np.abs(pert['c'])) for pert in perts])
err = np.inf
ax2 = ax.twinx()

θ.require_scales(dealias)
#ax.plot(z, θ['g'], color='black')
#ax.plot(z, Υ['g'], linestyle='dashed', color='black')
#ax.plot(z, S['g'], color='darkgrey')
#ax2.plot(z, exp(θ).evaluate()['g'], linestyle='dotted', color='black')
while err > tolerance and np.max(np.abs(τθ['g'])) < 1 and solver.iteration < int(float(args['--iter'])):
    solver.newton_iteration()
    err = error(solver.perturbations)
    logger.info("current error {:.3g}, and |τ| {:.3g}".format(err, np.max(np.abs(τθ['g']))))
    for f in [Υ, θ, S]:
        f.require_scales(dealias)
    ax.plot(z, θ['g'])
    ax.plot(z, Υ['g'], linestyle='dashed')
    ax.plot(z, S['g'], color='darkgrey')
    ax2.plot(z, exp(θ).evaluate()['g'], linestyle='dotted')
fig.savefig("hs_balance.png", dpi=300)
print("density  contrast nρ = {:.3g}".format(Υ['g'][0]-Υ['g'][-1]))
print("enthalpy contrast nh = {:.3g}".format(θ['g'][0]-θ['g'][-1]))
print("enthalpy slope = {:.3g}".format(grad(exp(θ)).evaluate()['g'][0][-1]))
print("departure from thermal eq: {:.3g}".format(np.max(np.abs(lap(θ)+dot(grad(θ), grad(θ))).evaluate()['g'])))
