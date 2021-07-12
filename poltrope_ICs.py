"""
Computes hydrostatic equilbrium in polytropes for an enthalphy/entropy formulation.

Usage:
    polytrope_ICs.py [options]

Options:
    --n_rho=<n_rho>                      Density scale heights across unstable layer [default: 3]
    --epsilon=<epsilon>                  The level of superadiabaticity of our polytrope background [default: 1e-4]
    --gamma=<gamma>                      Gamma of ideal gas (cp/cv) [default: 5/3]

    --nz=<nz>                            vertical z (chebyshev) resolution [default: 128]
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
nx = args['--nx']
if nx is not None:
    nx = int(nx)
else:
    nx = int(nz*float(args['--aspect']))

γ  = float(Fraction(args['--gamma']))

# wrong height
Lx, Lz = (float(args['--aspect']), 1)
print(nx, Lx)

dealias = 2
c = de.coords.CartesianCoordinates('z')
d = de.distributor.Distributor((c,))
zb = de.basis.ChebyshevT(c.coords[0], size=nz, bounds=(0, Lz), dealias=dealias)
z = zb.local_grid(dealias)

# Parameters and operators
grad = lambda A: de.operators.Gradient(A, c)
exp = lambda A: de.operators.UnaryGridFunction(np.exp, A)
dz = lambda A: de.operators.Differentiate(A, c.coords[0])
ez = de.field.Field(name='ez', dist=d, bases=(zb,), dtype=np.float64, tensorsig=(c,))
ez['g'][0] = 1

θ = de.field.Field(name='θ', dist=d, bases=(zb,), dtype=np.float64)
# Taus
τθ = de.field.Field(name='τθ', dist=d, tensorsig=(c,), dtype=np.float64)
#τθ = de.field.Field(name='τθ', dist=d, dtype=np.float64)
zb1 = de.basis.ChebyshevU(c.coords[0], size=nz, bounds=(0, Lz), alpha0=0, dealias=dealias)
P1 = de.field.Field(name='P1', dist=d, bases=(zb1,), dtype=np.float64)
if rank == 0:
    P1['c'][-1] = 1

grad_φ = 0.775
m = 1
HS_problem = problems.NLBVP([θ, τθ])
HS_problem.add_equation(((γ-1)/γ*(1+m)*grad(θ)+τθ*P1, -1*exp(-θ)*grad_φ*ez))
#HS_problem.add_equation(((γ-1)/γ*(1+m)*dz(θ)+τθ*P1, -1*exp(-θ)*grad_φ))
HS_problem.add_equation((θ(z=0),0))

print("Problem built")
ncc_cutoff = 1e-8
tolerance = 1e-8

solver = solvers.NonlinearBoundaryValueSolver(HS_problem, ncc_cutoff=ncc_cutoff)

# Initial guess
θ['g'] = 0

fig, ax = plt.subplots()

# Iterations
def error(perts):
    return np.sum([np.sum(np.abs(pert['c'])) for pert in perts])
err = np.inf
ax2 = ax.twinx()

θ.require_scales(dealias)
ax.plot(z, θ['g'], linestyle='dashed', color='black')
ax2.plot(z, m*θ['g'], linestyle='dotted', color='black')
while err > tolerance:
    solver.newton_iteration()
    err = error(solver.perturbations)
    logger.info("current error {:}".format(err))
    ax.plot(z, θ['g'])
    ax2.plot(z, m*θ['g'])
    fig.savefig("hs_balance.png", dpi=300)
print(m*θ['g'][-1])
