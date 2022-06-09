"""
Dedalus script for computing equilibrated background for heated
atmospheres (using adiabaitc polytropes to set initial guess),
with specified number of density scale heights of stratification.

Usage:
    structure.py [options]

Options:
    --n_h=<n_h>                          Enthalpy scale heights [default: 0.5]
    --epsilon=<epsilon>                  The level of superadiabaticity of our polytrope background [default: 0.5]
    --m=<m>                              Polytopic index of our polytrope
    --gamma=<gamma>                      Gamma of ideal gas (cp/cv) [default: 5/3]
    --nz=<nz>                            vertical z (chebyshev) resolution [default: 64]

    --ncc_cutoff=<ncc_cutoff>            Amplitude cutoff for NCCs [default: 1e-8]

    --verbose                            Show structure plots at end of solve
"""

import numpy as np
import logging
logger = logging.getLogger(__name__)
dlog = logging.getLogger('evaluator')
dlog.setLevel(logging.WARNING)
dlog = logging.getLogger('matplotlib')
dlog.setLevel(logging.WARNING)

def heated_adiabatic_polytrope(nz, γ, ε, n_h,
                     tolerance = 1e-8,
                     ncc_cutoff = 1e-10,
                     dealias = 2,
                     verbose=False):

    import dedalus.public as de

    cP = γ/(γ-1)
    m_ad = 1/(γ-1)

    s_c_over_c_P = scrS = 1 # s_c/c_P = 1

    logger.info("γ = {:.3g}, ε={:.3g}".format(γ, ε))

    # this assumes h_bot=1, grad_φ = (γ-1)/γ (or L=Hρ)
    h_bot = 1
    # generally, h_slope = -1/(1+m)
    # start in an adibatic state, heat from there
    h_slope = -1/(1+m_ad)
    grad_φ = (γ-1)/γ

    Lz = -1/h_slope*(1-np.exp(-n_h))

    print(n_h, Lz, h_slope)

    c = de.CartesianCoordinates('z')
    d = de.Distributor(c, dtype=np.float64)
    zb = de.ChebyshevT(c.coords[-1], size=nz, bounds=(0, Lz), dealias=dealias)
    b = zb
    z = zb.local_grid(1)
    zd = zb.local_grid(dealias)

    # Fields
    θ = d.Field(name='θ', bases=b)
    Υ = d.Field(name='Υ', bases=b)
    s = d.Field(name='s', bases=b)

    # Taus
    lift_basis = zb.clone_with(a=zb.a+2, b=zb.b+2)
    lift = lambda A, n: de.Lift(A, lift_basis, n)
    lift_basis1 = zb.clone_with(a=zb.a+1, b=zb.b+1)
    lift1 = lambda A, n: de.Lift(A, lift_basis1, n)
    τ_h1 = d.VectorField(c,name='τ_h1')
    #τ_h1 = d.Field(name='τ_h1')
    τ_s1 = d.Field(name='τ_s1')
    τ_s2 = d.Field(name='τ_s2')

    # Parameters and operators
    lap = lambda A: de.Laplacian(A, c)
    grad = lambda A: de.Gradient(A, c)
    dz = lambda A: de.Differentiate(A, c)
    ez = d.VectorField(c,bases=zb)
    ez['g'][-1]=1
    # NLBVP goes here
    # intial guess
    h0 = d.Field(name='h0', bases=zb)
    θ0 = d.Field(name='θ0', bases=zb)
    Υ0 = d.Field(name='Υ0', bases=zb)
    s0 = d.Field(name='s0', bases=zb)
    structure = {'h':h0,'s':s0,'θ':θ0,'Υ':Υ0}
    for key in structure:
        structure[key].change_scales(dealias)
    h0['g'] = h_bot + zd*h_slope #(Lz+1)-z
    θ0['g'] = np.log(h0).evaluate()['g']
    Υ0['g'] = (m_ad*θ0).evaluate()['g']
    s0['g'] = 0

    problem = de.NLBVP([h0, s0, Υ0, τ_s1, τ_s2, τ_h1])
    problem.add_equation((grad(h0) + lift1(τ_h1,-1),
                         -grad_φ*ez + h0*grad(s0)))
    problem.add_equation((-lap(h0)
    + lift(τ_s1,-1) + lift(τ_s2,-2), ε))
    problem.add_equation(((γ-1)*Υ0 + s_c_over_c_P*γ*s0, np.log(h0)))
    # go to integral density (mass) boundary condition --Geoff
    problem.add_equation(((ez@grad(h0))(z=0), -1/(1+m_ad)))
    problem.add_equation((Υ0(z=0), 0))
    problem.add_equation((Υ0(z=Lz), -m_ad*n_h))
    # Solver
    solver = problem.build_solver(ncc_cutoff=ncc_cutoff)
    pert_norm = np.inf
    while pert_norm > tolerance:
        solver.newton_iteration()
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
        logger.info('current perturbation norm = {:.3g}'.format(pert_norm))

    if verbose:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax2 = ax.twinx()

        ax.plot(zd, h0['g'], linestyle='dashed', color='xkcd:dark grey', label='h')
        ax2.plot(zd, np.log(h0).evaluate()['g'], label=r'$\ln h$')
        ax2.plot(zd, Υ0['g'], label=r'$\ln \rho$')
        ax2.plot(zd, s0['g'], color='xkcd:brick red', label=r'$s$')
        ax.legend()
        ax2.legend()
        fig.savefig('heated_adiabatic_polytrope_nh{}_eps{:.3g}_gamma{:.3g}.pdf'.format(n_h,ε,γ))

    print(h0(z=0).evaluate()['g'])

    for key in structure:
        structure[key].change_scales(1)

    return structure

def heated_polytrope(nz, γ, ε, n_h,
                     tolerance = 1e-8,
                     ncc_cutoff = 1e-10,
                     dealias = 2,
                     verbose=False):

    import dedalus.public as de

    cP = γ/(γ-1)
    m_ad = 1/(γ-1)

    s_c_over_c_P = scrS = 1 # s_c/c_P = 1

    logger.info("γ = {:.3g}, ε={:.3g}".format(γ, ε))

    # this assumes h_bot=1, grad_φ = (γ-1)/γ (or L=Hρ)
    h_bot = 1
    # generally, h_slope = -1/(1+m)
    # start in an adibatic state, heat from there
    h_slope = -1/(1+m_ad)
    grad_φ = (γ-1)/γ

    Lz = -1/h_slope*(1-np.exp(-n_h))

    print(n_h, Lz, h_slope)

    c = de.CartesianCoordinates('z')
    d = de.Distributor(c, dtype=np.float64)
    zb = de.ChebyshevT(c.coords[-1], size=nz, bounds=(0, Lz), dealias=dealias)
    b = zb
    z = zb.local_grid(1)
    zd = zb.local_grid(dealias)

    # Fields
    θ = d.Field(name='θ', bases=b)
    Υ = d.Field(name='Υ', bases=b)
    s = d.Field(name='s', bases=b)

    # Taus
    lift_basis = zb.clone_with(a=zb.a+2, b=zb.b+2)
    lift = lambda A, n: de.Lift(A, lift_basis, n)
    lift_basis1 = zb.clone_with(a=zb.a+1, b=zb.b+1)
    lift1 = lambda A, n: de.Lift(A, lift_basis1, n)
    τ_h1 = d.VectorField(c,name='τ_h1')
    #τ_h1 = d.Field(name='τ_h1')
    τ_s1 = d.Field(name='τ_s1')
    τ_s2 = d.Field(name='τ_s2')

    # Parameters and operators
    lap = lambda A: de.Laplacian(A, c)
    grad = lambda A: de.Gradient(A, c)
    integ = lambda A: de.Integrate(A, 'z')
    ez, = c.unit_vector_fields(d)

    # NLBVP goes here
    # intial guess
    h0 = d.Field(name='h0', bases=zb)
    θ0 = d.Field(name='θ0', bases=zb)
    Υ0 = d.Field(name='Υ0', bases=zb)
    s0 = d.Field(name='s0', bases=zb)
    structure = {'h':h0,'s':s0,'θ':θ0,'Υ':Υ0}
    for key in structure:
        structure[key].change_scales(dealias)
    h0['g'] = h_bot + zd*h_slope #(Lz+1)-z
    θ0['g'] = np.log(h0).evaluate()['g']
    Υ0['g'] = (m_ad*θ0).evaluate()['g']
    s0['g'] = 0

    problem = de.NLBVP([h0, s0, Υ0, τ_s1, τ_s2, τ_h1])
    problem.add_equation((grad(h0) + lift1(τ_h1,-1),
                         -grad_φ*ez + h0*grad(s0)))
    problem.add_equation((-lap(h0) + lift(τ_s1,-1) + lift(τ_s2,-2), ε))
    problem.add_equation(((γ-1)*Υ0 + s_c_over_c_P*γ*s0, np.log(h0)))
    problem.add_equation((h0(z=0), 1))
    problem.add_equation((h0(z=Lz), np.exp(-n_h)))
    # integral density (mass) boundary condition --Geoff
    problem.add_equation((integ(Υ0), 1))
    # Solver
    solver = problem.build_solver(ncc_cutoff=ncc_cutoff)
    pert_norm = np.inf
    while pert_norm > tolerance:
        solver.newton_iteration()
        pert_norm = sum(pert.allreduce_data_norm('c', 2) for pert in solver.perturbations)
        logger.info('current perturbation norm = {:.3g}'.format(pert_norm))

    # re-normalize density and entropy (Υ0(z=0)=0, s(z=0)=0)
    Υ0 = (Υ0-Υ0(z=0)).evaluate()
    Υ0.name='Υ0'
    structure['Υ'] = Υ0
    s0 = (s0-s0(z=0)).evaluate()
    s0.name = 's0'
    structure['s'] = s0

    if verbose:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax2 = ax.twinx()

        ax.plot(zd, h0['g'], linestyle='dashed', color='xkcd:dark grey', label='h')
        ax2.plot(zd, np.log(h0).evaluate()['g'], label=r'$\ln h$')
        ax2.plot(zd, Υ0['g'], label=r'$\ln \rho$')
        ax2.plot(zd, s0['g'], color='xkcd:brick red', label=r'$s$')
        ax.legend()
        ax2.legend()
        fig.savefig('heated_polytrope_nh{}_eps{:.3g}_gamma{:.3g}.pdf'.format(n_h,ε,γ))

    for key in structure:
        structure[key].change_scales(1)

    return structure

if __name__=='__main__':
    from docopt import docopt
    args = docopt(__doc__)
    from fractions import Fraction

    ncc_cutoff = float(args['--ncc_cutoff'])

    #Resolution
    nz = int(args['--nz'])

    γ  = float(Fraction(args['--gamma']))
    m_ad = 1/(γ-1)

    if args['--m']:
        m = float(args['--m'])
        strat_label = 'm{}'.format(args['--m'])
    else:
        m = m_ad - float(args['--epsilon'])
        strat_label = 'eps{}'.format(args['--epsilon'])
    ε = m_ad - m

    n_h = float(args['--n_h'])

    verbose = args['--verbose']

    structure = heated_polytrope(nz, γ, ε, n_h, verbose=verbose)
    for key in structure:
        print(structure[key], structure[key]['g'])
