"""
Plot scalar outputs from scalar_output.h5 file.

Usage:
    plot_scalar.py <file> [options]

Options:
    --times=<times>      Range of times to plot over; pass as a comma separated list with t_min,t_max.  Default is whole timespan.
    --output=<output>    Output directory; if blank, a guess based on <file> location will be made.
"""
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pathlib
import h5py

import logging
logger = logging.getLogger(__name__.split('.')[-1])

from docopt import docopt
args = docopt(__doc__)
file = args['<file>']

if args['--output'] is not None:
    output_path = pathlib.Path(args['--output']).absolute()
else:
    data_dir = args['<file>'].split('/')[0]
    data_dir += '/'
    output_path = pathlib.Path(data_dir).absolute()

f = h5py.File(file, 'r')
data = {}
data_slice = (slice(None),0,0,0)
t = f['scales/sim_time'][:]
for key in f['tasks']:
    data[key] = f['tasks/'+key][data_slice]
f.close()

if args['--times']:
    subrange = True
    t_min, t_max = args['--times'].split(',')
    t_min = float(t_min)
    t_max = float(t_max)
    print("plotting over range {:g}--{:g}, data range {:g}--{:g}".format(t_min, t_max, min(t), max(t)))
else:
    subrange = False

energy_keys = ['KE','IE']#,'PE']

fig_E, ax_E = plt.subplots(nrows=2)
for key in energy_keys:
    ax_E[0].plot(t, data[key], label=key)
ax_E[1].plot(t, data['KE'], label='KE')
ax2 = ax_E[1].twinx()
ax2.plot(t, data['Re'], label='Re', linestyle='dotted')
ax2.legend(loc='upper left')

for ax in ax_E:
    if subrange:
        ax.set_xlim(t_min,t_max)
    ax.set_xlabel('time')
    ax.set_ylabel('energy density')
    ax.legend(loc='lower left')
fig_E.savefig('{:s}/energies.pdf'.format(str(output_path)))
for ax in ax_E:
    ax.set_yscale('log')
fig_E.savefig('{:s}/log_energies.pdf'.format(str(output_path)))

fig_E, ax_E = plt.subplots(nrows=2)
for key in energy_keys:
    ax_E[0].plot(t, data[key]-data[key][0], label=key+"'")
ax_E[1].plot(t, data['KE'], label='KE')

for ax in ax_E:
    if subrange:
        ax.set_xlim(t_min,t_max)
    ax.set_xlabel('time')
    ax.set_ylabel('energy density')
    ax.legend(loc='lower left')
fig_E.savefig('{:s}/energies_fluctuating.pdf'.format(str(output_path)))


fig_tau, ax_tau = plt.subplots(nrows=2)
for i in [0,1]:
    ax_tau[i].plot(t, data['τ_u1'], label=r'$\tau_{u1}$')
    ax_tau[i].plot(t, data['τ_u2'], label=r'$\tau_{u2}$')
    ax_tau[i].plot(t, data['τ_s1'], label=r'$\tau_{s1}$')
    ax_tau[i].plot(t, data['τ_s2'], label=r'$\tau_{s2}$')

for ax in ax_tau:
    if subrange:
        ax.set_xlim(t_min,t_max)
    ax.set_xlabel('time')
    ax.set_ylabel(r'$L_\inf(\tau)$')
    ax.legend(loc='lower left')
ax_tau[1].set_yscale('log')
fig_tau.savefig('{:s}/tau_error.pdf'.format(str(output_path)))

benchmark_set = ['KE', 'IE', 'Re']
i_ten = int(0.9*data[benchmark_set[0]].shape[0])
for benchmark in benchmark_set:
    print("{:s} = {:14.12g} +- {:4.2g} (averaged from {:g}-{:g})".format(benchmark, np.mean(data[benchmark][i_ten:]), np.std(data[benchmark][i_ten:]), t[i_ten], t[-1]))
print()
for benchmark in benchmark_set:
    print("{:s} = {:14.12g} (at t={:g})".format(benchmark, data[benchmark][-1], t[-1]))
print("total simulation time {:6.2g}".format(t[-1]-t[0]))
