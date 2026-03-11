"""
Plot averaged quantities from joint analysis file.

Usage:
    plot_fluxes.py <files>... [options]

Options:
    --output=<output>    Output directory; if blank a guess based on likely case name will be made

    --fraction=<frac>    Fraction of time to average over, from end of simulation [default: 0.1]
"""
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pathlib
import h5py
import scipy.integrate as sci

import logging
logger = logging.getLogger(__name__.split('.')[-1])
dlog = logging.getLogger('matplotlib')
dlog.setLevel(logging.WARNING)

from docopt import docopt
args = docopt(__doc__)

import dedalus.public as de
from dedalus.tools import post
from dedalus.tools.general import natural_sort
files = natural_sort(args['<files>'])
case = args['<files>'][0].split('/')[0]

if args['--output'] is not None:
    output_path = pathlib.Path(args['--output']).absolute()
else:
    data_dir = case +'/'
    output_path = pathlib.Path(data_dir).absolute()

def accumulate_files(filename,start,count,file_list):
    if start==0:
        file_list.append(filename)

file_list = []
post.visit_writes(files,  accumulate_files, file_list=file_list)
logger.debug(file_list)

data = {}
z = None
times = None
for file in file_list:
    logger.debug("opening file: {}".format(file))
    f = h5py.File(file, 'r')
    data_slices = (slice(None), 0, 0, slice(None))
    for task in f['tasks']:
        if '(z)' in task: # fluxes denoted with 'f(z)'
            logger.info("task: {}".format(task))
            if task in data:
                data[task] = np.append(data[task], f['tasks'][task][data_slices], axis=0)
            else:
                data[task] = np.array(f['tasks'][task][data_slices])
            if z is None:
                z = f['tasks'][task].dims[3][0][:]
    if times is None:
        times = f['scales/sim_time'][:]
    else:
        times = np.append(times, f['scales/sim_time'][:])
    f.close()

n_times = times.shape[0]
if args['--fraction']:
    i_cut = int((1-float(args['--fraction']))*n_times)
    times = times[i_cut:]
    for task in data:
        data[task] = data[task][i_cut:]

logger.info("averaged from t={:.3g}--{:.3g}".format(min(times),max(times)))


print(file_list)
for task in data:
    print(task, data[task].shape)

def time_avg(f, axis=0):
    n_avg = f.shape[axis]
    return np.squeeze(np.sum(f, axis=axis))/n_avg

s_avg = time_avg(data['s(z)'])
s0_avg = time_avg(data['s0(z)'])
fig_s, ax_s = plt.subplots(figsize=(4.5,4/1.5))
fig_s.subplots_adjust(top=0.9, right=0.95, bottom=0.2, left=0.15)
for si in data['s(z)']:
    ax_s.plot(z, si, alpha=0.3)
ax_s.plot(z, s_avg, linewidth=1.5, color='black')
ax_s.plot(z, s0_avg, linewidth=1, color='black', linestyle='dashed')
fig_s.savefig('{:s}/thermal_profile.png'.format(str(output_path)), dpi=300)

Ma_avg = time_avg(data['Ma(z)'])
Re_avg = time_avg(data['Re(z)'])

fig, ax = plt.subplots(figsize=(4.5,4/1.5))
fig.subplots_adjust(top=0.9, right=0.85, bottom=0.2, left=0.15)
ax.plot(z, Re_avg, linewidth=2, label='Re')
ax_r = ax.twinx()
ax_r.plot(z, Ma_avg, linewidth=2, label='Ma', linestyle='dotted', color='tab:red')
ax.set_xlabel(r'$z$')
ax.set_ylabel(r'$\mathrm{Re}$')
ax_r.set_ylabel(r'$\mathrm{Ma}$')

handles_l, labels_l = ax.get_legend_handles_labels()
handles_r, labels_r = ax_r.get_legend_handles_labels()
handles = handles_l + handles_r
labels = labels_l + labels_r
ax.legend(handles,labels)
fig.savefig('{:s}/fluid_properties_with_depth.png'.format(str(output_path)), dpi=300)

F_h = time_avg(data['F_h(z)'])
F_κ = time_avg(data['F_κ(z)'])
F_KE = time_avg(data['F_KE(z)'])
F_PE = time_avg(data['F_PE(z)'])
F_μ = time_avg(data['F_μ(z)'])

F_tot = F_h + F_κ + F_μ + F_KE + F_PE

fig, ax = plt.subplots(figsize=(4.5,4/1.5))
fig.subplots_adjust(top=0.9, right=0.95, bottom=0.2, left=0.15)
ax.plot(z, F_tot, color='black', label=r'$F_\mathrm{tot}$', linewidth=3)
ax.plot(z, F_h, label=r'$F_\mathrm{h}$')
ax.plot(z, F_KE, label=r'$F_\mathrm{KE}$')
ax.plot(z, F_PE, label=r'$F_\mathrm{PE}$')
ax.plot(z, F_κ, label=r'$F_\kappa$')
ax.plot(z, F_μ, label=r'$F_\mu$')
ax.axhline(y=F_tot[-1], linestyle='dotted', color='black')
ax.legend()
ax.set_ylabel(r'Flux')
ax.set_xlabel(r'$z$')
fig.savefig('{:s}/flux_balance.png'.format(str(output_path)), dpi=300)
