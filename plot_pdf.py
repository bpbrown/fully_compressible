"""
Plot PDFs from joint analysis files.

Usage:
    plot_PDF.py <files>... [options]

Options:
    --output=<dir>     Output directory; defaults to case dir
    --tasks=<tasks>    Tasks to plot [default: s,vorticity]
"""

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from dedalus.extras import plot_tools


def main(filename, start, count, tasks, output):
    """Save plot of specified tasks for given range of analysis writes."""

    # Plot settings
    dpi = 300
    title_func = lambda sim_time: 't = {:.3f}'.format(sim_time)
    cmap = None
    # Layout

    avg_PDF = {}
    N_PDF = {}
    avg_bin_edges = {}
    for task in tasks:
        avg_PDF[task] = 0
        N_PDF[task] = 0
        avg_bin_edges[task] = 0

    # Plot writes
    with h5py.File(filename, mode='r') as f:
        logger.info('reading file {:}'.format(filename))
        t = np.array(f['scales/sim_time'])
        for i, task in enumerate(tasks):
            time = t
            center_zero=False
            title = task
            savename_func = lambda write: '{:s}_{:06d}_PDF.png'.format(title, write)
            data = f['tasks'][task]
            x = data.dims[2][0][:]
            z = data.dims[3][0][:]
            Δx = np.gradient(x, edge_order=2)
            Δz = np.gradient(z, edge_order=2)
            area = np.expand_dims(Δx, axis=1)*np.expand_dims(Δz, axis=0)
            vmin = np.min(data) # this logic works for one file, not for many files
            vmax = np.max(data) # this logic works for one file, not for many files
            Lz = np.max(z)-np.min(z)
            Lx = np.max(x)-np.min(x)
            figsize = (6.4, 6.4/1.6)

            for k in range(len(t)):
                data_slice = (k,0,slice(None),slice(None))
                time = t[k]
                fig, ax = plt.subplots(1, figsize=figsize)
                PDF, bin_edges = np.histogram(data[data_slice], weights=area, density=True,
                                              bins=100, range=(vmin,vmax))
                ax.stairs(PDF, edges=bin_edges, fill=True, alpha=0.3)
                ax.set_yscale('log')
                fig.subplots_adjust(left=0.075,right=0.95,top=0.95, bottom=0.2)
                savename = savename_func(f['scales/write_number'][k])
                savepath = output.joinpath(savename)
                fig.savefig(str(savepath), dpi=dpi)
                #fig.clear()
                plt.close(fig)
                avg_PDF[task] += PDF
                N_PDF[task] += 1
            avg_bin_edges[task] = bin_edges

    for task in tasks:
        avg_PDF[task] /= N_PDF[task]
        fig, ax = plt.subplots(1, figsize=figsize)
        ax.stairs(avg_PDF[task], edges=avg_bin_edges[task], fill=True, alpha=0.3)
        ax.set_yscale('log')
        fig.subplots_adjust(left=0.075,right=0.95,top=0.95, bottom=0.2)
        savename = '{:s}_average_PDF.png'.format(task)
        savepath = output.joinpath(savename)
        fig.savefig(str(savepath), dpi=dpi)
        plt.close(fig)

if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync
    import logging
    logger = logging.getLogger(__name__)

    dlog = logging.getLogger('matplotlib')
    dlog.setLevel(logging.WARNING)


    args = docopt(__doc__)
    tasks = args['--tasks'].split(',')
    if args['--output']:
        output_path = pathlib.Path(args['--output']).absolute()
    else:
        case_name = args['<files>'][0].split('/')[0]
        output_path = pathlib.Path(case_name+'/frames').absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], main, tasks=tasks, output=output_path)
