"""
Plot planes from joint analysis files.

Usage:
    plot_2d_series.py <files>... [options]

Options:
    --output=<dir>     Output directory [default: ./frames]
    --tasks=<tasks>    Tasks to plot [default: s]
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
    savename_func = lambda write: 'write_{:06}.png'.format(write)
    # Layout


    # Plot writes
    with h5py.File(filename, mode='r') as f:
        t = np.array(f['scales/sim_time'])
        print(f['scales/write_number'][:])
        for i, task in enumerate(tasks):
            time = t
            center_zero=False
            title = task
            task = f['tasks'][task]
            x = task.dims[1][0][:]
            z = task.dims[2][0][:]
            Lz = np.max(z)-np.min(z)
            Lx = np.max(x)-np.min(x)
            print(x.shape, z.shape, task.shape)
            for k in range(len(t)):
                fig, ax = plt.subplots(1)
                ax.set_aspect(1)
                ax.pcolormesh(x, z, task[k,:].T, shading='nearest')
                savename = savename_func(f['scales/write_number'][k])
                savepath = output.joinpath(savename)
                fig.savefig(str(savepath), dpi=dpi)
                #fig.clear()
                plt.close(fig)


if __name__ == "__main__":

    import pathlib
    from docopt import docopt
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)
    print(args)
    tasks = args['--tasks']
    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], main, tasks=tasks, output=output_path)
