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
    cmap = None
    # Layout


    # Plot writes
    with h5py.File(filename, mode='r') as f:
        t = np.array(f['scales/sim_time'])
        print(f['scales/write_number'][:])
        for i, task in enumerate(tasks):
            time = t
            center_zero=False
            title = task
            savename_func = lambda write: '{:s}_{:06d}.png'.format(title, write)
            task = f['tasks'][task]
            x = task.dims[1][0][:]
            z = task.dims[2][0][:]
            Lz = np.max(z)-np.min(z)
            Lx = np.max(x)-np.min(x)
            figsize = (6.4, 1.2*Lz/Lx*6.4)
            print(x.shape, z.shape, task.shape)
            for k in range(len(t)):
                time = t[k]
                fig, ax = plt.subplots(1, figsize=figsize)
                ax.set_aspect(1)
                pcm = ax.pcolormesh(x, z, task[k,:].T, shading='nearest',cmap=cmap)
                pmin,pmax = pcm.get_clim()
                if center_zero:
                    cNorm = matplotlib.colors.TwoSlopeNorm(vmin=pmin, vcenter=0, vmax=pmax)
                    logger.info("centering zero: {} -- 0 -- {}".format(pmin, pmax))
                else:
                    cNorm = matplotlib.colors.Normalize(vmin=pmin, vmax=pmax)
                pcm = ax.pcolormesh(x, z, task[k,:].T, shading='nearest',cmap=cmap, norm=cNorm)
                ax_cb = fig.add_axes([0.91, 0.4, 0.02, 1-0.4*2])
                cb = fig.colorbar(pcm, cax=ax_cb)
                cb.formatter.set_scientific(True)
                cb.formatter.set_powerlimits((0,4))
                cb.ax.yaxis.set_offset_position('left')
                cb.update_ticks()
                fig.subplots_adjust(left=0.1,right=0.9,top=0.95)
                if title is not None:
                    ax_cb.text(0.5, 1.75, title, horizontalalignment='center', verticalalignment='center', transform=ax_cb.transAxes)
                if time is not None:
                    ax_cb.text(0.25, -0.5, "t = {:.0f}".format(time), horizontalalignment='left', verticalalignment='center', transform=ax_cb.transAxes)
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
    tasks = args['--tasks'].split(',')
    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], main, tasks=tasks, output=output_path)
