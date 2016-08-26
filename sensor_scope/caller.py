#!/usr/bin/env python

"""caller.py
Main front-end of script, to call experiments and GUI to run, and reads 
config files.[

Usage:
    caller.py align [<configs>...]
    caller.py autofocus [<configs>...]
    caller.py tiled [<configs>...]
    caller.py raster_3d [<configs>...]
    caller.py controller [<configs>...]
    caller.py move [--x=<x>] [--y=<y>] [--z=<z>]
    caller.py timed
    caller.py beam_walk
    caller.py (-h | --help)

Options:
    -h, --help   Display this usage statement."""

from docopt import docopt
import nplab

import experiments as exp
import controller as c
import microscope as micro

# Edit the paths of the config files.
DEFAULT_PATH = '../configs/config.yaml'

if __name__ == '__main__':
    sys_args = docopt(__doc__)

    if not sys_args['<configs>']:
        # configs is a list of paths of config files, each of which will be
        # tested in turn.
        configs = [DEFAULT_PATH]
    else:
        configs = sys_args['<configs>']

    # List of baked functions to do post-processing, if any.

    for config in configs:
        if not sys_args['move']:
            scope = micro.SensorScope(config)
            if sys_args['autofocus']:
                focus = exp.AlongZ(scope, config)
                focus.run()
            elif sys_args['tiled']:
                tiled = exp.RasterXY(scope, config)
                tiled.run()
            elif sys_args['align']:
                align = exp.Align(scope, config)
                align.run()
            elif sys_args['controller']:
                gui = c.KeyboardControls(scope, config)
                gui.run_gui()
            elif sys_args['raster_3d']:
                raster3d = exp.RasterXYZ(scope, config)
                raster3d.run()
            elif sys_args['timed']:
                timed = exp.TimedMeasurements(scope, config)
                timed.run()
            elif sys_args['beam_walk']:
                beam_walk = exp.BeamWalk(scope, config, raster_n_step=[[64, 1000]])
                beam_walk.run()
        elif sys_args['move']:
            positions = []
            for axis in ['--x', '--y', '--z']:
                if sys_args[axis] is None:
                    sys_args[axis] = 0
                positions.append(int(sys_args[axis]))
            print "Moved by {}".format(positions)
            stage = micro.Stage(config)
            stage.move_rel(positions)

    nplab.close_current_datafile()
