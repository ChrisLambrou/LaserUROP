#!/usr/bin/env python

"""caller.py
Main front-end of script, to call experiments and GUI to run, and reads 
config files.

Usage:
    caller.py align
    caller.py autofocus
    caller.py calibrate
    caller.py centre
    caller.py tiled
    caller.py tiled_image
    caller.py timelapse_tiled_image
    caller.py compensate
    caller.py gui
    caller.py (-h | --help)

Options:
    -h, --help   Display this usage statement."""

from docopt import docopt
import nplab

import experiments as exp
import gui as g
import helpers as h
import image_proc as proc
import measurements as mmts
import microscope as micro

# Edit the paths of the config files.
CONFIG_PATH = './configs/config.yaml'


if __name__ == '__main__':
    sys_args = docopt(__doc__)

    # Calculate brightness of central spot by taking a tiled section of
    # images, cropping the central 55 x 55 pixels, moving to the region of 
    # maximum brightness and repeating. Return an array of the positions and 
    # the brightness.
    fun_list = [h.bake(proc.crop_array, args=['IMAGE_ARR'],
                       kwargs={'mmts': 'pixel', 'dims': 55}),
                h.bake(mmts.brightness, args=['IMAGE_ARR'])]

    try:
        if sys_args['gui']:
            gui = g.ScopeGUI(CONFIG_PATH)
            gui.run_gui()
        else:
            # Control pre-processing manually.
            scope = micro.CamScope(CONFIG_PATH, manual=True)
            if sys_args['autofocus']:
                focus = exp.AutoFocus(scope, CONFIG_PATH)
                focus.run()
            elif sys_args['centre']:
                exp.centre_spot(scope)
            elif sys_args['calibrate']:
                scope.calibrate()
            elif sys_args['tiled']:
                tiled = exp.Tiled(scope, CONFIG_PATH)
                tiled.run(func_list=fun_list)
            elif sys_args['tiled_image']:
                tiled = exp.TiledImage(scope, CONFIG_PATH)
                tiled.run()
            elif sys_args['timelapse_tiled_image']:
                tiled = exp.TimelapseTiledImage(scope, CONFIG_PATH)
                tiled.run()
            elif sys_args['align']:
                align = exp.Align(scope, CONFIG_PATH)
                align.run(func_list=fun_list)
            elif sys_args['compensate']:
                compensate = exp.CompensationImage(scope, CONFIG_PATH)
                compensate.run()
    finally:
        nplab.close_current_datafile()
