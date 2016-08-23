"""
Script to profile capture via PiBayerArray
"""
import picamera
import picamera.array
import numpy as np
import time
import pifastbayerarray

def take_bayer_image(fast=True, **kwargs):
    cam = picamera.PiCamera()
    cam.start_preview()
    time.sleep(2)

    if fast:
        array_class = pifastbayerarray.PiFastBayerArray
    else:
        array_class = picamera.array.PiBayerArray

    start = time.time()
    bayer = array_class(cam)
    cam.capture(bayer, 'jpeg', bayer=True)
    capture_t = time.time()
    rgb = bayer.demosaic(**kwargs)
    demosaic_t = time.time()
    print "capture time", capture_t - start
    print "demosaic time", demosaic_t - capture_t
    cam.stop_preview()
    cam.close()
    return rgb



