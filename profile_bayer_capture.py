"""
Script to profile capture via PiBayerArray
"""
import picamera
import picamera.array
import numpy as np
import time
import pifastbayerarray

if __name__ == "__main__":
    cam = picamera.PiCamera()
    cam.start_preview()
    time.sleep(2)

#    array_class = pifastbayerarray.PiFastBayerArray
    array_class = picamera.array.PiBayerArray

    capture_times = []
    demosaic_times = []
    for i in range(5):
        start = time.time()
        bayer = array_class(cam)
        cam.capture(bayer, 'jpeg', bayer=True)
        capture_t = time.time()
        rgb = bayer.demosaic()
        demosaic_t = time.time()
        capture_times.append(capture_t-start)
        demosaic_times.append(demosaic_t-capture_t)
#    print "bayer shape", bayer.shape
    print "rgb shape", rgb.shape

    print "capture times", capture_times
    print "demosaic times", demosaic_times
    print "average capture time", np.mean(capture_times)
    print "average demosaic time", np.mean(demosaic_times)
    cam.stop_preview()
    cam.close()


