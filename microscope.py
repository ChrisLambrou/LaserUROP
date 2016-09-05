#!/usr/bin/env python

"""microscope.py
This script contains all the classes required to make the microscope work. This
includes the abstract Camera and Stage classes and their combination into a
single CamScope class that allows both to be controlled together. It is based
on the script by James Sharkey, which was used for the paper in Review of
Scientific Instruments titled: A one-piece 3D printed flexure translation stage
for open-source microscopy."""

import io
import sys
import time
import cv2
import smbus
import serial
import numpy as np
from scipy import ndimage as sn
from nplab.instrument import Instrument

import data_io as d
import helpers as h
import image_proc as proc

try:
    import picamera
    import picamera.array
    import pifastbayerarray
except ImportError:
    pass  # Don't fail on error; simply force cv2 camera later


class CamScope(Instrument):
    """Class to combine camera and stage into a single usable class. The
    microscope may be slow to be created; it must wait for the camera,and stage
    to be ready for use."""
    
    def __init__(self, config, **kwargs):
        """Use this class instead of using the Camera and Stage classes!
        :param config: Either a string specifying a path to the config file,
        ending in .yaml, or the dictionary of default configuration parameters.
        :param kwargs: Specify optional keyword arguments, which will
        override the defaults specified in the config file. Valid kwargs are:
        - resolution: A tuple of the image's resolution along (x, y).
        - max_resolution: A tuple of the camera's max resolution along (x, y).
        - cv2camera: Set to True if cv2-type camera will be used.
        - channel: Channel of I2C bus to connect to motors for the stage.
        - manual: Boolean for whether to control pre-processing manually.
        - um_per_pixel:
        - camera_stage_transform
        - mode
        - tolerance
        - max_iterations:"""
        super(CamScope, self).__init__()

        # If config is entered as a path string, the file from the path
        # will be read. If it is a dictionary, it is not changed. This
        # prevents the config file being read repeatedly by different
        # objects, rather than once and its value passed around.
        self.config_dict = d.make_dict(config, **kwargs)

        self._UM_PER_PIXEL = self.config_dict["um_per_pixel"]
        self.CAMERA_TO_STAGE_MATRIX = np.array(self.config_dict[
                                                   "camera_stage_transform"])
        self.camera = Camera(self.config_dict)
        self.stage = Stage(self.config_dict)
        # self.light = twoLED.Lightboard()

        # Set up data recording. Default values will be saved with the
        # group. TODO MOVE TO EACH METHOD.
        attrs = {}
        for key in ["max_resolution", "resolution", "greyscale", "videoport",
                    "mode", "iterator", "cv2camera", "manual", "channel",
                    "xyz_bound", "microsteps", "backlash", "override",
                    "um_per_pixel", "camera_stage_transform"]:
            attrs[key] = self.config_dict[key]

        self.gr = self.create_data_group('Run', attrs=attrs)

    def __del__(self):
        del self.camera
        del self.stage
        # del self.light

    def _camera_centre_move(self, template, mode=None):
        """Code to return the movement in pixels needed to centre a template
        image, as well as the actual camera position of the template."""
        # Mode values can either be specified in the config file itself,
        # at the __init__ stage or when using this method (by replacing the
        # default value of None).
        [mode] = h.check_defaults([mode], self.config_dict, ['mode'])

        if mode == 'bayer':
            width, height = self.camera.FULL_RPI_WIDTH, \
                            self.camera.FULL_RPI_HEIGHT
        else:
            width, height = self.camera.resolution

        template_pos = self.camera.find_template(template, box_d=-1,
                                                 decimal=True)
        # The camera needs to move (-delta_x, -delta_y); given (0,0) is top
        # left, not centre as needed.
        camera_move = (-(template_pos[0] - (width/2.0)),
                       -(template_pos[1] - (height/2.0)))
        assert ((camera_move[0] >= -(width/2.0)) and
                (camera_move[0] <= (width/2.0)))
        assert ((camera_move[1] >= -(height/2.0)) and
                (camera_move[1] <= (height/2.0)))
        self.log('INFO: Template is at {}; the camera needs to move '
                 '{}.'.format(camera_move, template_pos))
        return camera_move, template_pos

    def _camera_move_distance(self, camera_move):
        """Code to convert an (x,y) displacement in pixels to a distance in
        microns."""
        camera_move = np.array(camera_move)
        assert camera_move.shape == (2,)
        return np.power(np.sum(np.power(camera_move, 2.0)), 0.5) * \
            self._UM_PER_PIXEL

    def centre_on_template(self, template, tolerance=None,
                           max_iterations=None):
        """Given a template image, move the stage until the template is
        centred. Returns a tuple containing the number of iterations, the
        camera positions and the stage moves as (number, camera_positions,
        stage_moves), where number is returned as -1 * max_iterations if failed
        to converge.
        - If a tolerance is specified, keep iterating until the template is
        within this distance from the centre or the maximum number of
        iterations is exceeded. Measured in microns.
        - The max_iterations is how many times the code will run to attempt to
        centre the template image to within tolerance before aborting.
        - The stage will be held in position after motion, unless release is
        set to True.
        - A return value for iteration less than zero denotes failure, with the
        absolute value denoting the maximum number of iterations.
       - If centre_on_template(...)[0] < 0 then failure."""
        # TODO LINK THIS WITH THE CENTRE SPOT FUNCTION IN EXPERIMENTS.PY

        # Tolerance can be specified in init, for this function or config
        # file directly.
        [tolerance, max_iterations] = h.check_defaults(
            [tolerance, max_iterations], self.config_dict,
            ['tolerance', 'max_iterations'])

        stage_moves = []
        camera_move, position = self._camera_centre_move(template)
        camera_positions = [position]
        iteration = 0
        self.log('INFO: Begun centering template on image.')
        while ((self._camera_move_distance(camera_move)) > tolerance) and \
                (iteration < max_iterations):
            iteration += 1
            # Rotate to stage coordinates.
            stage_move = np.dot(camera_move, self.CAMERA_TO_STAGE_MATRIX)
            # Append the z-component of zero.
            stage_move = np.append(stage_move, [0], axis=1)
            # Need integer microsteps (round to zero).
            stage_move = np.trunc(stage_move).astype(int)
            self.stage.move_rel(stage_move)
            stage_moves.append(stage_move)
            time.sleep(0.5)
            camera_move, position = self._camera_centre_move(template)
            camera_positions.append(position)
        if iteration == max_iterations:
            self.log('ERROR: Aborted - tolerance not reached in %d '
                     'iterations.') % iteration
            print "Abort: Tolerance not reached in %d iterations" % iteration
            iteration *= -1

        self.gr.create_dataset('centre_on_template',
                               data=np.hstack((np.array(camera_positions),
                                               np.array(stage_moves))),
                               attrs={'num_iterations': iteration,
                                      'tolerance': tolerance,
                                      'data_labels': 'camera_xy, stage _xy'})
        return iteration, np.array(camera_positions), np.array(stage_moves)

    def calibrate(self, template=None, d=None, crop_frac=None):
        """Calibrate the stage-camera coordinates by finding the transformation
        between them.
        - If a template is specified, it will be used as the calibration track
        which is searched for in each image. The central half of the image will
        be used if one is not specified.
        - The size of the calibration square can be adjusted using d,
        in microsteps. Care should be taken that the template or central part
        of the image does not leave the field of view!"""

        # Get the default values - specify here, in init or config file.
        [d, crop_frac] = h.check_defaults([d, crop_frac], self.config_dict,
                                        ['d', 'crop_frac'])

        # Set up the necessary variables:
        self.camera.preview()
        pos = [np.array([d, d, 0]), np.array([d, -d, 0]),
               np.array([-d, -d, 0]), np.array([-d, d, 0])]
        camera_displacement = []
        stage_displacement = []

        # Move to centre (scope_stage takes account of backlash already).
        self.stage.move_to_pos([0, 0, 0])
        if template is None:
            # Default capture mode is bayer.
            self.log('INFO: No template specified. Capturing image.')
            template = self.camera.get_frame(mode='compressed')
            # Crop the central 1/2 of the image - can replace by my central
            # crop function or the general crop function (to be written).
            template = proc.crop_array(template, mmts='frac', dims=crop_frac)
            self.log('INFO: Cropped central {}\% of image.'.format(
                crop_frac * 100))
        time.sleep(1)

        # Store the initial configuration:
        init_cam_pos = np.array(self.camera.find_template(template, box_d=-1,
                                                          decimal=True))
        init_stage_vector = self.stage.position  # 3 component form
        init_stage_pos = init_stage_vector[0:2]  # xy part

        # Now make the motions in square specified by pos
        for p in pos:
            # Relate the microstep motion to the pixels measured on the
            # camera.
            self.stage.move_to_pos(np.add(init_stage_vector, p))
            time.sleep(1)
            cam_pos = np.array(self.camera.find_template(template, box_d=-1,
                                                         decimal=True))
            cam_pos = np.subtract(cam_pos, init_cam_pos)
            stage_pos = np.subtract(self.stage.position[0:2],
                                    init_stage_pos)
            camera_displacement.append(cam_pos)
            stage_displacement.append(stage_pos)

        self.log('INFO: Measurements complete.')
        self.stage.centre_stage()
        self.log('INFO: Moving to initial position.')

        camera_displacement = np.array(camera_displacement)
        stage_displacement = np.array(stage_displacement)
        self.gr.create_dataset('calibration_raw', data=np.hstack((
            camera_displacement, stage_displacement)), attrs={
            'microstep_increment': d, 'crop_fraction': crop_frac,
            'data_labels': 'camera_xy_stage_xy'})

        # Do the required analysis:
        self.log('INFO: Performing regression analysis.')
        camera_displacement -= np.mean(camera_displacement, axis=0)
        a, res, rank, s = np.linalg.lstsq(camera_displacement,
                                          stage_displacement)

        cali_results = {'pixel_microstep_transform': a, 'residuals': res,
                        'norm': np.linalg.norm(a)}
        for key in cali_results:
            self.gr.create_dataset(key, data=cali_results[key])

        print "matrix:  ", a
        print "residuals:  ", res
        print "norm:  ", np.linalg.norm(a)

        self.camera.preview(show=False)
        self.CAMERA_TO_STAGE_MATRIX = a
        self.log('INFO: CAMERA_TO_STAGE_MATRIX updated with new values.')
        return a


class Camera:

    def __init__(self, config_file, **kwargs):
        """An abstracted camera class. Always use through the CamScope class.
        :param kwargs:
        Valid ones include resolution, cv2camera, manual, and max_resolution
        :param width, height: Specify an image width and height.
        :param cv2camera: Choosing cv2camera=True allows testing on non RPi
        systems, though code will detect if picamera is not present and
        assume that cv2 must be used instead.
        :param manual: Specifies whether pre-processing (ISO, white balance,
        exposure) are to be manually controlled or not."""

        # If config_file is entered as a path string, the file from the path
        # will be read. If it is a dictionary, it is not changed. This
        # prevents the config file being read repeatedly by different
        # objects, rather than once and its value passed around.
        self.config_dict = d.make_dict(config_file, **kwargs)

        (self.FULL_RPI_WIDTH, self.FULL_RPI_HEIGHT) = self.config_dict[
            "max_resolution"]
        width = self.config_dict["resolution"][0]
        height = self.config_dict["resolution"][1]
        cv2camera = self.config_dict["cv2camera"]
        manual = self.config_dict["manual"]

        if "picamera" not in sys.modules:  # If cannot use picamera, force cv2
            cv2camera = True
        self._usecv2 = cv2camera
        self._view = False
        self._camera = None
        self._stream = None
        self.latest_frame = None

        # Check the resolution is valid.
        if 0 < width <= self.FULL_RPI_WIDTH and 0 < height <= \
                self.FULL_RPI_HEIGHT:
            # Note this is irrelevant for bayer images which always capture
            # at full resolution.
            self.resolution = (width, height)
        elif (width <= 0 or height <= 0) and not cv2camera:
            # Negative dimensions - use full sensor
            self.resolution = (self.FULL_RPI_WIDTH, self.FULL_RPI_HEIGHT)
        else:
            raise ValueError('Camera resolution has incorrect dimensions.')

        if self._usecv2:
            self._camera = cv2.VideoCapture(0)
            self._camera.set(3, width)  # Set width
            self._camera.set(4, height)  # Set height
        else:
            self._camera = picamera.PiCamera()
            self._rgb_stream = picamera.array.PiRGBArray(self._camera)
            self._camera.resolution = (width, height)
            self._fast_capture_iterator = None

        if manual and not cv2camera:
            # This only works with RPi camera.
            self._make_manual()

    def _close(self):
        """Closes the camera devices correctly. Called on deletion, do not call
         explicitly."""
        del self.latest_frame

        if self._usecv2:
            self._camera.release()
        else:
            if self._fast_capture_iterator is not None:
                del self._fast_capture_iterator
            self._camera.close()
            self._rgb_stream.close()

    def __del__(self):
        self._close()

    def _cv2_frame(self, greyscale):
        """Uses the cv2 VideoCapture method to obtain an image. Use get_frame()
        to access."""
        if not self._usecv2:
            raise TypeError("_cv2_frame() should ONLY be used when camera is "
                            "cv2.VideoCapture(0)")
        # We seem to be one frame behind always. So simply get current frame by
        # updating twice.
        frame = self._camera.read()[1]
        frame = self._camera.read()[1]
        if greyscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        return frame

    def _jpeg_frame(self, greyscale, videoport):
        """Captures via a jpeg, code may be adapted to save jpeg. Use
        get_frame() to access."""

        if self._fast_capture_iterator is not None:
            raise Warning("_jpeg_frame cannot be used while use_iterator(True)"
                          " is set.")

        stream = io.BytesIO()
        stream.seek(0)
        self._camera.capture(stream, format='jpeg', use_video_port=videoport)
        data = np.fromstring(stream.getvalue(), dtype=np.uint8)
        frame = cv2.imdecode(data, 1)
        return proc.make_greyscale(frame, greyscale)

    def _fast_bayer_frame(self, greyscale):
        """Capture a raw bayer image, de-mosaic it and output a BGR numpy
        array."""
        # Normally bayer images are not post-processed via white balance,
        # etc in # the camera and are thus of much worse quality if this were
        # done. But the combination of lenses in the Pi means that the
        # reverse is true.
        frame = pifastbayerarray.PiFastBayerArray(self._camera)
        self._camera.capture(frame, 'jpeg', bayer=True)
        frame = frame.demosaic()
        return proc.make_greyscale(frame, greyscale)

    def _bayer_frame(self, greyscale):
        """Capture a raw bayer image, de-mosaic it and output a BGR numpy
        array."""
        # Normally bayer images are not post-processed via white balance,
        # etc in # the camera and are thus of much worse quality if this were
        # done. But the combination of lenses in the Pi means that the
        # reverse is true.
        frame = picamera.array.PiBayerArray(self._camera)
        self._camera.capture(frame, 'jpeg', bayer=True)
        frame = (frame.demosaic() >> 2).astype(np.uint8)
        return proc.make_greyscale(frame, greyscale)

    def _bgr_frame(self, greyscale, videoport):
        """Captures straight to a BGR array object; a raw format. Use
        get_frame() to access."""

        if self._fast_capture_iterator is not None:
            raise Warning("_bgr_frame cannot be used while use_iterator(True) "
                          "is set.")
        self._rgb_stream.seek(0)
        self._camera.capture(self._rgb_stream, 'bgr', use_video_port=videoport)
        frame = self._rgb_stream.array
        return proc.make_greyscale(frame, greyscale)

    def _fast_frame(self, greyscale):
        """Captures really fast with the iterator method. Must be set up to run
        using use_iterator(True). Use get_frame() to access."""
        if self._fast_capture_iterator is None:
            raise Warning("_fast_frame cannot be used while use_iterator(True)"
                          " is not set.")
        self._rgb_stream.seek(0)
        self._fast_capture_iterator.next()
        frame = self._rgb_stream.array
        return proc.make_greyscale(frame, greyscale)

    def _make_manual(self):
        # Set ISO to the desired value
        self._camera.iso = 100
        # Wait for the automatic gain control to settle
        time.sleep(2)
        # Now fix the values
        self._camera.shutter_speed = self._camera.exposure_speed
        self._camera.exposure_mode = 'off'
        g = self._camera.awb_gains
        self._camera.awb_mode = 'off'
        self._camera.awb_gains = g

    def preview(self, show=True):
        """If using picamera, turn preview on and off. Set 'show' to False
        to turn off."""
        if not self._usecv2:
            if self._view and not show:
                self._camera.stop_preview()
                self._view = False
            elif not self._view and show:
                self._camera.start_preview(fullscreen=False, window=(
                    20, 20, int(640*1.5), int(480*1.5)))
                self._view = True
                time.sleep(2)   # Let the image be properly received.

    def get_frame(self, greyscale=None, videoport=None, mode=None):
        """Manages obtaining a frame from the camera device.
        :param greyscale: Toggle to obtain either a grey frame or BGR colour
        one.
        :param videoport: Use to select RPi option "use_video_port",
        which speeds up capture of images but has an offset compared to not
        using it.
        :param mode: Allows choosing RPi camera method; via 'compressed', 'bgr'
        or 'bayer'. 'bgr' is less CPU intensive than 'compressed'. If
        use_iterator(True) has been used to initiate the iterator method of
        capture, this method will be overridden to use that, regardless of
        choice."""

        [greyscale, videoport, mode] = h.check_defaults(
            [greyscale, videoport, mode], self.config_dict, [
                'greyscale', 'videoport', 'mode'])

        if self._usecv2:
            frame = self._cv2_frame(greyscale)
        elif self._fast_capture_iterator is not None:
            frame = self._fast_frame(greyscale)
        elif mode == 'compressed':
            frame = self._jpeg_frame(greyscale, videoport)
        elif mode == 'bgr':
            frame = self._bgr_frame(greyscale, videoport)
        elif mode == 'bayer':
            frame = self._bayer_frame(greyscale)
        elif mode == 'fast_bayer':
            frame = self._fast_bayer_frame(greyscale)
        else:
            raise ValueError('The parameter \'mode\' has an invalid value: '
                             '{}.'.format(mode))
        self.latest_frame = frame
        return frame

    def use_iterator(self, iterator=None):
        """For the RPi camera only, use the capture_continuous iterator to
        capture frames many times faster.
        :param iterator: Call this function with iterator=True to turn on the
        method, and use get_frame() as usual. To turn off the iterator and
        allow capture via jpeg/raw then call with iterator=False."""
        [iterator] = h.check_defaults([iterator], self.config_dict, ['iterator'])

        if self._usecv2:
            return
        if iterator:
            if self._fast_capture_iterator is None:
                self._fast_capture_iterator = self._camera.capture_continuous(
                    self._rgb_stream, 'bgr', use_video_port=True)
        else:
            self._fast_capture_iterator = None

    def set_roi(self, (x, y, width, height)=(0, 0, -1, -1), normed=False):
        """For the RPi camera only, set the Region of Interest on the sensor
        itself.
        - The tuple should be (x,y,width, height) so x,y position then width
        and height in pixels. Setting width, height negative will use maximum
        size. Reset by calling as set_roi().
        - Take great care: changing this will change the camera coordinate
        system, since the zoomed in region will be treated as the whole
        image afterwards.
        - Will NOT behave as expected if applied when already zoomed!
        - Set normed to True to adjust raw normalise coordinates."""
        if self._usecv2:
            pass
        else:
            # TODO Binning and native resolution hard coding
            (frame_w, frame_h) = self.resolution
            if width <= 0:
                width = frame_w
            if height <= 0:
                height = frame_h
            if not normed:
                self._camera.zoom = (x * 1.0 / frame_w,
                                     y * 1.0 / frame_h,
                                     width * 1.0 / frame_w,
                                     height * 1.0 / frame_h)
            else:
                self._camera.zoom = (x, y, width, height)

    def find_template(self, template, frame=None, bead_pos=None, box_d=None,
                      centre_mass=None, cross_corr=None, tol=None,
                      decimal=None, mode=None):
        """Finds a dot given a camera and a template image. Returns a camera
        coordinate for the centre of where the template has matched a part
        of the image. Default behaviour is to search the entire image.
        :param template: The template image array to search for.
        :param frame: Providing a frame as an argument will allow searching
        of an existing image, which avoids taking a frame from the camera.
        :param bead_pos: Specifying a bead_pos will centre the search on that
        location; use when a previous location is known and bead has not moved.
        The camera co-ordinate system should be used. Default is (-1,-1) which
        actually looks at the centre.
        :param box_d = Specifying box_d allows the dimensions of the search
        box to be altered. A negative or zero value will search the whole
        image. box_d ought to be larger than the template dimensions.
        :param centre_mass: Toggle to use Centre of Mass searching (default:
        True) or Maximum Value (False).
        :param cross_corr: Use either Cross Correlation (cross_corr=True,
        the default) or Square Difference (False) to find the likely position
        of the template.
        :param tol: The tolerance in the thresholding when filtering.
        :param decimal: Determines whether a float or int is returned.
        :param mode: Mode of image capture."""

        [bead_pos, box_d, centre_mass, cross_corr, tol, decimal, mode] = \
            h.check_defaults([bead_pos, box_d, centre_mass, cross_corr, tol,
                            decimal, mode], self.config_dict,
                           ['bead_pos', 'box_d', 'centre_mass', 'cross_corr',
                            'tol', 'decimal', 'mode'])

        # If the template is a colour image (3 channels), make greyscale.
        if len(template.shape) == 3:
            template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # Take an image if one is not supplied.
        if frame is None:
            frame = self.get_frame(greyscale=True, mode=mode)
        else:
            frame = frame.copy()

        # These offsets are needed to find position in uncropped image.
        frame_x_off, frame_y_off = 0, 0
        temp_w, temp_h = template.shape[::-1]
        if box_d > 0:  # Only crop if box_d is positive
            if bead_pos == (-1, -1):  # Search the centre if default:
                frame_w, frame_h = frame.shape[::-1]
                frame_x_off, frame_y_off = int(frame_w / 2 - box_d / 2), int(
                    frame_h / 2 - box_d / 2)
            else:  # Otherwise search centred on bead_pos
                frame_x_off, frame_y_off = int(bead_pos[0] - box_d / 2), \
                                           int(bead_pos[1] - box_d / 2)
            frame = frame[frame_y_off: frame_y_off + box_d,
                          frame_x_off: frame_x_off + box_d]

        # Check the size of the frame is bigger than the template to avoid
        # OpenCV Error:
        frame_w, frame_h = frame.shape[::-1]
        if (frame_w < temp_w) or (frame_h < temp_h):
            raise RuntimeError("Template larger than Frame dimensions! %dx%d "
                               "> %dx%d" % (temp_w, temp_h, frame_w, frame_h))

        # If all good, then do the actual correlation:
        # Use either Cross Correlation or Square Difference to match
        if cross_corr:
            corr = cv2.matchTemplate(frame, template, cv2.TM_CCORR_NORMED)
        else:
            corr = cv2.matchTemplate(frame, template, cv2.TM_SQDIFF_NORMED)
            # Actually want minima with this method so reverse values.
            corr *= -1.0

        corr += (corr.max()-corr.min()) * tol - corr.max()
        corr = cv2.threshold(corr, 0, 0, cv2.THRESH_TOZERO)[1]
        if centre_mass:  # Either centre of mass:
            # Get co-ordinates of peak from origin at top left of array.
            peak = sn.measurements.center_of_mass(corr)
            # Array indexing means peak has (y,x) not (x,y):
            centre = (peak[1] + temp_w/2.0, peak[0] + temp_h/2.0)
        else:  # or crudely using max pixel
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(corr)
            centre = (max_loc[0] + temp_w/2.0, max_loc[1] + temp_h/2.0)
        centre = (centre[0] + frame_x_off, frame_y_off + centre[1])

        corr *= 255/corr.max()
        if not decimal:
            centre = (int(centre[0]), int(centre[1]))
        print "Template found at {}".format(centre)
        return centre


class Stage:

    def __init__(self, config_file, **kwargs):
        """Class representing a 3-axis microscope stage.
        :param config_file: Either file path or dictionary.
        :param kwargs: Valid ones are the xyz_bound, microsteps and channel."""

        # If config_file is entered as a path string, the file from the path
        # will be read. If it is a dictionary, it is not changed. This
        # prevents the config file being read repeatedly by different
        # objects, rather than once and its value passed around.
        self.config_dict = d.make_dict(config_file, **kwargs)

        # Check these bounds.
        self._XYZ_BOUND = np.array(self.config_dict["xyz_bound"])
        # How many micro-steps per step?
        self._MICROSTEPS = self.config_dict["microsteps"]

        self.bus = smbus.SMBus(self.config_dict["channel"])
        time.sleep(2)
        self._position = np.array([0, 0, 0])
        self.drift = np.array([0, 0, 0])

    @property
    def position(self):
        """The current position of the stage"""
        return self._position - self.drift

    def move_rel(self, vector, backlash=None, override=None):
        """Move the stage by (x,y,z) micro steps.
        :param vector: The increment to move by along [x, y, z].
        :param backlash: An array of the backlash along [x, y, z].
        :param override: Set to True to ignore the limits set by _XYZ_BOUND,
        otherwise an error is raised when the bounds are exceeded."""

        [backlash, override] = h.check_defaults(
            [backlash, override], self.config_dict, ['backlash', 'override'])

        # Check backlash  and the vector to move by have the correct format.
        assert np.all(backlash >= 0), "Backlash must >= 0 for all [x, y, z]."
        backlash = h.verify_vector(backlash)
        r = h.verify_vector(vector)

        # Generate the list of movements to make. If backlash is [0, 0, 0],
        # there is only one motion to make.
        movements = []
        if np.any(backlash != np.zeros(3)):
            # Subtract an extra amount where vector is negative.
            r[r < 0] -= backlash[np.where(r < 0)]
            r2 = np.zeros(3)
            r2[r < 0] = backlash[np.where(r < 0)]
            movements.append(r2)
        movements.insert(0, r)

        for movement in movements:
            new_pos = np.add(self._position, movement)
            # If all elements of the new position vector are inside bounds (OR
            # overridden):
            if np.all(np.less_equal(
                    np.absolute(new_pos), self._XYZ_BOUND)) or override:
                _move_motors(self.bus, *movement)
                self._position = new_pos
            else:
                raise ValueError('New position is outside allowed range.')

    def move_to_pos(self, final, override=None):

        [override] = h.check_defaults([override], self.config_dict, ["override"])

        new_position = h.verify_vector(final)
        rel_mov = np.subtract(new_position, self.position)
        return self.move_rel(rel_mov, override=override)

    def focus_rel(self, z):
        """Move the stage in the Z direction by z micro steps."""
        self.move_rel([0, 0, z])

    def centre_stage(self):
        """Move the stage such that self.position is (0,0,0) which in theory
        centres it."""
        self.move_to_pos([0, 0, 0])

    def _reset_pos(self):
        # Hard resets the stored position, just in case things go wrong.
        self._position = np.array([0, 0, 0])


class BrightnessSensor:
    """Class to read brightness value from sensor by providing a Serial
    command to the Arduino."""

    def __init__(self, tty="/dev/ttyACM0"):
        # Initialise connection as appropriate.
        self.ser = serial.Serial(tty)

    def read(self):
        """Read the voltage value from the Arduino."""
        self.ser.write('h')
        self.ser.read()  # change to get all the data

    def __del__(self):
        self.ser.close()


def _move_motors(bus, x, y, z):
    """Move the motors for the connected module (addresses hardcoded) by a
    certain number of steps.
    :param bus: The smbus.SMBus object connected to appropriate i2c channel.
    :param x: Move x-direction-controlling motor by specified number of steps.
    :param y: "
    :param z: "."""
    [x, y, z] = [int(x), int(y), int(z)]

    # The arguments for write_byte_data are: the I2C address of each motor,
    # the register and how much to move it by. Currently hardcoded in.
    bus.write_byte_data(0x08, x >> 8, x & 255)
    bus.write_byte_data(0x10, y >> 8, y & 255)
    bus.write_byte_data(0x18, z >> 8, z & 255)

    # Empirical formula for how micro step values relate to rotational speed.
    # This is only valid for the specific set of motors tested.
    time.sleep(np.ceil(max([abs(x), abs(y), abs(z)]))*(1.4/1000) + 0.1)

if __name__ == '__main__':
    scope = CamScope('./configs/config.yaml')
    scope.calibrate()
