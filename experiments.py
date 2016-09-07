#!/usr/bin/env python

"""experiments.py
Contains functions to perform a set of measurements and output the results to a
datafile or graph. These functions are built on top of measurements.py,
microscope.py, image_proc.py and data_io.py."""

import time as t
import time
import cv2
import numpy as np
from nplab.experiment.experiment import Experiment
from scipy import ndimage as sn

import data_io as d
import end_functions as e
import helpers as h
import image_proc as proc
import measurements as mmts


class AutoFocus(Experiment):
    """Autofocus the image on the camera by varying z only, using the Laplacian
    method of calculating the sharpness and compressed JPEG image."""

    def __init__(self, microscope, config_file, **kwargs):
        """
        :param microscope: A microscope object.
        :param config_file: A string with a path to the YAML config file.
        :param kwargs: Valid kwargs are: backlash, z_range, crop_frac, mode."""
        super(AutoFocus, self).__init__()
        self.config_dict = d.make_dict(config_file, **kwargs)
        self.scope = microscope
        # Preview the microscope so we can see auto-focusing.
        self.scope.camera.preview()

    def run(self, backlash=None, mode=None, z_range=None, crop_frac=None):
        # Read the default parameters.
        [backlash, mode, z_range, crop_frac] = h.check_defaults(
            [backlash, mode, z_range, crop_frac], self.config_dict,
            ['backlash', 'mode', 'mmt_range', 'crop_fraction'])
        print mode
        # Set up the data recording.
        attributes = {'resolution': self.scope.camera.resolution,
                      'backlash': backlash,
                      'capture_func': mode}

        funcs = [h.bake(proc.crop_array,
                        args=['IMAGE_ARR'],
                        kwargs={'mmts': 'frac',
                                'dims': crop_frac}),
                 h.bake(mmts.sharpness_lap, args=['IMAGE_ARR'])]
        # At the end, move to the position of maximum brightness.
        end = h.bake(e.max_fourth_col, args=['IMAGE_ARR', self.scope])

        for n_step in z_range:
            # Allow the iteration to take place as many times as specified
            # in the scope_dict file.
            _move_capture(self, {'z': [n_step]}, 'bayer', func_list=funcs,
                          save_mode=None, end_func=end)
            print self.scope.stage.position


class Tiled(Experiment):
    """Class to conduct experiments where a tiled sequence of images is 
    taken and post-processed.
    Valid kwargs are: step_pair, backlash, focus."""

    def __init__(self, microscope, config_file, **kwargs):
        super(Tiled, self).__init__()
        self.config_file = d.make_dict(config_file, **kwargs)
        self.scope = microscope
        self.scope.log('INFO: Initiating Tiled experiment.')
        self.scope.camera.preview()

    def run(self, func_list=None, save_mode=None, step_pair=[None, None]):
        # Get default values.
        [step_pair[0], step_pair[1]] = h.check_defaults([
            step_pair[0], step_pair[1]], self.config_file, ["n", "steps"])

        # Set up the data recording.
        attributes = {'n': step_pair[0], 'step_increment': step_pair[1],
                      'backlash': self.config_file["backlash"],
                      'focus': self.config_file["focus"]}

        # Set mutable default values.
        if func_list is None:
            func_list = [h.bake(h.unchanged, args=['IMAGE_ARR'])]

        end = h.bake(e.max_fourth_col, args=['IMAGE_ARR', self.scope])

        # Take measurements and move to position of maximum brightness.
        _move_capture(self, {'x': [step_pair], 'y': [step_pair]},
                      image_mode='compressed', func_list=func_list,
                      save_mode=save_mode, end_func=end)
        print self.scope.stage.position


def autofocus(scope, dz, data_group=None):
    """Take pictures at a range of z positions and move to the sharpest."""
    print "Autofocusing"
    sharpnesses = []
    positions = []

    for index, pos in raster_scan(scope.stage, dz=dz):
        image = _capture_image_from_microscope(scope)
        if data_group is not None:
            _save_image_to_datagroup(data_group, image, pos)
        sharpness = mmts.sharpness_clippedlog(image)
        sharpnesses.append(sharpness)
        positions.append(pos)

    best_index = np.argmax(sharpnesses)
    best_position = positions[best_index]

    # Use quadratic curve fitting to determine the best z-value, based on the best sample
    # and its two adjacent neighbours.
    if 0 < best_index < len(sharpnesses) - 1:
        best_position = _quadratic_curve_fit_on_z_values(positions, sharpnesses, best_index)

    # And finally move the stage to the best focus position.
    scope.stage.move_to_pos(best_position)


def _quadratic_curve_fit_on_z_values(positions, sharpnesses, best_index):
    best_position = positions[best_index]
    z1 = positions[best_index - 1][2]
    z2 = positions[best_index][2]
    z3 = positions[best_index + 1][2]
    s1 = sharpnesses[best_index - 1]
    s2 = sharpnesses[best_index]
    s3 = sharpnesses[best_index + 1]
    zbest, sbest = _calculate_parabola_vertex(z1, s1, z2, s2, z3, s3)
    return [best_position[0], best_position[1], int(round(zbest))]


def _calculate_parabola_vertex(x1, y1, x2, y2, x3, y3):
    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    B = (x3 * x3 * (y1 - y2) + x2 * x2 * (y3 - y1) + x1 * x1 * (y2 - y3)) / denom
    C = (x2 * x3 * (x2 - x3) * y1 + x3 * x1 * (x3 - x1) * y2 + x1 * x2 * (x1 - x2) * y3) / denom
    vx = -B / (2 * A)
    vy = C - B * B / (4 * A)
    return vx, vy


class TiledImage(Experiment):
    """Class to conduct experiments where a tiled sequence of images is 
    taken and post-processed.
    Valid kwargs are: step_pair, backlash, focus."""

    def __init__(self, microscope, config_file, **kwargs):
        super(TiledImage, self).__init__()
        self.config_file = d.make_dict(config_file, **kwargs)
        self.scope = microscope
        self.scope.log('INFO: starting tiled image.')
        self.scope.camera.preview()

    def run(self, data_group=None):
        # Get default values.
        n = self.config_file["n"]
        step_increment = self.config_file["steps"]
        fine_af_dz = symmetric_range(
            self.config_file["autofocus_fine_n"],
            self.config_file["autofocus_fine_step"])
        coarse_af_dz = symmetric_range(
            self.config_file["autofocus_coarse_n"],
            self.config_file["autofocus_coarse_step"])

        # Set up the data recording.
        attributes = {'n': n, 'step_increment': step_increment,
                      'backlash': self.config_file["backlash"],
                      'focus': self.config_file["focus"]}

        # Make a new data group to store results
        if data_group is None:
            data_group = self.create_data_group("tiled_image_%d")

        # Now move over the grid of positions and save images.
        displacements = symmetric_range(n, step_increment)
        z_shift = 0
        for index, pos in raster_scan(self.scope.stage,
                                      dx=displacements,
                                      dy=displacements):
            # We autofocus, remembering the previous z position
            self.scope.stage.focus_rel(z_shift)
            if index[0] == 0:
                autofocus(self.scope, coarse_af_dz, log=True)
            autofocus(self.scope, fine_af_dz, log=True, data_group=data_group)
            z_shift = self.scope.stage.position[2]
            # now capture and save the image at the best z-axis position of focus.
            image = _capture_image_from_microscope(self.scope)

            _save_image_to_datagroup(data_group, image, self.scope.stage.position)


class CompensationImage(Experiment):
    def __init__(self, microscope, config_file, **kwargs):
        super(CompensationImage, self).__init__()
        self.config_file = d.make_dict(config_file, **kwargs)
        self.scope = microscope
        self.scope.log('INFO: starting compensation image.')
        self.scope.camera.preview()

    def run(self, data_group=None):
        print "Compensation Image currently does nothing"

        compensation_image_path = self.config_file["compensation_image_path"]

        print "Compensation image path = '%s'" % (compensation_image_path)


class TimelapseTiledImage(TiledImage):
    """Take a TiledImage every n minutes (assumint the TiledImage takes less time)"""

    def run(self):
        interval_minutes = self.config_file["timelapse_interval_minutes"]
        last_image = time.time() - interval_minutes * 60
        while self.wait_or_stop(max(0.1, interval_minutes * 60 - (time.time() - last_image))):
            last_image = time.time()
            print "acquiring another tiled image..."
            TiledImage.run(self)
            print "done"
        print "all finished."


class Align(Experiment):
    """Class to align the spot to position of maximum brightness."""

    def __init__(self, microscope, config_file, **kwargs):
        super(Align, self).__init__()
        self.config_file = d.make_dict(config_file, **kwargs)
        # Valid kwargs are step_pair, backlash, focus.
        self.scope = microscope

    def run(self, func_list=None, save_mode='save_final'):
        """Algorithm for alignment is to iterate the Tiled procedure several
        times with decreasing width and increasing precision, and then using
        the parabola of brightness to try and find the maximum point by
        shifting slightly."""
        # Set mutable default values.
        if func_list is None:
            func_list = [h.bake(h.unchanged, args=['IMAGE_ARR'])]

        tiled_set = Tiled(self.scope, self.config_file)
        for step_pairs in self.config_file["n_steps"]:
            # Take measurements and move to position of maximum brightness.
            tiled_set.run(func_list=func_list, save_mode=save_mode,
                          step_pair=step_pairs)

        par = ParabolicMax(self.scope, self.config_file)

        for i in range(self.config_file["parabola_iterations"]):
            for ax in ['x', 'y']:
                par.run(func_list=func_list, save_mode=save_mode, axis=ax)
        image = _capture_image_from_microscope(self.scope, mode="fast_bayer")
        mod = proc.crop_array(image, mmts='pixel', dims=55)
        self.create_dataset('FINAL', data=mod)


class ParabolicMax(Experiment):
    """Takes a sequence of N measurements, fits a parabola to them and moves to
    the maximum brightness value. Make sure the microstep size is not too
    small, otherwise noise will affect the parabola shape."""

    def __init__(self, microscope, config_file, **kwargs):
        super(ParabolicMax, self).__init__()
        self.config_file = d.make_dict(config_file, **kwargs)
        self.scope = microscope

    def run(self, func_list=None, save_mode='save_final', axis='x',
            step_pair=None):
        """Operates on one axis at a time."""
        # Get default values.
        if step_pair is None:
            step_pair = (self.config_file["parabola_N"],
                         self.config_file["parabola_step"])

        # Set mutable default values.
        if func_list is None:
            func_list = [h.bake(h.unchanged, args=['IMAGE_ARR'])]

        end = h.bake(e.move_to_parmax, args=['IMAGE_ARR', self.scope, axis])
        _move_capture(self, {axis: [step_pair]},
                      image_mode='compressed', func_list=func_list,
                      save_mode=save_mode, end_func=end)


class DriftReCentre(Experiment):
    """Experiment to allow time for the spot to drift from its initial
    position for some time, then bring it back to the centre and measure
    the drift."""

    def __init__(self, microscope, config_file, **kwargs):
        super(DriftReCentre, self).__init__()
        self.config_file = d.make_dict(config_file, **kwargs)
        self.scope = microscope

    def run(self, func_list=None, save_mode='save_final', sleep_for=600):
        """Default is to sleep for 10 minutes."""
        # Do an initial alignment and then take that position as the initial
        # position.
        align = Align(self.scope, self.config_file)
        align.run(func_list=func_list, save_mode=save_mode)
        initial_pos = self.scope.stage.position

        t.sleep(sleep_for)

        # Measure the position after it has drifted, then bring back to centre.
        final_pos = self.scope.stage.position
        align.run(func_list=func_list, save_mode=save_mode)

        drift = final_pos - initial_pos

        # TODO Add logs to store the time and drift.


class KeepCentred(Experiment):
    """Iterate the parabolic method repeatedly after the initial alignment """

    def __init__(self, microscope, config_file, **kwargs):
        super(KeepCentred, self).__init__()
        self.config_file = d.make_dict(config_file, **kwargs)
        self.scope = microscope

    def run(self, func_list=None, save_mode='save_final'):
        pass


def centre_spot(scope_obj):
    """Find the spot on the screen, if it exists, and bring it to the
    centre. If no spot is found, an error is raised.
    :param scope_obj: A microscope object."""

    # TODO Need some way to identify if spot is not on screen.

    transform = scope_obj.calibrate()
    scope_obj.camera.preview()
    # TODO 'compressed' mode for speed, 'bayer' for accuracy.
    frame = scope_obj.camera.get_frame(mode='compressed', greyscale=True)

    # This is strongly affected by any other bright patches in the image -
    # need a better way to distinguish the bright spot.
    thresholded = cv2.threshold(frame, 180, 0, cv2.THRESH_TOZERO)[1]
    # gr = scope_obj.datafile.new_group('crop')
    cropped = proc.crop_array(thresholded, mmts='pixel', dims=np.array([
        300, 300]), centre=np.array([0, 0]))
    peak = sn.measurements.center_of_mass(thresholded)
    half_dimensions = np.array(np.array(h.get_size(frame)[:2]) / 2., dtype=int)

    # Note that the stage moves in x and y, so to calculate how much to move
    # by to centre the spot, we need half_dimensions - peak.
    thing = np.dot(half_dimensions - peak[::-1], transform)
    move_by = np.concatenate((thing, np.array([0])))
    scope_obj.stage.move_rel(move_by)
    # scope_obj.datafile.add_data(cropped, gr, 'cropped')
    return


def symmetric_range(n, increment):
    """Calculate a range with n points, spaced evenly about 0.

    The distance between each pair of adjacent points is `increment`.
    """
    return (np.arange(n) - (n - 1) / 2) * increment


def raster_scan(stage, dx=[0], dy=[0], dz=[0]):
    """Iterate over positions - returns a tuple of (index, pos) each time"""
    initial_pos = stage.position
    xs = np.array(dx) + initial_pos[0]
    ys = np.array(dy) + initial_pos[1]
    zs = np.array(dz) + initial_pos[2]
    try:
        for k, z in enumerate(zs):
            for j, y in enumerate(ys):
                for i, x in enumerate(xs):
                    stage.move_to_pos([x, y, z])
                    yield np.array([i, j, k]), np.array([x, y, z])
    except KeyboardInterrupt:
        print "Keyboard Interrupt: aborting scan."
        raise KeyboardInterrupt
    except Exception as e:
        print "An error occurred.  Moving back to initial position."
        raise e
    finally:
        stage.move_to_pos(initial_pos)


def _move_capture(exp_obj, iter_dict, image_mode, func_list=None,
                  save_mode='save_subset', end_func=None):
    """Function to carry out a sequence of measurements as per iter_list,
    take an image at each position, post-process it and return a final result.

    :param exp_obj: The experiment object.

    :param iter_dict: A dictionary of lists of 2-tuples to indicate all
    positions where images should be taken:
        {'x': [(n_x1, step_x1), ...], 'y': [(n_y1, step_y1), ...], 'z': ...],
    where each key indicates the axis to move, 'n' is the number of times
    to step (resulting in n+1 images) and 'step' is the number of microsteps
    between each subsequent image. So {'x': [(3, 100)]} would move only the
    x-axis of the microscope, taking 4 images at x=-150, x=-50, x=50,
    x=150 microsteps relative to the initial position. Note that:
    - Not all keys need to be specified.
    - All lists must be the same length.
    - All measurements will be taken symmetrically about the initial position.

    The position of each tuple in the list is important. If we have
    {'x': [(1, 100), (0, 100)],
     'y': [(2,  50), (3,  40)]},
    then tuples of the same index from each list will be combined into an
    array. This means that for the 0th index of the list, for x we have the
    positions [-50, 50] and [0] and for y [-50, 0, 50] and [-60, -20, 20, 60]
    respectively. [-50, 50] and [-50, 0, 50] will be combined to get the
    resulting array of [[-50, -50], [-50, 0], [-50, 50], [50, -50], [50, 0],
    [50, 50]], and the latter two to get [[0, -60], [0, -20], [0, 20],
    [0, 60]]. These are all the positions the stage will move to (the format
    here is [x, y]), iterating through each array in the order given.

    If you prefer to take images once for all the 'x' and, separately, once
    for all the 'y', run this function twice, once for 'x', once for 'y'.

    :param image_mode: The camera's capture mode: 'bayer', 'bgr' or
    'compressed'. Greyscale is off by default.

    :param func_list: The post-processing curried function list, created using
    the bake function in the helpers.py module.

    :param save_mode: How to save at the end of each iteration.
    - 'save_each': Every single measurement is saved: {'x': [(3, 100)]}
      would result in 4 post-processed results being saved, which is useful if
      the post-processed results are image arrays.
    - 'save_final': Every single measurement is made before the entire set of
      results, as an array, is saved along with their positions, in the
      format [[x-column], [y-column], [z-column], [measurements-column]]. This
      is good for the post-processed results that are single numerical value.
    - 'save_subset': Each array is measured before being saved (for example, in
      the description of iter_dict, there are two arrays being iterated
      through).
    - None: Data is not saved at all, but is returned. Might be useful if
      this is intermediate step.

    :param end_func: A curried function, which is executed on the array of
    final results. This can be useful to move to a position where the final
    measurement is maximised. Note: if save_mode is 'save_each', the results
    array will be empty so end_func must be None."""

    # Verify iter_dict format:
    valid_keys = ['x', 'y', 'z']
    len_lists = []
    assert len(iter_dict) <= len(valid_keys)
    for key in iter_dict.keys():
        print iter_dict
        print key
        for tup in iter_dict[key]:
            print tup
            assert len(tup) == 2, 'Invalid tuple format.'
        assert np.any(key == np.array(valid_keys)), 'Invalid key.'
        # For robustness, the lengths of the lists for each key must be
        # the same length.
        len_lists.append(len(iter_dict[key]))
    if len(len_lists) > 1:
        assert [len_lists[0] == element for element in len_lists[1:]], \
            'Lists of unequal lengths.'

    # Get initial position, which may not be [0, 0, 0] if scope object
    # has been used for something else prior to this experiment.
    initial_position = exp_obj.scope.stage.position

    # A set of results to be collected if save_mode == 'save_final'.
    results = []

    for i in range(len_lists[0]):
        # For the length of each list, combine every group of tuples in the
        # same position to get array of positions to move to.
        move_by = {}
        for key in valid_keys:
            try:
                (n, steps) = iter_dict[key][i]
                move_by[key] = np.linspace(-n / 2. * steps, n / 2. * steps,
                                           n + 1)
            except KeyError:
                # If key does not exist, then keep this axis fixed.
                move_by[key] = np.array([0])
        print "move-by", move_by
        # Generate array of positions to move to.
        pos = h.positions_maker(x=move_by['x'], y=move_by['y'],
                                z=move_by['z'], initial_pos=initial_position)

        try:
            # For each position in the range specified, take an image, apply
            # all the functions in func_list on it, then either save the
            # measurement if save_mode = 'save_final', or append the
            # calculation to a results file and save it all at the end.
            while True:
                next_pos = next(pos)  # This returns StopIteration at end.
                print next_pos
                exp_obj.scope.stage.move_to_pos(next_pos)

                image = _capture_image_from_microscope(exp_obj.scope,
                                                       mode=image_mode)
                modified = image

                # Post-process.
                for function in func_list:
                    modified = function(modified)
                    exp_obj.scope.gr.create_dataset('modd', data=modified)

                # Save this array in HDF5 file.
                if save_mode == 'save_each':
                    exp_obj.scope.gr.create_dataset('modified_image', attrs={
                        'Position': exp_obj.scope.stage.position,
                        'Cropped size': 300}, data=modified)
                else:
                    # The curried function and 'save_final' both use the
                    # array of final results.
                    results.append([exp_obj.scope.stage.position[0],
                                    exp_obj.scope.stage.position[1],
                                    exp_obj.scope.stage.position[2], modified])

        except StopIteration:
            # Iterations finished - save the subset of results and take the
            # next set.
            if save_mode == 'save_subset':
                results = np.array(results, dtype=np.float)
                exp_obj.create_dataset('brightness_results', data=results)
                exp_obj.log("Test - brightness results added.")

        except KeyboardInterrupt:
            print "Aborted, moving back to initial position."
            exp_obj.scope.stage.move_to_pos(initial_position)
            exp_obj.wait_or_stop(10)

    results = np.array(results, dtype=np.float)
    if save_mode == 'save_final':
        exp_obj.create_dataset('brightness_results', data=results)
        exp_obj.log("Test - brightness results added.")
    elif save_mode is None:
        return results
    elif save_mode != 'save_each' and save_mode != 'save_subset':
        raise ValueError('Invalid save mode.')

    if end_func is not None:
        if save_mode != 'save_each':
            # Process the result and return it.
            try:
                return end_func(results)
            except:
                raise NameError('Invalid function name.')
        elif save_mode == 'save_each':
            raise ValueError('end_func must be None if save_mode = '
                             '\'save_each\', because the results array is '
                             'empty.')


def _save_image_to_datagroup(data_group, image, position):
    compressed_image = cv2.imencode(".jpg", image)[1]
    ds = data_group.create_dataset("image_%d",
                                   data=compressed_image)
    ds.attrs['position'] = position
    ds.attrs['compressed_image_format'] = 'JPEG'


def _capture_image_from_microscope(scope, greyscale=False, mode="compressed"):
    return scope.camera.get_frame(greyscale, mode)
