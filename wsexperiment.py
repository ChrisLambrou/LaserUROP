import cv2
import data_io as d
import measurements as mmts
import numpy as np
from nplab.experiment.experiment import Experiment


class WSExperiment(Experiment):
    def __init__(self, microscope, config_file, **kwargs):
        super(WSExperiment, self).__init__()
        self.config_file = d.make_dict(config_file, **kwargs)
        self.scope = microscope
        self.scope.camera.preview()
        self._compensation_factors = {}

    def capture_image(self, mode="compressed", compensate=True, crop=False):
        image = self.scope.camera.get_frame(greyscale=False, mode=mode)
        if crop:
            image = self._crop_image(image)
        if compensate:
            compensation_factor = self._get_compensation_factor(mode)
            if crop:
                compensation_factor = self._crop_image(compensation_factor)
            image = (image * compensation_factor).astype(np.int).clip(0, 255)
        return image

    def _crop_image(self, image):
        shape = image.shape
        width = shape[0]
        height = shape[1]
        x1 = width / 3
        x2 = 2 * width / 3
        y1 = height / 3
        y2 = 2 * height / 3
        return image[x1:x2,y1:y2,:]

    def _get_compensation_factor(self, mode):
        # Lazy-load the compensation factor image.
        compensation_factor = self._compensation_factors.get(mode)
        if compensation_factor is None:
            images_dir = self.config_file["compensation_images_path"]
            image_path = "%s/CompensationImage_%s.png" % (images_dir, mode)
            compensation_image = cv2.imread(image_path)
            compensation_factor = 224.0 / (compensation_image + 1)  # Avoid divide by zero issues.
            self._compensation_factors[mode] = compensation_factor
        return compensation_factor

    def save_image_to_datagroup(self, image, data_group, position=None):
        compressed_image = cv2.imencode(".jpg", image)[1]
        dataset = data_group.create_dataset("image_%d", data=compressed_image)
        dataset.attrs['position'] = position or self.scope.stage.position
        dataset.attrs['compressed_image_format'] = 'JPEG'

    def autofocus(self, dz, data_group=None):
        """Take pictures at a range of z positions and move to the sharpest."""
        sharpnesses = []
        positions = []

        for index, pos in self.raster_scan(dz=dz):
            image = self.capture_image(crop=True)
            if data_group is not None:
                self.save_image_to_datagroup(image, data_group)
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
        self.scope.stage.move_to_pos(best_position)

    def raster_scan(self, dx=[None], dy=[None], dz=[None]):
        """Iterate over positions - returns a tuple of (index, pos) each time"""
        stage = self.scope.stage
        origin = stage.position

        try:
            for k, z in enumerate(dz):
                for j, y in enumerate(dy):
                    for i, x in enumerate(dx):
                        new_x = stage.position[0] if x is None else (origin[0] + x)
                        new_y = stage.position[1] if y is None else (origin[1] + y)
                        new_z = stage.position[2] if z is None else (origin[2] + z)
                        new_pos = [new_x, new_y, new_z]
                        stage.move_to_pos(new_pos)
                        yield np.array([i, j, k]), np.array(new_pos)
        except KeyboardInterrupt:
            print "Keyboard Interrupt: aborting scan."
            raise KeyboardInterrupt
        except Exception as e:
            print "An error occurred.  Moving back to initial position."
            raise e
        finally:
            stage.move_to_pos(origin)


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
