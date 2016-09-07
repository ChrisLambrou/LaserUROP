
import data_io as d
from nplab.experiment.experiment import Experiment


class WSExperiment(Experiment):

    def __init__(self, microscope, config_file, **kwargs):
        super(WSExperiment, self).__init__()
        self.config_file = d.make_dict(config_file, **kwargs)
        self.scope = microscope
        self.scope.camera.preview()
