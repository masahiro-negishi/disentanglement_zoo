import os

import h5py
from full_data import GroundTruthDataset


class Shapes3D(GroundTruthDataset):
    """3D Shapes dataset.

    Reference: https://github.com/deepmind/3d-shapes/tree/master

    The ground-truth factors of variation are:
    0 - floor color (10 different values linearly spaced in [0, 1])
    1 - wall color (10 different values linearly spaced in [0, 1])
    2 - object color (10 different values linearly spaced in [0, 1])
    3 - object size (8 different values linearly spaced in [0, 1])
    4 - object shape (4 different values in [0, 1, 2, 3]])
    5 - object orientation (15 different values linearly spaced in [-30, 30])

    Note: all the combinations of factors are present in the data set.
    """

    def __init__(self):
        source_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'source/3dshapes.h5')
        self.h5pydata = h5py.File(source_path, 'r')
        self.images = self.h5pydata['images'] # (480000, 64, 64, 3)
        self.labels = self.h5pydata['labels'] # (480000, 6)

    def __len__(self):
        return 480000

    @property
    def num_factors(self):
        return 6

    @property
    def factors_num_values(self):
        return [10, 10, 10, 8, 4, 15]

    @property
    def observation_shape(self):
        return [64, 64, 3]

    def close(self):
        self.h5pydata.close()
