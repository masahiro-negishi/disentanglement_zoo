class GroundTruthDataset:
    """Class of datasets"""

    def __len__(self):
        raise NotImplementedError()

    @property
    def num_factors(self):
        raise NotImplementedError()

    @property
    def factors_num_values(self):
        raise NotImplementedError()

    @property
    def observation_shape(self):
        raise NotImplementedError()
