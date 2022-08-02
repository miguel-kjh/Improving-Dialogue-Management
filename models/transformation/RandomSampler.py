import torch as th

from models.transformation.samplers import Sampler


class RandomSampler(Sampler):
    """
    Samples elements randomly, without replacement.
    """

    def __init__(self, nb_samples):
        super().__init__()
        self.num_samples = nb_samples

    def __iter__(self):
        return iter(th.randperm(self.num_samples).long())

    def __len__(self):
        return self.num_samples