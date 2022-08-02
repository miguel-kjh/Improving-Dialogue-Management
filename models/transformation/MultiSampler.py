import math

import torch as th

from models.transformation.samplers import Sampler
from utils.UtilityForPolicies import th_random_choice


class MultiSampler(Sampler):
    """Samples elements more than once in a single pass through the data.
    This allows the number of samples per epoch to be larger than the number
    of samples itself, which can be useful when training on 2D slices taken
    from 3D images, for instance.
    """

    def __init__(self, nb_samples, desired_samples, shuffle=False):
        """Initialize MultiSampler
        Arguments
        ---------
        nb_samples : the numbers of samples in the dataset

        desired_samples : number of samples per batch you want
            whatever the difference is between an even division will
            be randomly selected from the samples.
            e.g. if len(data_source) = 3 and desired_samples = 4, then
            all 3 samples will be included and the last sample will be
            randomly chosen from the 3 original samples.
        shuffle : boolean
            whether to shuffle the indices or not
        """
        super().__init__()
        self.sample_idx_array = None
        self.data_samples = nb_samples
        self.desired_samples = desired_samples
        self.shuffle = shuffle

    def gen_sample_array(self):
        n_repeats = self.desired_samples / self.data_samples
        cat_list = []
        for i in range(math.floor(n_repeats)):
            cat_list.append(th.arange(0, self.data_samples))
        # add the left over samples
        left_over = self.desired_samples % self.data_samples
        if left_over > 0:
            cat_list.append(th_random_choice(self.data_samples, left_over))
        self.sample_idx_array = th.cat(cat_list).long()
        return self.sample_idx_array

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return self.desired_samples