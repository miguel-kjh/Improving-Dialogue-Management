from models.transformation.samplers import Sampler


class SequentialSampler(Sampler):
    """
    Samples elements sequentially, always in the same order.
    """

    def __init__(self, nb_samples):
        super().__init__()
        self.num_samples = nb_samples

    def __iter__(self):
        return iter(range(self.num_samples))

    def __len__(self):
        return self.num_samples