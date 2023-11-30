class BaseMetric:
    def __init__(self, name=None,
                 calc_on_train: bool = True,
                 calc_on_non_train: bool = True,
                 *args, **kwargs):
        self.name = name if name is not None else type(self).__name__
        self.calc_on_train = calc_on_train
        self.calc_on_non_train = calc_on_non_train

    def __call__(self, **batch):
        raise NotImplementedError()
