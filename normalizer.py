import torch
from enum import Enum

class Mode(Enum):
    MIN_MAX = 'min_max'
    MAX_ABS = 'max_abs'


class Normalizer:
    def __init__(self):
        self.parameters = {}

    def normalize(self, name, tensor, mode):

        if mode == Mode.MIN_MAX:

            minimum = tensor.min(dim=0).values
            maximum = tensor.max(dim=0).values

            span = (maximum - minimum).clamp_min(1e-6)

            self.parameters[name] = {
                'mode': mode.value,
                'min': minimum,
                'span': span
            }
        elif mode == Mode.MAX_ABS:

            scale = tensor.abs().max(dim=0).values.clamp_min(1e-6)

            self.parameters[name] = {
                'mode': mode.value,
                'scale': scale
            }
        else:
            raise ValueError(f'Unsupported normalization mode: {mode}')

        return self.transform(name, tensor)

    def transform(self, name, tensor):

        parameters = self._get_parameters(name)
        mode = Mode(parameters['mode'])

        #           x - xmin
        # x = 2 * -------------   - 1
        #           xmax - xmin
        if mode == Mode.MIN_MAX:
            return 2.0 * (tensor - parameters['min']) / parameters['span'] - 1.0

        #        x
        # x = --------
        #      max|x|
        if mode == Mode.MAX_ABS:
            return tensor / parameters['scale']

        raise ValueError(f'Unsupported normalization mode: {mode}')

    def denormalize(self, name, tensor):
        parameters = self._get_parameters(name)
        mode = Mode(parameters['mode'])

        if mode == Mode.MIN_MAX:
            return (tensor + 1.0) * parameters['span'] / 2.0 + parameters['min']
        if mode == Mode.MAX_ABS:
            return tensor * parameters['scale']

        raise ValueError(f'Unsupported normalization mode: {mode}')

    def _get_parameters(self, name):
        if name not in self.parameters:
            raise KeyError(f'Normalization parameters for {name!r} not found')
        return self.parameters[name]