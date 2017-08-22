import numpy as np  # noqa
import sonnet as snt
import tensorflow as tf  # noqa

from luminoth.models import get_model


class SSD(snt.AbstractModule):
    def __init__(self, config, debug=False, num_classes=None, name='ssd'):
        super(SSD, self).__init__(name=name)
        self._num_classes = num_classes
        self._config = config
        self._debug = debug

        self.pretrained = get_model(config.pretrained.net)(
            config.pretrained, parent_name=self.module_name
        )

    def _instantiate_layers(self):
        self._classifier

    def _build(self):
        pass
