import sonnet as snt

from luminoth.utils.vars import get_activation_function


class Subnet(snt.AbstractModule):
    def __init__(self, config, num_final_channels, prefix=None,
                 name='subnet'):
        super(Subnet, self).__init__(name=name)
        if prefix is None:
            prefix = name
        self._config = config
        self._kernel_shape = config.kernel_shape
        self._num_final_channels = num_final_channels

        self._prefix = prefix

        self._hidden_activation = get_activation_function(
            config.hidden.activation
        )
        self._final_activation = get_activation_function(
            config.final.activation
        )

    def _build(self, fpn_level, anchors):
        layers = []
        for i in range(self._config.hidden.net_depth):
            new_layer = snt.Conv2D(
                output_channels=self._config.hidden.channels,
                kernel_shape=self._config.hidden.kernel_shape,
                name='{}_hidden_{}'.format(self._prefix, i)
            )
            layers.append(new_layer)

        pred = fpn_level
        for layer in layers:
            pred = self._hidden_activation(layer(pred))

        final_layer = snt.Conv2D(
            output_channels=self._num_final_channels,
            kernel_shape=self._config.final.kernel_shape,
            name='{}_final'
        )
        return self._final_activation(final_layer(pred))
