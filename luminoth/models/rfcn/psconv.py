import tensorflow as tf
import sonnet as snt

from sonnet.python.modules.conv import Conv2D

from luminoth.utils.vars import (
    get_initializer, get_activation_function
)


class PSConv(snt.AbstractModule):
    """Position Sensitive Convoltional Neural Network classifier for R-FCN.
    """
    def __init__(self, num_classes, config, debug=False, seed=None,
                 name='ps_conv'):
        super(PSConv, self).__init__(name=name)
        self._num_classes = num_classes

        # This is what is referred to across the paper as 'k'.
        self._score_bank_size = config.score_bank_size
        self._kernel_size = config.kernel_size

        self._activation = get_activation_function(config.activation_function)

        self._initializer = get_initializer(config.initializer, seed=seed)
        self._regularizer = tf.contrib.layers.l2_regularizer(
            scale=config.l2_regularization_scale)

        self._debug = debug
        self._config = config
        self._seed = seed

    def _build(self, conv_feature_map, proposals, im_shape,
               gt_boxes=None):
        """Classifies proposals by getting position-sensitive class scores.

        Args:
            conv_feature_map: Output of the pretrained base network.
            proposals: proposed bboxes for object detection. Shape is
                (num_proposals, 5). First is batch number.
            im_shape: Shape of the image.
            gt_boxes: Shape is (num_gt_boxes, 5). Last is label.

        Returns:
            score_bank: 3D tensor (with k=score_bank_size)
                height: k
                width: k
                depth: k**2 * (num_classes + 1)
        """
        channels = (self._score_bank_size ** 2) * (self._num_classes + 1)
        self.ps_convolution = Conv2D(
            output_channels=channels,
            kernel_shape=[self._kernel_size, self._kernel_size],
            initializers={'w': self._initializer},
            regularizers={'w': self._regularizer},
            padding='VALID', name='ps_convolution'
        )
        score_bank = self.ps_convolution(conv_feature_map)

        return {
            'score_bank': score_bank
        }
