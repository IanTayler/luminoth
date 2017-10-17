import numpy as np
import sonnet as snt
import tensorflow as tf

from luminoth.models.fasterrcnn.rpn import RPN
from luminoth.models.rfcn.psconv import PSConv
from luminoth.models.rfcn.roi_pool import RoIPool
from luminoth.models.rfcn.voter import Voter
from luminoth.models.rfcn.voter_target import VoterTarget
from luminoth.models.base import TruncatedBaseNetwork
from luminoth.utils.anchors import generate_anchors_reference
from luminoth.utils.config import get_base_config
from luminoth.utils.vars import variable_summaries, get_saver, layer_summaries


class RFCN(snt.AbstractModule):
    """R-FCN Network module

    Follows Jifeng Dai, Yi Li, et al. (2016)'s paper 'R-FCN: Object Detection
    via Region-based Fully Convolutional Networks'.
    """

    base_config = get_base_config(__file__)

    def __init__(self, config, name='rfcn'):
        super(RFCN, self).__init__(name=name)

        self._config = config

        self._num_classes = config.model.network.num_classes

        self._debug = config.train.debug
        self._seed = config.train.seed

        self._anchor_base_size = config.model.anchors.base_size
        self._anchor_scales = np.array(config.model.anchors.scales)
        self._anchor_ratios = np.array(config.model.anchors.ratios)
        self._anchor_stride = config.model.anchors.stride

        self._anchor_reference = generate_anchors_reference(
            self._anchor_base_size, self._anchor_ratios, self._anchor_scales
        )

        self._num_anchors = self._anchor_reference.shape[0]

        self._rpn_cls_loss_weight = config.model.loss.rpn_cls_loss_weight
        self._rpn_reg_loss_weight = config.model.loss.rpn_reg_loss_weight
        self._classif_loss_weight = config.model.loss.classif_loss_weight

        self._losses_collections = ['rfcn_losses']

        self.base_network = TruncatedBaseNetwork(
            config.model.base_network, parent_name=self.module_name
        )

        self._score_bank_size = config.model.ps_conv.score_bank_size

    def _build(self, image, gt_boxes=None, is_training=True):
        if gt_boxes is not None:
            gt_boxes = tf.cast(gt_boxes, tf.float32)

        conv_feature_map = self.base_network(image, is_training=is_training)

        image_shape = tf.shape(image)[1:3]

        variable_summaries(
            conv_feature_map, 'conv_feature_map', ['rpn'])

        # TODO: we're using fasterrcnn's RPN which isn't fully convolutional.
        # Consider rewriting the RPN or making a more abstract version shared
        # by both models.
        self._rpn = RPN(
            self._num_anchors, self._config.model.rpn,
            debug=self._debug, seed=self._seed
        )
        self._ps_conv = PSConv(
            self._num_classes, self._config.model.ps_conv,
            debug=self._debug, seed=self._seed
        )
        self._roi_pool = RoIPool(
            self._config.model.roi_pool, image_shape, self._score_bank_size,
            self._num_classes
        )
        self._voter = Voter(
            self._config.model.voter, self._num_classes, self._score_bank_size
        )
        self._voter_target = VoterTarget(
            self._config.model.voter.target, self._num_classes,
            seed=self._seed, debug=self._debug
        )

        all_anchors = self._generate_anchors(tf.shape(conv_feature_map))
        rpn_prediction = self._rpn(
            conv_feature_map, image_shape, all_anchors,
            gt_boxes=gt_boxes
        )
        ps_conv_prediction = self._ps_conv(
            conv_feature_map, rpn_prediction['proposals'],
            image_shape, gt_boxes=gt_boxes
        )
        roi_pool_dict = self._roi_pool(
            rpn_prediction['proposals'], ps_conv_prediction['score_bank']
        )
        cls_score = self._voter(roi_pool_dict['pooled_scores_per_class'])
        cls_prob = tf.nn.softmax(cls_score)

        cls_target = self._voter_target(
            rpn_prediction['proposals'], gt_boxes=gt_boxes)

        variable_summaries(cls_prob, 'cls_prob', ['voter'])
        variable_summaries(
            rpn_prediction['proposals'], 'bbox_offsets', ['rpn']
        )

        layer_summaries(self._ps_conv, ['ps_conv'])

        prediction_dict = {
            'rpn_prediction': rpn_prediction,
            'objects': rpn_prediction['proposals'],
            'cls_score': cls_score,
            'cls_prob': cls_prob,
            'cls_target': cls_target
        }
        return prediction_dict

    def loss(self, prediction_dict, return_all=False):
        """Compute the joint training loss for R-FCN.

        Args:
            prediction_dict: The output dictionary of the _build method from
                which we use two different main keys:

                rpn_prediction: A dictionary with the output Tensors from the
                    RPN.
                classif_prediction: A dictionary with the output Tensors
                    at the end of the network.

        Returns:
            If `return_all` is False, a tensor for the total loss. If True, a
            dict with all the internal losses (RPN's, the end loss,
            regularization and total loss).
        """

        with tf.name_scope('losses'):
            rpn_loss_dict = self._rpn.loss(
                prediction_dict['rpn_prediction']
            )

            # Losses have a weight assigned, we multiply by them before saving
            # them.
            rpn_loss_dict['rpn_cls_loss'] = (
                rpn_loss_dict['rpn_cls_loss'] * self._rpn_cls_loss_weight)
            rpn_loss_dict['rpn_reg_loss'] = (
                rpn_loss_dict['rpn_reg_loss'] * self._rpn_reg_loss_weight)

            prediction_dict['rpn_loss_dict'] = rpn_loss_dict

            classif_loss_dict = self._voter.loss(
                prediction_dict
            )

            classif_loss_dict['classif_loss'] = (
                classif_loss_dict['classif_loss'] *
                self._classif_loss_weight
            )

            prediction_dict['classif_loss_dict'] = classif_loss_dict

            all_losses_items = (
                list(rpn_loss_dict.items()) + list(classif_loss_dict.items()))

            for loss_name, loss_tensor in all_losses_items:
                tf.summary.scalar(
                    loss_name, loss_tensor,
                    collections=self._losses_collections
                )
                # We add losses to the losses collection instead of manually
                # summing them just in case somebody wants to use it in another
                # place.
                tf.losses.add_loss(loss_tensor)

            # Regularization loss is automatically saved by TensorFlow, we log
            # it differently so we can visualize it independently.
            regularization_loss = tf.losses.get_regularization_loss()
            # Total loss without regularization
            no_reg_loss = tf.losses.get_total_loss(
                add_regularization_losses=False
            )
            total_loss = tf.losses.get_total_loss()

            tf.summary.scalar(
                'total_loss', total_loss,
                collections=self._losses_collections
            )
            tf.summary.scalar(
                'no_reg_loss', no_reg_loss,
                collections=self._losses_collections
            )
            tf.summary.scalar(
                'regularization_loss', regularization_loss,
                collections=self._losses_collections
            )

            if return_all:
                loss_dict = {
                    'total_loss': total_loss,
                    'no_reg_loss': no_reg_loss,
                    'regularization_loss': regularization_loss,
                }

                for loss_name, loss_tensor in all_losses_items:
                    loss_dict[loss_name] = loss_tensor

                return loss_dict

            # We return the total loss, which includes:
            # - rpn loss
            # - rcnn loss (if activated)
            # - regularization loss
            return total_loss

    def _generate_anchors(self, feature_map_shape):
        """Generate anchor for an image.

        Using the feature map, the output of the pretrained network for an
        image, and the anchor_reference generated using the anchor config
        values. We generate a list of anchors.

        Anchors are just fixed bounding boxes of different ratios and sizes
        that are uniformly generated throught the image.

        Args:
            feature_map_shape: Shape of the convolutional feature map used as
                input for the RPN. Should be (batch, height, width, depth).

        Returns:
            all_anchors: A flattened Tensor with all the anchors of shape
                `(num_anchors_per_points * feature_width * feature_height, 4)`
                using the (x1, y1, x2, y2) convention.
        """
        with tf.variable_scope('generate_anchors'):
            grid_width = feature_map_shape[2]  # width
            grid_height = feature_map_shape[1]  # height
            shift_x = tf.range(grid_width) * self._anchor_stride
            shift_y = tf.range(grid_height) * self._anchor_stride
            shift_x, shift_y = tf.meshgrid(shift_x, shift_y)

            shift_x = tf.reshape(shift_x, [-1])
            shift_y = tf.reshape(shift_y, [-1])

            shifts = tf.stack(
                [shift_x, shift_y, shift_x, shift_y],
                axis=0
            )

            shifts = tf.transpose(shifts)
            # Shifts now is a (H x W, 4) Tensor

            # Expand dims to use broadcasting sum.
            all_anchors = (
                np.expand_dims(self._anchor_reference, axis=0) +
                tf.expand_dims(shifts, axis=1)
            )

            # Flatten
            all_anchors = tf.reshape(
                all_anchors, (-1, 4)
            )
            return all_anchors

    @property
    def summary(self):
        """
        Generate merged summary of all the sub-summaries used inside the
        R-FCN network.
        """
        summaries = [
            tf.summary.merge_all(key='rpn'),
        ]

        summaries.append(
            tf.summary.merge_all(key=self._losses_collections[0])
        )

        # if self._with_rcnn:
        #     summaries.append(tf.summary.merge_all(key='rcnn'))

        return tf.summary.merge(summaries)

    def get_trainable_vars(self):
        """Get trainable vars included in the module.
        """
        trainable_vars = snt.get_variables_in_module(self)
        if self._config.model.base_network.trainable:
            pretrained_trainable_vars = self.base_network.get_trainable_vars()
            tf.logging.info('Training {} vars from pretrained module.'.format(
                len(pretrained_trainable_vars)))
            trainable_vars += pretrained_trainable_vars
        else:
            tf.logging.info('Not training variables from pretrained module')

        return trainable_vars

    def get_saver(self, ignore_scope=None):
        """Get an instance of tf.train.Saver for all modules and submodules.
        """
        return get_saver((self, self.base_network), ignore_scope=ignore_scope)

    def load_pretrained_weights(self):
        """Get operation to load pretrained weights from file.
        """
        return self.base_network.load_weights()
