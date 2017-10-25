import tensorflow as tf
import sonnet as snt

from luminoth.models.base.fpn import FPN
from luminoth.models.retina.class_subnet import ClassSubnet
from luminoth.models.retina.box_subnet import BoxSubnet
from luminoth.utils.losses import smooth_l1_loss, focal_loss
from luminoth.utils.vars import get_saver


class Retina(snt.AbstractModule):
    def __init__(self, config, name='retina'):
        super(Retina, self).__init__(name=name)

        self._config = config
        self._num_classes = config.model.network.num_classes
        self._num_anchors = config.model.network.num_anchors

        self._debug = config.train.debug
        self._seed = config.train.seed

        self.fpn = FPN(
            config.model.fpn
        )

        self._losses_collections = ['retina_losses']

        self._gamma = self.model.loss.gamma
        self._alpha_balance = self.model.loss.alpha_balance

    def _build(self, image, gt_boxes=None, is_training=True):
        if gt_boxes is not None:
            gt_boxes = tf.cast(gt_boxes, tf.float32)

        fpn_levels = self.fpn(image, is_training=is_training)

        box_subnet = BoxSubnet(
            self._config.model.box_subnet, num_anchors=self._num_anchors
        )
        class_subnet = ClassSubnet(
            self._config.model.class_subnet, num_anchors=self._num_anchors,
            num_classes=self._num_classes
        )

        box_preds = []
        class_probs = []
        for level in fpn_levels:
            anchors = self._generate_anchors(tf.shape(level))
            level_box_pred = box_subnet(level, anchors)
            level_class_pred = class_subnet(level, anchors)
            level_class_probs = level_class_pred['cls_probs']

            box_preds.append(level_box_pred)
            class_probs.append(level_class_probs)

        pred_dict = {
            'cls_probs': tf.stack(class_probs),
            'bbox_pred': tf.stack(box_preds),
        }
        return pred_dict

    def loss(self, pred_dict):
        """Compute training loss for object detection with Retina.

        We use focal loss for classification, and smooth L1 loss for bbox
        regression.
        """
        with tf.name_scope('losses'):
            cls_probs = pred_dict['cls_probs']
            cls_target = pred_dict['cls_target']

            bbox_pred = pred_dict['bbox_pred']
            bbox_target = pred_dict['bbox_target']

            cls_loss = focal_loss(
                cls_probs, cls_target, gamma=self._gamma,
                alpha_balance=self._alpha_balance
            )
            reg_loss = smooth_l1_loss(
                bbox_pred, bbox_target
            )

            tf.losses.add_loss(cls_loss)
            tf.losses.add_loss(reg_loss)

            total_loss = tf.losses.get_total_loss()
            return total_loss

    def _generate_anchors(self, level_shape):
        return

    @property
    def summary(self):
        pass

    def get_trainable_vars(self):
        trainable_vars = snt.get_variables_in_module(self)
        trainable_vars += self.fpn.get_trainable_vars(
            train_base=self._config.model.fpn.train_base
        )
        return trainable_vars

    def get_saver(self):
        return get_saver((self, self.fpn))

    def load_pretrained_weights(self):
        return self.fpn.load_weights()
