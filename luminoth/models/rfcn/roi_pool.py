import tensorflow as tf
import sonnet as snt

from luminoth.utils.bbox_transform_tf import (
    bboxes_to_relative_coord, change_order
)


class RoIPool(snt.AbstractModule):
    def __init__(self, config, im_shape, score_bank_size, num_classes,
                 name="roi_pool"):
        super(RoIPool, self).__init__(name=name)
        self._im_shape = im_shape
        self._config = config
        self._score_bank_size = score_bank_size
        self._num_classes = num_classes

    def _build(self, proposals, score_bank):
        relative_proposed_bboxes = bboxes_to_relative_coord(
            proposals[:, 1:], self._im_shape)
        # Now we want to extract the portion of the score_bank that corresponds
        # to each patch.
        scores_for_patches = self._scores_for_patches(
            relative_proposed_bboxes, score_bank
        )
        flattened_scores = tf.reshape(
            scores_for_patches,
            (
                tf.shape(proposals)[0],
                -1
            ), name="flatten_scores"
        )
        pooled_scores = self._get_pooled_scores(flattened_scores)
        # Reshape to the shaped required by the Voter.
        pooled_scores_per_class = tf.reshape(
            pooled_scores,
            [tf.shape(proposals)[0], self._num_classes + 1, -1]
        )
        return {
            'pooled_scores_per_class': pooled_scores_per_class
        }

    def _scores_for_patches(self, relative_proposed_bboxes, score_bank):
        tf_style_bboxes = change_order(relative_proposed_bboxes)
        # Get a fake batch number.
        batch_ids = tf.zeros([tf.shape(tf_style_bboxes)[0]], dtype=tf.int32)

        # TODO: bilinear interpolation may not be the best pooling mechanism.
        # Rewrite to use avg or max pooling.
        return tf.image.crop_and_resize(
            score_bank, tf_style_bboxes, batch_ids,
            [self._score_bank_size, self._score_bank_size], name="crops"
        )

    def _get_pooled_scores(self, flattened_scores):
        k = self._score_bank_size
        num_classes = self._num_classes

        # This list will hold the flat pooled scores so that the i-th score
        # for each class comes from the i-th position of the i-th channel for
        # that class. This pooling makes it so that each channel gets trained
        # to identify labels in a specific relative portion of the RoI.
        pooled_scores = []

        max_base = k ** 2 * (k ** 2 * (num_classes + 1))
        base_step = (k ** 2 * (num_classes + 1))
        for base in range(0, max_base, base_step):
            for ind in range(0, base_step, k ** 2 + 1):
                pooled_scores.append(flattened_scores[:, ind + base])
        return tf.stack(pooled_scores, name="stack_scores")
