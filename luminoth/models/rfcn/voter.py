import tensorflow as tf
import sonnet as snt


class Voter(snt.AbstractModule):
    def __init__(self, config, num_classes, score_bank_size, debug=False,
                 seed=None, name="voter"):
        super(Voter, self).__init__(name=name)
        self._num_classes = num_classes
        self._score_bank_size = score_bank_size
        self._config = config

        self._debug = debug
        self._seed = seed

    def _build(self, ps_scores_pooled):
        """Aggregate scores from the pooled position-sensitive score banks.

        Args:
            ps_scores_pooled: 3D Tensor with shape
                (num_proposals, num_classes + 1, k**2)

        Returns:
            cls_scores: Shaped (num_proposals, num_classes + 1)
        """
        # As is the case in the paper, we're simply averaging the scores.
        cls_scores = tf.reduce_mean(
            ps_scores_pooled,
            axis=2, name="vote"
        )

        return cls_scores

    def loss(self, prediction_dict):
        """
        Returns cost for R-FCN's classification based on:

        Args:
            prediction_dict with keys:
                rcnn:
                    cls_score: shape (num_proposals, num_classes + 1)
                        Has the class scoring for each the proposals. Classes
                        are 1-indexed with 0 being the background.

                    cls_prob: shape (num_proposals, num_classes + 1)
                        Application of softmax on cls_score.

                    cls_target: shape (num_proposals,)
                        Has the correct label for each of the proposals.
                        0 => background
                        1..n => 1-indexed classes
                target:
                    bbox_offsets: shape (num_proposals, num_classes * 4)
                        Has the offset for each proposal for each class.
                        We have to compare only the proposals labeled with the
                        offsets for that label.

                    bbox_offsets_target: shape (num_proposals, 4)
                        Has the true offset of each proposal for the true
                        label.
                        In case of not having a true label (non-background)
                        then it's just zeroes.

        Returns:
            loss_dict with keys:
                rcnn_cls_loss: The cross-entropy or log-loss of the
                    classification tasks between then num_classes + background.
                rcnn_reg_loss: The smooth L1 loss for the bounding box
                    regression task to adjust correctly labeled boxes.

        """
        with tf.name_scope('RCNNLoss'):
            cls_score = prediction_dict['cls_score']
            # cls_prob = prediction_dict['rcnn']['cls_prob']
            # Cast target explicitly as int32.
            cls_target = tf.cast(
                prediction_dict['cls_target'], tf.int32
            )

            # First we need to calculate the log loss betweetn cls_prob and
            # cls_target

            # We only care for the targets that are >= 0
            not_ignored = tf.reshape(tf.greater_equal(
                cls_target, 0), [-1], name='not_ignored')
            # We apply boolean mask to score, prob and target.
            cls_score_labeled = tf.boolean_mask(
                cls_score, not_ignored, name='cls_score_labeled')
            # cls_prob_labeled = tf.boolean_mask(
            #    cls_prob, not_ignored, name='cls_prob_labeled')
            cls_target_labeled = tf.boolean_mask(
                cls_target, not_ignored, name='cls_target_labeled')

            tf.summary.scalar(
                'batch_size',
                tf.shape(cls_score_labeled)[0], ['rcnn']
            )

            # Transform to one-hot vector
            cls_target_one_hot = tf.one_hot(
                cls_target_labeled, depth=self._num_classes + 1,
                name='cls_target_one_hot'
            )

            # We get cross entropy loss of each proposal.
            cross_entropy_per_proposal = (
                tf.nn.softmax_cross_entropy_with_logits(
                    labels=cls_target_one_hot, logits=cls_score_labeled
                )
            )

            if self._debug:
                prediction_dict['_debug']['losses'] = {}
                # Save the cross entropy per proposal to be able to
                # visualize proposals with high and low error.
                prediction_dict['_debug']['losses'][
                    'cross_entropy_per_proposal'
                ] = (
                    cross_entropy_per_proposal
                )

            cls_target_labeled = tf.boolean_mask(
                cls_target, not_ignored, name='cls_target_labeled')
            # `cls_target_labeled` is based on `cls_target` which has
            # `num_classes` + 1 classes.
            # for making `one_hot` with depth `num_classes` to work we need
            # to lower them to make them 0-index.
            cls_target_labeled = cls_target_labeled - 1

            cls_target_one_hot = tf.one_hot(
                cls_target_labeled, depth=self._num_classes,
                name='cls_target_one_hot'
            )

            return {
                'classif_loss': tf.reduce_sum(cross_entropy_per_proposal),
            }
