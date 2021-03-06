import numpy as np
import tensorflow as tf

from easydict import EasyDict
from luminoth.models.fasterrcnn.rcnn_target import RCNNTarget


class RCNNTargetTest(tf.test.TestCase):

    def setUp(self):
        super(RCNNTargetTest, self).setUp()

        # We don't care about the class labels or the batch number in most of
        # these tests.
        self._num_classes = 5
        self._placeholder_label = 3.
        self._batch_number = 1

        self._config = EasyDict({
            'foreground_threshold': 0.5,
            'background_threshold_high': 0.5,
            'background_threshold_low': 0.1,
            'foreground_fraction': 0.5,
            'minibatch_size': 2,
        })
        # We check for a difference smaller than this numbers in our tests
        # instead of checking for exact equality.
        self._equality_delta = 1e-03

        self._shared_model = RCNNTarget(self._num_classes, self._config)

    def _run_rcnn_target(self, model, gt_boxes, proposed_boxes):
        """Runs an instance of RCNNTarget

        Args:
            model: an RCNNTarget model.
            gt_boxes: a Tensor holding the ground truth boxes, with shape
                (num_gt, 5). The last value is the class label.
            proposed_boxes: a Tensor holding the proposed boxes. Its shape is
                (num_proposals, 5). The first value is the batch number.

        Returns:
            The tuple returned by RCNNTarget._build().
        """
        rcnn_target_net = model(proposed_boxes, gt_boxes)
        with self.test_session() as sess:
            return sess.run(rcnn_target_net)

    def testBasic(self):
        """Tests a basic case.

        We have one ground truth box and three proposals. One should be
        background, one foreground, and one should be an ignored background
        (i.e. less IoU than whatever value is set as
        config.background_threshold_low).
        """

        gt_boxes = tf.constant(
            [(20, 20, 80, 100, self._placeholder_label)],
            dtype=tf.float32
        )

        proposed_boxes = tf.constant(
            [
                (self._batch_number, 55, 75, 85, 105),  # Background
                                                        # IoU ~0.1293
                (self._batch_number, 25, 21, 85, 105),  # Foreground
                                                        # IoU ~0.7934
                (self._batch_number, 78, 98, 99, 135),  # Ignored
                                                        # IoU ~0.0015
            ],
            dtype=tf.float32
        )

        proposals_label, bbox_targets = self._run_rcnn_target(
            self._shared_model, gt_boxes, proposed_boxes
        )
        # We test that all values are 'close' (up to self._equality_delta)
        # instead of equal to avoid failing due to a floating point rounding
        # error.
        # We sum 1 to the placeholder label because rcnn_target does the same
        # due to the fact that it uses 0 to signal 'background'.
        self.assertAllClose(
            proposals_label,
            np.array([0., self._placeholder_label + 1, -1.]),
            atol=self._equality_delta
        )

    def testEmptyCase(self):
        """Tests we're choosing the best box when none are above the
        foreground threshold.
        """

        gt_boxes = tf.constant(
            [(423, 30, 501, 80, self._placeholder_label)],
            dtype=tf.float32
        )

        proposed_boxes = tf.constant(
            [
                (self._batch_number, 491, 70, 510, 92),  # IoU 0.0277
                (self._batch_number, 400, 60, 450, 92),  # IoU 0.1147
                (self._batch_number, 413, 40, 480, 77),  # IoU 0.4998: highest
                (self._batch_number, 411, 40, 480, 77),  # IoU 0.4914
            ],
            dtype=tf.float32
        )

        proposals_label, bbox_targets = self._run_rcnn_target(
            self._shared_model, gt_boxes, proposed_boxes
        )

        # Assertions
        self.assertAlmostEqual(
            proposals_label[2], self._placeholder_label + 1,
            delta=self._equality_delta
        )

        for i, label in enumerate(proposals_label):
            if i != 2:
                self.assertLess(label, 1)

    def testAbsolutelyEmptyCase(self):
        """Tests the code doesn't break when there's no proposals with IoU > 0.
        """

        gt_boxes = tf.constant(
            [(40, 90, 100, 105, self._placeholder_label)],
            dtype=tf.float32
        )

        proposed_boxes = tf.constant(
            [
                (self._batch_number, 0, 0, 39, 89),
                (self._batch_number, 101, 106, 300, 450),
                (self._batch_number, 340, 199, 410, 420),
            ],
            dtype=tf.float32
        )

        (proposals_label, bbox_targets) = self._run_rcnn_target(
            self._shared_model,
            gt_boxes,
            proposed_boxes
        )
        foreground_fraction = self._config.foreground_fraction
        minibatch_size = self._config.minibatch_size
        correct_foreground_number = np.floor(
            foreground_fraction * minibatch_size
        )

        foreground_number = proposals_label[proposals_label >= 1].shape[0]
        background_number = proposals_label[proposals_label == 0].shape[0]

        self.assertAlmostEqual(
            foreground_number, correct_foreground_number,
            delta=self._equality_delta
        )

        self.assertLessEqual(
            foreground_number,
            np.floor(foreground_fraction * minibatch_size)
        )
        self.assertLess(
            background_number,
            np.ceil(foreground_fraction * minibatch_size)
        )

    def testMultipleOverlap(self):
        """Tests that we're choosing a foreground box when there's several for
        the same gt box.
        """

        gt_boxes = tf.constant(
            [(200, 300, 250, 390, self._placeholder_label)],
            dtype=tf.float32
        )

        proposed_boxes = tf.constant(
            [
                (self._batch_number, 12, 70, 350, 540),  # noise
                (self._batch_number, 190, 310, 240, 370),  # IoU: 0.4763
                (self._batch_number, 197, 300, 252, 389),  # IoU: 0.9015
                (self._batch_number, 196, 300, 252, 389),  # IoU: 0.8859
                (self._batch_number, 197, 303, 252, 394),  # IoU: 0.8459
                (self._batch_number, 180, 310, 235, 370),  # IoU: 0.3747
                (self._batch_number, 0, 0, 400, 400),  # noise
                (self._batch_number, 197, 302, 252, 389),  # IoU: 0.8832
                (self._batch_number, 0, 0, 400, 400),  # noise
            ],
            dtype=tf.float32
        )

        proposals_label, bbox_targets = self._run_rcnn_target(
            self._shared_model, gt_boxes, proposed_boxes
        )

        # Assertions
        foreground_number = proposals_label[proposals_label >= 1].shape[0]
        background_number = proposals_label[proposals_label == 0].shape[0]

        foreground_fraction = self._config.foreground_fraction

        self.assertAlmostEqual(
            foreground_number,
            np.floor(self._config.minibatch_size * foreground_fraction),
            delta=self._equality_delta
        )
        self.assertAlmostEqual(
            background_number,
            np.ceil(self._config.minibatch_size * foreground_fraction),
            delta=self._equality_delta
        )

        foreground_idxs = np.nonzero(proposals_label >= 1)
        for foreground_idx in foreground_idxs:
            self.assertIn(foreground_idx, [2, 3, 4, 7])

    def testOddMinibatchSize(self):
        """Tests we're getting the right results when there's an odd minibatch
        size.
        """

        config = EasyDict({
            'foreground_threshold': 0.5,
            'background_threshold_high': 0.5,
            'background_threshold_low': 0.1,
            'foreground_fraction': 0.5,
            'minibatch_size': 5,
        })

        model = RCNNTarget(self._num_classes, config)

        gt_boxes = tf.constant(
            [(200, 300, 250, 390, self._placeholder_label)],
            dtype=tf.float32
        )

        proposed_boxes = tf.constant(
            [
                (self._batch_number, 12, 70, 350, 540),  # noise
                (self._batch_number, 190, 310, 240, 370),  # IoU: 0.4763
                (self._batch_number, 197, 300, 252, 389),  # IoU: 0.9015
                (self._batch_number, 196, 300, 252, 389),  # IoU: 0.8859
                (self._batch_number, 197, 303, 252, 394),  # IoU: 0.8459
                (self._batch_number, 180, 310, 235, 370),  # IoU: 0.3747
                (self._batch_number, 0, 0, 400, 400),  # noise
                (self._batch_number, 197, 302, 252, 389),  # IoU: 0.8832
                (self._batch_number, 180, 310, 235, 370),  # IoU: 0.3747
                (self._batch_number, 180, 310, 235, 370),  # IoU: 0.3747
                (self._batch_number, 0, 0, 400, 400),  # noise
            ],
            dtype=tf.float32
        )

        (proposals_label, bbox_targets) = self._run_rcnn_target(
            model,
            gt_boxes,
            proposed_boxes
        )

        foreground_number = proposals_label[proposals_label >= 1].shape[0]
        background_number = proposals_label[proposals_label == 0].shape[0]

        foreground_fraction = config.foreground_fraction
        minibatch_size = config.minibatch_size

        self.assertLessEqual(
            foreground_number,
            np.floor(foreground_fraction * minibatch_size)
        )
        self.assertLessEqual(
            background_number,
            np.ceil(foreground_fraction * minibatch_size)
        )

    def testBboxTargetConsistency(self):
        """Tests that bbox_targets is consistent with proposals_label.

        That means we test that we have the same number of elements in
        bbox_targets and proposals_label, and that only the proposals
        marked with a class are assigned a non-zero bbox_target.
        """

        config = EasyDict({
            'foreground_threshold': 0.5,
            'background_threshold_high': 0.5,
            'background_threshold_low': 0.1,
            'foreground_fraction': 0.5,
            # We change the minibatch_size the catch all our foregrounds
            'minibatch_size': 8,
        })

        model = RCNNTarget(self._num_classes, config)

        gt_boxes = tf.constant(
            [(200, 300, 250, 390, self._placeholder_label)],
            dtype=tf.float32
        )

        proposed_boxes = tf.constant(
            [
                (self._batch_number, 12, 70, 350, 540),  # noise
                (self._batch_number, 190, 310, 240, 370),  # IoU: 0.4763
                (self._batch_number, 197, 300, 252, 389),  # IoU: 0.9015
                (self._batch_number, 196, 300, 252, 389),  # IoU: 0.8859
                (self._batch_number, 197, 303, 252, 394),  # IoU: 0.8459
                (self._batch_number, 180, 310, 235, 370),  # IoU: 0.3747
                (self._batch_number, 0, 0, 400, 400),  # noise
                (self._batch_number, 197, 302, 252, 389),  # IoU: 0.8832
                (self._batch_number, 0, 0, 400, 400),  # noise
            ],
            dtype=tf.float32
        )

        (proposals_label, bbox_targets) = self._run_rcnn_target(
            model,
            gt_boxes,
            proposed_boxes
        )

        foreground_idxs = np.nonzero(proposals_label >= 1)
        non_empty_bbox_target_idxs = np.nonzero(np.any(bbox_targets, axis=1))

        self.assertAllEqual(
            foreground_idxs, non_empty_bbox_target_idxs
        )

    def testMultipleGtBoxes(self):
        """Tests we're getting the right labels when there's several gt_boxes.
        """

        num_classes = 3
        config = EasyDict({
            'foreground_threshold': 0.5,
            'background_threshold_high': 0.5,
            'background_threshold_low': 0.1,
            'foreground_fraction': 0.5,
            # We change the minibatch_size the catch all our foregrounds
            'minibatch_size': 18,
        })
        model = RCNNTarget(num_classes, config)

        gt_boxes = tf.constant(
            [
                (10, 0, 398, 399, 0),
                (200, 300, 250, 390, 1),
                (185, 305, 235, 372, 2),
            ],
            dtype=tf.float32
        )
        proposed_boxes = tf.constant(
            [
                (self._batch_number, 12, 70, 350, 540),  # noise
                (self._batch_number, 190, 310, 240, 370),  # 2
                (self._batch_number, 197, 300, 252, 389),  # 1
                (self._batch_number, 196, 300, 252, 389),  # 1
                (self._batch_number, 197, 303, 252, 394),  # 1
                (self._batch_number, 180, 310, 235, 370),  # 2
                (self._batch_number, 0, 0, 400, 400),  # 0
                (self._batch_number, 197, 302, 252, 389),  # 1
                (self._batch_number, 0, 0, 400, 400),  # 0
            ],
            dtype=tf.float32
        )

        (proposals_label, bbox_targets) = self._run_rcnn_target(
            model,
            gt_boxes,
            proposed_boxes
        )
        # We don't care much about the first value.
        self.assertAllClose(
            proposals_label[1:],
            # We sum one to normalize for RCNNTarget's results.
            np.add([2., 1., 1., 1., 2., 0., 1., 0.], 1),
            self._equality_delta
        )


if __name__ == '__main__':
    tf.test.main()
