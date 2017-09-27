import numpy as np
import tensorflow as tf

from PIL import Image
from luminoth.models import get_model
from luminoth.tools.server.web import get_prediction


class WebTest(tf.test.TestCase):

    def setUp(self):
        super(WebTest, self).setUp()
        self._model_class = get_model('fasterrcnn')
        self._image_resize = self._model_class.base_config.dataset.image_preprocessing  # noqa
        self._image_resize_min = self._image_resize.min_size
        self._image_resize_max = self._image_resize.max_size

        # Does a prediction without resizing the image
        self._image = Image.fromarray(
            np.random.randint(
                low=0, high=255,
                size=(self._image_resize_min + 100, self._image_resize_max - 100, 3)  # noqa
            ).astype(np.uint8)
        )
        self._results = get_prediction('fasterrcnn', self._image)

    def testFasterRCNN(self):
        """
        Tests the FasterRCNN's predict
        """

        # Check that scale_factor and inference_time are corrects values
        self.assertEqual(self._results['scale_factor'], 1.0)
        self.assertGreaterEqual(self._results['inference_time'], 0)

        # Check that objects, labels and probs aren't None
        self.assertNotEqual(self._results['objects'], None)
        self.assertNotEqual(self._results['objects_labels'], None)
        self.assertNotEqual(self._results['objects_labels_prob'], None)

        # Does a prediction resizing the image
        image = Image.fromarray(
            np.random.randint(
                low=0, high=255,
                size=(self._image_resize_max + 100, self._image_resize_max + 100, 3)  # noqa
            ).astype(np.uint8)
        )
        results = get_prediction('fasterrcnn', image)

        # Check that scale_factor and inference_time are corrects values
        self.assertGreaterEqual(1.0, results['scale_factor'])
        self.assertGreaterEqual(results['inference_time'], 0)

        # Check that objects, labels and probs aren't None
        self.assertNotEqual(results['objects'], None)
        self.assertNotEqual(results['objects_labels'], None)
        self.assertNotEqual(results['objects_labels_prob'], None)


if __name__ == '__main__':
    tf.test.main()
