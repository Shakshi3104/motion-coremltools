import coremltools as ct
from tensorflow.keras.models import Model

from .tensorflow import tf_converter


def convert(model, source='auto', inputs=None, outputs=None, classifier_config=None, minimum_deployment_target=None) -> ct.models.MLModel:
    """Convert TensorFlow models for motion sensor data to the Core ML model format.

    :param model: TensorFlow 2 model as tf.keras.Model
    :param source: One of [`auto`, `tensorflow`, `pytorch`, `mil`]. `auto` determines the framework automatically for most cases.
    :param inputs: list of `TensorType` or `ImageType`
    :param outputs: list[str] (optional)
    :param classifier_config: ClassifierConfig class (optional)
    :param minimum_deployment_target: coremltools.target.enumeration (optional)
    :return: coremltools.models.MLModel
    """
    if isinstance(model, Model) or source == 'tensorflow':
        return tf_converter(model, inputs=inputs, outputs=outputs, classifier_config=classifier_config,
                            minimum_deployment_target=minimum_deployment_target)
    else:
        raise NotImplementedError()