import coremltools as ct
from tensorflow.keras.models import Model

from .tensorflow import tf_converter


def convert(model, source='auto', inputs=None, outputs=None, classifier_config=None, minimum_deployment_target=None) -> ct.models.MLModel:

    if isinstance(model, Model) or source == 'tensorflow':
        return tf_converter(model, inputs=inputs, outputs=outputs, classifier_config=classifier_config,
                            minimum_deployment_target=minimum_deployment_target)
    else:
        raise NotImplementedError()