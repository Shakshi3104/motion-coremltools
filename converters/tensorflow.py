import coremltools as ct
from tensorflow.keras.models import Model

from .utils import add_reshape_layer


def tf_converter(model: Model, inputs=None, outputs=None, classifier_config: ct.ClassifierConfig=None, minimum_deployment_target=None):
    # add reshape layer
    model = add_reshape_layer(model)

    mlmodel = ct.convert(model, classifier_config=classifier_config,
                         inputs=inputs, outputs=outputs,
                         minimum_deployment_target=minimum_deployment_target)

    return mlmodel