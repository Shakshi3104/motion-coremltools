from tensorflow.keras.layers import Input, Reshape
from tensorflow.keras.models import Model


# Add Reshape layer
def add_reshape_layer(model: Model) -> Model:
    width = model.input.shape[1]
    channels = model.input.shape[2]

    # Add Reshape Layer
    # MLMultiArray is 1-dimensional NSNumber array
    # input shape of tf.keras is (window_size, )
    inputs = Input(shape=(width * channels, ), name="input")
    x = Reshape((width, channels), name="reshape")(inputs)

    # concat reshape_input and model
    model_output = model(x)

    reshaped_model = Model(inputs=inputs, outputs=model_output)
    return reshaped_model