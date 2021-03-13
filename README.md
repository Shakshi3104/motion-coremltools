# motion-coremltools

motion-coremltools is the wrapper tool for converting neural networks trained with motion sensor data.

## Usage
The usage is the same as coremltools' Unified Conversation API. Currently, only tensorflow.keras.Model is supported. Also, only **1-demensional** CNNs are supported.

```python
import coremltools as ct
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense
from tensorflow.keras.models import Model

from coremotiontools import convert

# Build model
inputs = Input(shape=(256*3, 1))
x = Conv1D(16, kernel_size=3, strides=1, padding='same', activation='relu', kernel_initalizer='he_normal')(inputs)
x = MaxPooling1D(pool_size=2, padding='same')(x)

x = Flatten()(x)
x = Dense(1024, activation='relu', kernel_initalizer='he_normal')(x)
outputs = Dense(6, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

# Convert to Core ML
classifier_config = ct.ClassifierConfig(class_labels=["stay", "walk", "jog", "skip", "stUp", "stDown"])
mlmodel = convert(model, classifier_config=classifier_config)
mlmodel.save("ActivityClassifier.mlmodel")
```

See [here](https://github.com/Shakshi3104/TF_CNN_Collection_to_CoreML/blob/master/main.py) for more detailed usages.

### In Core ML

When the converted model is used in Core ML, the input type is **MLMultiArray**.

```swift
let input: [Double] = [0.1, 0.2, 0.3, ...] // Order of x-axis, y-axis, z-axis, x, y, z, ...
let mlArray = try! MLMultiArray.fromDouble(input) // MLMultiArray.fromDouble() is extension

// Predict
let model = ActivityClassifier()
let output = try! model.prediction(input: ActivityClassifierInput(input: mlArray))
```

See [here](https://github.com/Shakshi3104/footway/blob/master/SidewalkSurfaceTypeClassification/SidewalkSurfaceTypeClassifier.swift) for more detailed usages.



## Requirements

- `coremltools 4.1`
- `tensorflow 2.1`