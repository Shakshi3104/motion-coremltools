# motion-coremltools

motion-coremltools is the wrapper tool for neural networks with motion sensor data.

## Usage
The usage is the same as coremltools' Unified Conversation API. Currently, only tensorflow.keras.Model is supported.

```python
import coremltools as ct
from coremotiontools import convert

classifier_config = ct.ClassifierConfig(class_labels=["stay", "walk", "jog", "skip", "stUp", "stDown"])
mlmodel = convert(model, classifier_config=classifier_config)
```

See [here](https://github.com/Shakshi3104/TF_CNN_Collection_to_CoreML/blob/master/main.py) for more detailed usages.