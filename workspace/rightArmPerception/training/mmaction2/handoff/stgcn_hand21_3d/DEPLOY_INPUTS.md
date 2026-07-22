# STGCN Hand21 Deployment Inputs

## Files to hand off

- `model.onnx` or `model.rknn`
- `meta/hand21_labels.txt`
- `meta/hand21_label_mapping.json`
- `meta/model_input_requirements.json`

## Common input convention

- Tensor layout: `(batch, num_frames, num_joints, channels)`
- `batch = 1` for normal realtime inference
- `num_joints = 21`
- `channels = (x, y, score)`
- Keypoint order must match the training data exactly
- Input to this model is skeleton data, not RGB video frames

## Model-specific input shapes

- `u36`: `(1, 36, 21, 3)`
- `u16`: `(1, 16, 21, 3)`
- `u8`: `(1, 8, 21, 3)`

## Label mapping

- `0 -> WAVE_IN`
- `1 -> FIVE`
- `2 -> NONE`

## Output interpretation

- Model output is a score vector with shape `(1, 3)`
- Use `argmax` on the class dimension to get the predicted label index
- Convert the index to the final label with `hand21_label_mapping.json` or `hand21_labels.txt`
