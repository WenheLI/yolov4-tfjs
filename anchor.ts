import * as tf from '@tensorflow/tfjs';

export const YOLOV4_ANCHORS = [
  tf.tensor([[12, 16], [19, 36], [40, 28]], [3, 2], 'float32'),
  tf.tensor([[36, 75], [76, 55], [72, 146]], [3, 2], 'float32'),
  tf.tensor([[142, 110], [192, 243], [459, 401]], [3, 2], 'float32')
];

export const computeNormAnchors = (anchors: Array<Array<number>>, inputShape: Array<number>) => {
  const height = inputShape[0];
  const width = inputShape[1];
  return tf.tidy(() => {
    const tensorOrigin = tf.tensor([width, height]);
    const tensorAnchors = anchors.map((it) => tf.tensor(it));
    return tensorAnchors.map((it) => tf.div(it, tensorOrigin));
  });
}