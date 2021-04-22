import * as tf from '@tensorflow/tfjs';

import { convBN } from '../layers';

const convClassesAnchors = (inputs: tf.SymbolicTensor, numAchorsStage: number, numClasses: number) => {
  let x = tf.layers.conv2d({
    filters: numAchorsStage * (numClasses + 5),
    kernelSize: 1,
    strides: 1,
    padding: 'same',
    useBias: true
  }).apply(inputs) as tf.SymbolicTensor;

  x = tf.layers.reshape({
    targetShape: [x.shape[1], x.shape[2], numAchorsStage, numClasses + 5]
  }).apply(x) as tf.SymbolicTensor;

  return x;
}

export const head = (inputShape: Array<Array<number>>, anchors: Array<Array<number>>, numClasses: number) => {
  const input1 = tf.layers.input({shape: inputShape[0].slice(1)});
  const input2 = tf.layers.input({shape: inputShape[1].slice(1)});
  const input3 = tf.layers.input({shape: inputShape[2].slice(1)});

  let x = convBN(input1, {
    filters: 256,
    kernelSize: 3,
    strides: 1,
    activation: 'leaky_relu'
  });

  const output1 = convClassesAnchors(x, anchors[0].length, numClasses);

  x = convBN(input1, {
    filters: 256,
    kernelSize: 3,
    strides: 2,
    zeroPad: true,
    padding: 'valid',
    activation: 'leaky_relu'
  });
  x = tf.layers.concatenate().apply([x, input2]) as tf.SymbolicTensor;
  x = convBN(x, {
    filters: 256,
    kernelSize: 1,
    strides: 1,
    activation: 'leaky_relu'
  });
  x = convBN(x, {
    filters: 512,
    kernelSize: 3,
    strides: 1,
    activation: 'leaky_relu'
  });
  x = convBN(x, {
    filters: 256,
    kernelSize: 1,
    strides: 1,
    activation: 'leaky_relu'
  });
  x = convBN(x, {
    filters: 512,
    kernelSize: 3,
    strides: 1,
    activation: 'leaky_relu'
  });
  const connection = convBN(x, {
    filters: 256,
    kernelSize: 1,
    strides: 1,
    activation: 'leaky_relu'
  });
  x = convBN(connection, {
    filters: 512,
    kernelSize: 3,
    strides: 1,
    activation: 'leaky_relu'
  });
  
  const output2 = convClassesAnchors(x, anchors[1].length, numClasses);

  x = convBN(connection, {
    filters: 512,
    kernelSize: 3,
    strides: 2,
    zeroPad: true,
    padding: 'valid',
    activation: 'leaky_relu'
  });
  x = tf.layers.concatenate().apply([x, input3]) as tf.SymbolicTensor;
  x = convBN(x, {
    filters: 512,
    kernelSize: 1,
    strides: 1,
    activation: 'leaky_relu'
  });
  x = convBN(x, {
    filters: 1024,
    kernelSize: 3,
    strides: 1,
    activation: 'leaky_relu'
  });
  x = convBN(x, {
    filters: 512,
    kernelSize: 1,
    strides: 1,
    activation: 'leaky_relu'
  });
  x = convBN(x, {
    filters: 1024,
    kernelSize: 3,
    strides: 1,
    activation: 'leaky_relu'
  });
  x = convBN(x, {
    filters: 512,
    kernelSize: 1,
    strides: 1,
    activation: 'leaky_relu'
  });
  x = convBN(x, {
    filters: 1024,
    kernelSize: 3,
    strides: 1,
    activation: 'leaky_relu'
  });

  const output3 = convClassesAnchors(x, anchors[2].length, numClasses);

  return tf.model({
    inputs: [input1, input2, input3],
    outputs: [output1, output2, output3]
  });
}