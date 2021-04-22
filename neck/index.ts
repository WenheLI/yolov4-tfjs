import * as tf from '@tensorflow/tfjs';
import { convBN } from '../layers';

export const neck = (inputShapes: Array<Array<number>>) => {
  const input1 = tf.input({shape: inputShapes[0].slice(1)});
  const input2 = tf.input({shape: inputShapes[1].slice(1)});
  const input3 = tf.input({shape: inputShapes[2].slice(1)});

  console.log(input3.shape)

  let x = convBN(input3, {
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

  let maxpool1 = tf.layers.maxPool2d({
    poolSize: [5, 5],
    strides: 1,
    padding: 'same'
  }).apply(x) as tf.SymbolicTensor;

  let maxpool2 = tf.layers.maxPool2d({
    poolSize: [9, 9],
    strides: 1,
    padding: 'same'
  }).apply(x) as tf.SymbolicTensor;

  let maxpool3 = tf.layers.maxPool2d({
    poolSize: [13, 13],
    strides: 1,
    padding: 'same'
  }).apply(x) as tf.SymbolicTensor;

  const spp = tf.layers.concatenate().apply([maxpool3, maxpool2, maxpool1, x]) as tf.SymbolicTensor;

  x = convBN(spp, {
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
  const output3 = convBN(x, {
    filters: 512,
    kernelSize: 1,
    strides: 1,
    activation: 'leaky_relu'
  });
  
  x = convBN(output3, {
    filters: 256,
    kernelSize: 1,
    strides: 1,
    activation: 'leaky_relu'
  });

  let upsampled = tf.layers.upSampling2d({}).apply(x) as tf.SymbolicTensor;
  
  x = convBN(input2, {
    filters: 256,
    kernelSize: 1,
    strides: 1,
    activation: 'leaky_relu'
  });
  x = tf.layers.concatenate().apply([x, upsampled]) as tf.SymbolicTensor;

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
  const output2 = convBN(x, {
    filters: 256,
    kernelSize: 1,
    strides: 1,
    activation: 'leaky_relu'
  });

  x = convBN(output2, {
    filters: 128, 
    kernelSize: 1,
    strides: 1,
    activation: 'leaky_relu'
  });

  upsampled = tf.layers.upSampling2d({}).apply(x) as tf.SymbolicTensor;
  x = convBN(input1, {
    filters: 128,
    kernelSize: 1,
    strides: 1,
    activation: 'leaky_relu'
  });
  x = tf.layers.concatenate().apply([x, upsampled]) as tf.SymbolicTensor;

  x = convBN(x, {
    filters: 128,
    kernelSize: 1,
    strides: 1,
    activation: 'leaky_relu'
  });
  x = convBN(x, {
    filters: 256,
    kernelSize: 3,
    strides: 1,
    activation: 'leaky_relu'
  });
  x = convBN(x, {
    filters: 128,
    kernelSize: 1,
    strides: 1,
    activation: 'leaky_relu'
  });
  x = convBN(x, {
    filters: 256,
    kernelSize: 3,
    strides: 1,
    activation: 'leaky_relu'
  });

  const output1 = convBN(x, {
    filters: 128,
    kernelSize: 1,
    strides: 1,
    activation: 'leaky_relu'
  });

  return tf.model({
    inputs: [input1, input2, input3],
    outputs: [output1, output2, output3],
    name: 'YOLOV4_neck'
  })
}