import * as tf from '@tensorflow/tfjs';
import { convBN, CSPBlock } from '../layers';

export const cspDarknet53 = (inputShape: number[]) => {
  const inputs = tf.input({shape: inputShape});

  let x = convBN(inputs, {
    filters: 32,
    kernelSize: 3,
    strides: 1,
    activation: 'mish'
  });

  x = convBN(x, {
    filters: 64,
    kernelSize: 3,
    strides: 2,
    zeroPad: true,
    padding: 'valid',
    activation: 'mish'
  });

  const route = convBN(x, {
    filters: 64,
    kernelSize: 1,
    strides: 1,
    activation: 'mish'
  });

  const shortCut = convBN(x, {
    filters: 64,
    kernelSize: 1,
    strides: 1,
    activation: 'mish'
  });

  x = convBN(shortCut, {
    filters: 32,
    kernelSize: 1,
    strides: 1,
    activation: 'mish'
  });
  x = convBN(x, {
    filters: 64,
    kernelSize: 3,
    strides: 1,
    activation: 'mish'
  });

  // x = x + shortCut
  x = tf.layers.add().apply([x, shortCut]) as tf.SymbolicTensor;

  x = convBN(x, {
    filters: 64,
    kernelSize: 1,
    strides: 1,
    activation: 'mish'
  });
  
  x = tf.layers.concatenate().apply([x, route]) as tf.SymbolicTensor;
  console.log(x.shape)
  x = convBN(x, {
    filters: 64,
    kernelSize: 1,
    strides: 1,
    activation: 'mish'
  });

  x = CSPBlock(x, 128, 2);
  console.log(x.shape)
  const output1 = CSPBlock(x, 256, 8);
  console.log(output1.shape)
  const output2 = CSPBlock(output1, 512, 8);
  console.log(output2.shape)
  const output3 = CSPBlock(output2, 1024, 4);
  console.log(output3.shape)

  return tf.model({
    inputs,
    outputs: [
      output1,
      output2,
      output3
    ],
    name: 'CSP53'
  });
}