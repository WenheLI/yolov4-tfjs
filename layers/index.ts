import * as tf from '@tensorflow/tfjs';

type Padding = 'same' | 'valid';
type Activation = 'mish' | 'leaky_relu';

interface ConvBNArgs {
  filters: number, 
  kernelSize: number, 
  strides: number, 
  padding?: Padding, 
  zeroPad?: boolean, 
  activation?: Activation
}

export const convBN = (inputs: tf.SymbolicTensor, args: ConvBNArgs): tf.SymbolicTensor => {
  if (args.zeroPad) {
    inputs = tf.layers.zeroPadding2d({
      padding: [[1, 0], [1, 0]]
    }).apply(inputs) as tf.SymbolicTensor;
  }

  inputs = tf.layers.conv2d({
    filters: args.filters,
    kernelSize: args.kernelSize,
    strides: args.strides,
    padding: args.padding ? args.padding : 'same',
    useBias: false
  }).apply(inputs) as tf.SymbolicTensor;

  inputs = tf.layers.batchNormalization().apply(inputs) as tf.SymbolicTensor;
  const activation = args.activation ? args.activation : 'leaky_relu';
  if (activation == 'leaky_relu') {
    inputs = tf.layers.leakyReLU({
      alpha: .1
    }).apply(inputs) as tf.SymbolicTensor;
  } else if (activation == 'mish') {
    // @TODO await https://github.com/tensorflow/tfjs/pull/4950
    inputs = tf.layers.leakyReLU({
      alpha: .1
    }).apply(inputs) as tf.SymbolicTensor;
  }

  return inputs;
}

export const ResidualBlock = (inputs: tf.SymbolicTensor, numBlocks: number): tf.SymbolicTensor => {
  const [b, w, h, filters] = inputs.shape;
  let x = inputs;
  for (let i = 0; i < filters; i++) {
    const blockInputs = x;
    x = convBN(x, {
      filters,
      kernelSize: 1,
      strides: 1,
      activation: 'mish'
    });
    x = convBN(x, {
      filters,
      kernelSize: 3,
      strides: 1
    });

    x = tf.layers.add().apply([x, blockInputs]) as tf.SymbolicTensor;
  }

  return x;
}

export const CSPBlock = (inputs: tf.SymbolicTensor, filters: number, numBlocks: number): tf.SymbolicTensor => {
  const halfFilters = Math.floor(filters / 2);

  let x = convBN(inputs, {
    filters,
    kernelSize: 3,
    strides: 2,
    zeroPad: true,
    padding: 'valid',
    activation: 'mish'
  });
  let route = convBN(x, {
    filters: halfFilters,
    kernelSize: 1,
    strides: 1,
    activation: 'mish'
  });

  x = convBN(x, {
    filters: halfFilters,
    kernelSize: 1,
    strides: 1,
    activation: 'mish'
  });
  x = ResidualBlock(x, numBlocks);
  x = convBN(x, {
    filters: halfFilters,
    kernelSize: 1,
    strides: 1,
    activation: 'mish'
  });
  x = tf.layers.concatenate().apply([x, route]) as tf.SymbolicTensor;

  x = convBN(x, {
    filters,
    kernelSize: 1,
    strides: 1,
    activation: 'mish'
  });

  return x;
}