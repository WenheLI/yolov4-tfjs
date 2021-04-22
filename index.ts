import * as tf from '@tensorflow/tfjs';
import { cspDarknet53 } from './backbone';
import { head } from './head';
import { neck } from './neck';
import { computeNormAnchors } from './anchor';

const yolo = async (inputShape: Array<number>, anchors: Array<Array<number>>, numClasses: number) => {
  if (inputShape[0] % 32 !== 0 || inputShape[1] % 32 !== 0) {
    throw new Error();
  }

  const backbone = cspDarknet53(inputShape);
  console.log(backbone.outputShape)
  const neckNetwork = neck(backbone.outputShape as number[][]);

  const normalizedAnchors = await Promise.all(computeNormAnchors(anchors, inputShape).map((it) => {
    return it.array();
  })) as Array<Array<number>>;
  const headNetwork = head(neckNetwork.outputShape as number[][], normalizedAnchors, numClasses);
  const yolo = tf.sequential();
  yolo.add(backbone);
  yolo.add(neckNetwork);
  yolo.add(headNetwork);

  yolo.summary()
;}

const main = async () => {
  await yolo([320, 320, 3], [[1, 2], [1, 2], [1, 2]], 10);
}

main()