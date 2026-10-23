import test from 'node:test';
import assert from 'node:assert/strict';
import {
  tokenize, softmax, sampleDistribution, uniqueCandidates, attentionMatrix, trainingStep,
  dpoLoss, kvCacheBytes, estimateTrainingFlops, formatNumber, seededRandom,
} from '../src/simulation.js';

const near = (actual, expected, tolerance = 1e-10) => assert.ok(
  Math.abs(actual - expected) < tolerance, `${actual} differs from ${expected}`,
);
const sum = (values) => values.reduce((total, value) => total + value, 0);

test('softmax is normalized, translation-invariant and stable for large logits', () => {
  const normal = softmax([2, 1, 0]);
  const shifted = softmax([10002, 10001, 10000]);
  near(sum(normal), 1);
  normal.forEach((value, index) => near(value, shifted[index]));
  assert.deepEqual(softmax([10000, -10000]), [1, 0]);
  assert.deepEqual(softmax([]), []);
});

test('lower temperature concentrates probability; zero is deterministic greedy', () => {
  assert.ok(softmax([2, 1, 0], 0.3)[0] > softmax([2, 1, 0], 1)[0]);
  assert.ok(softmax([2, 1, 0], 2)[0] < softmax([2, 1, 0], 1)[0]);
  assert.deepEqual(softmax([1, 3, 3], 0), [0, 1, 0]);
  assert.throws(() => softmax([1, 2], -1), RangeError);
});

test('nucleus filtering preserves source order and renormalizes the smallest prefix', () => {
  const result = sampleDistribution([Math.log(0.2), Math.log(0.5), Math.log(0.3)], 1, 0.75);
  assert.deepEqual(result.map((entry) => entry.index), [0, 1, 2]);
  assert.deepEqual(result.map((entry) => entry.retained), [false, true, true]);
  near(result[1].probability, 0.625);
  near(result[2].probability, 0.375);
  near(sum(result.map((entry) => entry.probability)), 1);
});

test('nucleus always retains at least one candidate even at zero threshold', () => {
  const result = sampleDistribution([1, 4, 2], 1, 0);
  assert.deepEqual(result.map((entry) => entry.probability), [0, 1, 0]);
  assert.equal(result.filter((entry) => entry.retained).length, 1);
  assert.deepEqual(sampleDistribution([], 1, 0.9), []);
  assert.ok(sampleDistribution([100, 1, 0], 1, 1).every((entry) => entry.retained));
  assert.throws(() => sampleDistribution([1], 1, 1.1), RangeError);
});

test('repeated punctuation and EOS become one candidate before sampling', () => {
  const candidates = uniqueCandidates(['。', '回答', '。', '[EOS]', '[EOS]'], [3.2, 2.3, 1.5, -1, 3.7]);
  assert.deepEqual(candidates.tokens, ['。', '回答', '[EOS]']);
  assert.deepEqual(candidates.logits, [3.2, 2.3, 3.7]);
  const distribution = sampleDistribution(candidates.logits, 0.8, 0.9);
  near(sum(distribution.map((entry) => entry.probability)), 1);
  assert.deepEqual(uniqueCandidates([], []), { tokens: [], logits: [] });
});

test('teaching attention has normalized rows and a strict causal mask', () => {
  const tokens = tokenize('语言模型 predicts 下一个 token。');
  const matrix = attentionMatrix(tokens);
  assert.equal(matrix.length, tokens.length);
  matrix.forEach((row, i) => {
    assert.equal(row.length, tokens.length);
    near(sum(row), 1);
    row.forEach((value, j) => {
      if (j > i) assert.equal(value, 0);
      else assert.ok(value > 0);
    });
  });
  assert.deepEqual(attentionMatrix([]), []);
  assert.notDeepEqual(matrix, attentionMatrix(tokens, 1));
});

test('DPO loss rewards increasing the preferred response log-ratio', () => {
  near(dpoLoss({ chosenLogRatio: 0, rejectedLogRatio: 0, beta: 0.1 }), Math.log(2));
  assert.ok(dpoLoss({ chosenLogRatio: 2, rejectedLogRatio: -1, beta: 0.1 })
    < dpoLoss({ chosenLogRatio: 0, rejectedLogRatio: -1, beta: 0.1 }));
  assert.equal(dpoLoss({ chosenLogRatio: 10000, rejectedLogRatio: -10000, beta: 1 }), 0);
  assert.equal(dpoLoss({ chosenLogRatio: -10000, rejectedLogRatio: 10000, beta: 1 }), 20000);
});

test('KV cache counts both K and V and scales with sequence, batch and precision', () => {
  const model = { layers: 32, sequenceLength: 4096, kvHeads: 8, headDim: 128 };
  const baseline = kvCacheBytes(model);
  assert.equal(baseline, 536870912);
  assert.equal(kvCacheBytes({ ...model, sequenceLength: 8192 }), baseline * 2);
  assert.equal(kvCacheBytes({ ...model, batchSize: 2 }), baseline * 2);
  assert.equal(kvCacheBytes({ ...model, bytesPerElement: 1 }), baseline / 2);
  assert.equal(kvCacheBytes({ ...model, sequenceLength: 0 }), 0);
  assert.throws(() => kvCacheBytes({ ...model, layers: -1 }), RangeError);
});

test('teaching tokenizer preserves visible units and assigns reproducible IDs', () => {
  const first = tokenize('你好 AI!');
  assert.deepEqual(first.map((token) => token.text), ['你', '好', 'AI', '!']);
  assert.deepEqual(first, tokenize('你好 AI!'));
  assert.deepEqual(tokenize('  '), []);
});

test('analytic training curve responds to rate and data quality without pretending to train', () => {
  const start = trainingStep({ step: 0 });
  const end = trainingStep({ step: 200 });
  assert.ok(end.loss < start.loss);
  assert.equal(start.progress, 0);
  assert.equal(end.progress, 1);
  near(trainingStep({ step: 200, learningRate: 0 }).loss, start.loss);
  assert.ok(trainingStep({ step: 200, dataQuality: 0.2 }).validationLoss > end.validationLoss);
});

test('FLOPs approximation and repeatable RNG support reproducible demonstrations', () => {
  assert.equal(estimateTrainingFlops(7e9, 1e12), 4.2e22);
  assert.equal(formatNumber(7e9), '7B');
  const a = seededRandom(2026);
  const b = seededRandom(2026);
  for (let index = 0; index < 30; index += 1) {
    const value = a();
    assert.equal(value, b());
    assert.ok(value >= 0 && value < 1);
  }
});
