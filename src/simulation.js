/**
 * YourAI's browser-local teaching mathematics.
 * Softmax, nucleus filtering, DPO loss, and cache accounting use their actual
 * formulas. Tokenization, attention weights, and loss curves are illustrative:
 * they are not outputs or measurements from a trained language model.
 */

const finite = (value, label) => {
  if (!Number.isFinite(value)) throw new TypeError(`${label} must be finite`);
  return value;
};
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const hash = (text) => {
  let result = 2166136261;
  for (const character of String(text)) {
    result ^= character.codePointAt(0);
    result = Math.imul(result, 16777619);
  }
  return result >>> 0;
};

/** Simplified deterministic word/character segmentation, NOT a BPE tokenizer. */
export function tokenize(text) {
  return (String(text ?? '').match(/[\p{Script=Han}]|[\p{L}\p{N}_]+|[^\s]/gu) ?? [])
    .map((text) => ({ text, id: hash(text) % 50000 }));
}

/** Numerically stable softmax; T = 0 denotes deterministic greedy selection. */
export function softmax(logits, temperature = 1) {
  if (!Array.isArray(logits) || logits.length === 0) return [];
  logits.forEach((value) => finite(value, 'logit'));
  finite(temperature, 'temperature');
  if (temperature < 0) throw new RangeError('temperature must be non-negative');
  const maximum = Math.max(...logits);
  if (temperature === 0) {
    const winner = logits.indexOf(maximum);
    return logits.map((_, index) => Number(index === winner));
  }
  const weights = logits.map((value) => Math.exp((value - maximum) / temperature));
  const total = weights.reduce((sum, value) => sum + value, 0);
  return weights.map((value) => value / total);
}

/**
 * Apply temperature, keep the smallest descending prefix reaching topP, then
 * renormalize. A zero threshold still retains the most probable token.
 */
export function sampleDistribution(logits, temperature = 1, topP = 1) {
  finite(topP, 'topP');
  if (topP < 0 || topP > 1) throw new RangeError('topP must be between 0 and 1');
  const probabilities = softmax(logits, temperature);
  if (topP === 1) return probabilities.map((probability, index) => ({ index, probability, retained: true }));
  const ranked = probabilities.map((probability, index) => ({ index, probability }))
    .sort((a, b) => b.probability - a.probability || a.index - b.index);
  const retained = new Set();
  let retainedMass = 0;
  for (const entry of ranked) {
    if (retained.size > 0 && retainedMass >= topP) break;
    retained.add(entry.index);
    retainedMass += entry.probability;
  }
  return probabilities.map((probability, index) => ({
    index,
    probability: retained.has(index) ? probability / retainedMass : 0,
    retained: retained.has(index),
  }));
}

/** Build unique teaching candidates; repeated labels keep their largest preset logit. */
export function uniqueCandidates(tokens, logits) {
  if (tokens.length !== logits.length) throw new RangeError('tokens and logits must have equal length');
  const candidates = new Map();
  tokens.forEach((token, index) => {
    const logit = finite(logits[index], 'logit');
    candidates.set(token, Math.max(candidates.get(token) ?? -Infinity, logit));
  });
  return { tokens: [...candidates.keys()], logits: [...candidates.values()] };
}

/** Toy Q/K features with a real causal mask and row-wise softmax. */
export function attentionMatrix(tokens, head = 0) {
  finite(head, 'head');
  const ids = tokens.map((token) => typeof token === 'object' && token !== null
    ? (token.id ?? hash(token.text)) : hash(token));
  const dimensions = 4;
  const feature = (id, position, channel, query) => Math.sin(
    (id % 997) * 0.013 * (channel + 1) + position * 0.19 + head * 0.83 + (query ? 0.4 : 0),
  );
  return ids.map((queryId, row) => {
    const logits = ids.slice(0, row + 1).map((keyId, col) => {
      let dot = 0;
      for (let channel = 0; channel < dimensions; channel += 1) {
        dot += feature(queryId, row, channel, true) * feature(keyId, col, channel, false);
      }
      return dot / Math.sqrt(dimensions);
    });
    return [...softmax(logits), ...Array(ids.length - row - 1).fill(0)];
  });
}

/**
 * Analytic teaching curve, NOT actual gradient descent or measured training.
 * The demonstration horizon is 200 steps. High learning rates oscillate;
 * lower-quality data produces a larger illustrative generalization gap.
 */
export function trainingStep({ step = 0, learningRate = 0.01, dataQuality = 0.85 } = {}) {
  finite(step, 'step');
  finite(learningRate, 'learningRate');
  finite(dataQuality, 'dataQuality');
  const iteration = Math.max(0, step);
  const rate = Math.max(0, learningRate);
  const quality = clamp(dataQuality, 0, 1);
  const convergence = 1 - Math.exp(-iteration * rate * 1.8);
  const instability = Math.max(0, rate - 0.03) * 22;
  const oscillation = instability * Math.abs(Math.sin(iteration * 0.65));
  const loss = 4.6 - 3.75 * convergence + oscillation;
  const validationLoss = loss + (1 - quality) * convergence * 1.5 + convergence * 0.08;
  const gradientNorm = 2.8 * (1 - convergence) + 0.12 + oscillation * 1.3;
  return { loss, validationLoss, gradientNorm, progress: clamp(iteration / 200, 0, 1) };
}

/** -log σ(β [log πθ(yw)/πref(yw) − log πθ(yl)/πref(yl)]). */
export function dpoLoss({ chosenLogRatio, rejectedLogRatio, beta = 0.1 }) {
  finite(chosenLogRatio, 'chosenLogRatio');
  finite(rejectedLogRatio, 'rejectedLogRatio');
  finite(beta, 'beta');
  if (beta < 0) throw new RangeError('beta must be non-negative');
  const negativeMargin = -beta * (chosenLogRatio - rejectedLogRatio);
  return Math.max(0, negativeMargin) + Math.log1p(Math.exp(-Math.abs(negativeMargin)));
}

/** K and V storage only; excludes model weights, activations and allocator overhead. */
export function kvCacheBytes({ layers, sequenceLength, kvHeads, headDim, bytesPerElement = 2, batchSize = 1 }) {
  const inputs = { layers, sequenceLength, kvHeads, headDim, bytesPerElement, batchSize };
  for (const [label, value] of Object.entries(inputs)) {
    finite(value, label);
    if (value < 0 || !Number.isInteger(value)) throw new RangeError(`${label} must be a non-negative integer`);
  }
  return 2 * layers * sequenceLength * kvHeads * headDim * bytesPerElement * batchSize;
}

/** Rough dense-transformer training FLOPs ≈ 6ND; attention and overhead omitted. */
export function estimateTrainingFlops(params, tokens) {
  finite(params, 'params');
  finite(tokens, 'tokens');
  if (params < 0 || tokens < 0) throw new RangeError('params and tokens must be non-negative');
  return 6 * params * tokens;
}

export function formatNumber(value) {
  finite(value, 'value');
  const magnitude = Math.abs(value);
  for (const [threshold, suffix] of [[1e15, 'P'], [1e12, 'T'], [1e9, 'B'], [1e6, 'M'], [1e3, 'K']]) {
    if (magnitude >= threshold) return `${Number((value / threshold).toFixed(2))}${suffix}`;
  }
  return Number(value.toFixed(2)).toString();
}

/** Repeatable Mulberry32 PRNG for teaching examples; not cryptographic. */
export function seededRandom(seed = 42) {
  let state = typeof seed === 'number' ? seed >>> 0 : hash(seed);
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}
