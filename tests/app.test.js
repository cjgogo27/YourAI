import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { Window } from 'happy-dom';

/** DOM integration only: Three.js rendering and browser layout are not exercised. */
async function mountApp() {
  const window = new Window({ url: 'https://example.test/YourAI/' });
  window.document.body.innerHTML = '<div id="app"></div>';
  Object.defineProperty(window.document, 'hidden', { value: false, configurable: true });
  window.HTMLElement.prototype.scrollIntoView = () => {};
  window.HTMLDialogElement.prototype.showModal = function () { this.open = true; };
  window.HTMLDialogElement.prototype.close = function () { this.open = false; };

  const intervals = new Map();
  const timeouts = new Map();
  const sceneCalls = [];
  let nextTimer = 0;
  const globals = {
    window,
    document: window.document,
    location: window.location,
    history: window.history,
    matchMedia: () => ({ matches: false, addEventListener() {}, removeEventListener() {} }),
    setInterval: (callback) => { const id = ++nextTimer; intervals.set(id, callback); return id; },
    clearInterval: (id) => intervals.delete(id),
    setTimeout: (callback) => { const id = ++nextTimer; timeouts.set(id, callback); return id; },
    clearTimeout: (id) => timeouts.delete(id),
    __youraiSceneFactory: () => Object.fromEntries(
      ['setStage', 'setPlaying', 'setSpeed', 'resetView', 'setExploded', 'setFlow', 'resize', 'dispose']
        .map((method) => [method, (...args) => sceneCalls.push({ method, args })]),
    ),
  };
  const originalDescriptors = new Map();
  for (const [name, value] of Object.entries(globals)) {
    originalDescriptors.set(name, Object.getOwnPropertyDescriptor(globalThis, name));
    Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
  }

  const restore = async () => {
    window.dispatchEvent(new window.Event('beforeunload'));
    for (const [name, descriptor] of originalDescriptors) {
      if (descriptor) Object.defineProperty(globalThis, name, descriptor);
      else delete globalThis[name];
    }
    await window.happyDOM.abort();
  };

  try {
    const source = (await readFile(new URL('../src/main.js', import.meta.url), 'utf8'))
      .replace("import './style.css';", '')
      .replace("import { createModelScene } from './scene.js';", 'const createModelScene = globalThis.__youraiSceneFactory;')
      .replace("from './simulation.js'", `from '${new URL('../src/simulation.js', import.meta.url).href}'`);
    await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
  } catch (error) {
    await restore();
    throw error;
  }

  const query = (selector) => {
    const element = window.document.querySelector(selector);
    assert.ok(element, `Expected element ${selector}`);
    return element;
  };
  const input = (selector, value) => {
    const element = query(selector);
    element.value = String(value);
    element.dispatchEvent(new window.Event('input', { bubbles: true }));
  };
  const tick = (count) => {
    assert.equal(intervals.size, 1, 'The app should have one playback timer');
    for (let i = 0; i < count; i += 1) [...intervals.values()][0]();
  };
  return { window, query, input, tick, sceneCalls, restore };
}

test('YourAI controls integrate through DOM events', async (t) => {
  const app = await mountApp();
  const { window, query: $, input, tick, sceneCalls } = app;
  const clickStage = (stage) => $(`[data-stage="${stage}"]`).click();
  try {
    await t.test('all five stages update navigation, inspector and 3D stage selection', () => {
      const names = ['数据准备', '预训练', '指令微调', '偏好对齐', '推理生成'];
      names.forEach((name, stage) => {
        clickStage(stage);
        assert.equal($('[data-stage][aria-pressed="true"]').dataset.stage, String(stage));
        assert.match($('#scene-title').textContent, new RegExp(name));
        assert.equal($('.panel-heading span').textContent, `0${stage + 1} / 05`);
        assert.equal(window.location.hash, `#stage-${stage}`);
        assert.equal(sceneCalls.filter((call) => call.method === 'setStage').at(-1).args[0], stage);
      });
    });

    await t.test('user-provided markup stays text in the token display', () => {
      clickStage(0);
      const text = '<img src=x onerror="alert(1)">';
      input('#data-text', text);
      assert.equal($('#data-text').value, text);
      assert.equal($('#token-content').querySelector('img'), null);
      assert.equal($('#token-content').querySelector('[onerror]'), null);
      assert.match($('#token-content').textContent, /img/);
      assert.match($('#token-content').innerHTML, /&lt;/);
      assert.ok(Number($('#metric-a').textContent) > 0);
    });

    await t.test('training advances by 20 steps and reset clears results', () => {
      clickStage(1);
      const initialLoss = Number($('#metric-a').textContent);
      $('#train-step').click();
      assert.equal(Number($('#metric-b').textContent), 20);
      assert.ok(Number($('#metric-a').textContent) < initialLoss);
      assert.equal($('#loss-chart polyline').getAttribute('points').split(' ').length, 21);
      $('#train-reset').click();
      assert.equal(Number($('#metric-b').textContent), 0);
      assert.equal(Number($('#metric-a').textContent), initialLoss);
      assert.match($('#loss-chart').textContent, /训练 20 步/);
    });

    await t.test('SFT mask changes target count without removing the prompt', () => {
      clickStage(2);
      const answerTargets = Number($('#metric-a').textContent);
      const prompt = $('#sft-example-text').textContent;
      $('#loss-mask').checked = false;
      $('#loss-mask').dispatchEvent(new window.Event('change', { bubbles: true }));
      assert.ok(Number($('#metric-a').textContent) > answerTargets);
      assert.equal($('#sft-example-text').textContent, prompt);
      assert.equal($('#token-content').querySelector('.muted-token'), null);
      $('#loss-mask').checked = true;
      $('#loss-mask').dispatchEvent(new window.Event('change', { bubbles: true }));
      assert.equal(Number($('#metric-a').textContent), answerTargets);
      assert.ok($('#token-content').querySelector('.muted-token'));
    });

    await t.test('DPO responds to preference reversal and PPO updates paused playback labels', () => {
      clickStage(3);
      const preferredA = Number($('#metric-a').textContent);
      $('[data-preference="1"]').click();
      assert.ok(Number($('#metric-a').textContent) > preferredA);
      assert.equal($('#margin-value').textContent, '2.0');
      $('[data-preference="0"]').click();
      assert.equal(Number($('#metric-a').textContent), preferredA);
      if ($('#play').getAttribute('aria-label') === '暂停动画') $('#play').click();
      $('[data-method="ppo"]').click();
      assert.equal($('#phase-label').textContent, '训练奖励模型');
      assert.equal($('#process-list .current h4').textContent, '训练奖励模型');
      $('#next-phase').click();
      assert.equal($('#phase-label').textContent, '在线采样与 PPO');
      assert.equal($('#process-list .current h4').textContent, '在线采样与 PPO');
      assert.equal($('#inspector').querySelector('#beta'), null);
    });

    await t.test('temperature affects probabilities, generation adds a token, and reset clears it', () => {
      clickStage(4);
      input('#temperature', 0);
      assert.equal($('#temperature-value').textContent, '0.0');
      const percentages = [...window.document.querySelectorAll('.prob-row > span:last-child')]
        .map((element) => Number.parseFloat(element.textContent));
      assert.equal(percentages.filter((value) => value === 100).length, 1);
      assert.equal(percentages.filter((value) => value !== 0 && value !== 100).length, 0);
      $('#generate').click();
      assert.equal($('#token-content .shift-label span:last-child').textContent, '1 TOKENS');
      assert.match($('#kv-note').textContent, /DECODE/);
      assert.doesNotMatch($('#generated-output').textContent, /点击生成/);
      const candidates = [...window.document.querySelectorAll('.prob-row > span:first-child')]
        .map((element) => element.textContent);
      assert.equal(new Set(candidates).size, candidates.length);
      $('#generate-reset').click();
      assert.equal($('#token-content .shift-label span:last-child').textContent, '0 TOKENS');
      assert.match($('#kv-note').textContent, /PREFILL/);
      assert.match($('#generated-output').textContent, /点击生成/);
      assert.equal($('#generate').disabled, false);
    });

    await t.test('full tour shows all three phases before advancing each of the five stages', () => {
      $('#start-tour').click();
      for (let stage = 0; stage < 5; stage += 1) {
        assert.equal($('[data-stage][aria-pressed="true"]').dataset.stage, String(stage));
        tick(49);
        assert.equal($('#playback-progress').textContent, '01 / 03');
        tick(50);
        assert.equal($('#playback-progress').textContent, '02 / 03');
        tick(50);
        assert.equal($('#playback-progress').textContent, '03 / 03');
        assert.equal($('#phase-label').textContent, $('#process-list .current h4').textContent);
        assert.ok(Number.parseFloat($('#track-fill').style.width) > 99);
        tick(1);
      }
      assert.equal($('[data-stage][aria-pressed="true"]').dataset.stage, '4');
      assert.match($('#start-tour').textContent, /播放完整流程/);
      assert.match($('#toast').textContent, /全流程导览完成/);
      assert.ok(sceneCalls.some((call) => call.method === 'setFlow' && call.args[0] === 'backward'));
      assert.ok(sceneCalls.some((call) => call.method === 'setFlow' && call.args[0] === 'cache'));
    });
  } finally {
    await app.restore();
  }
});
