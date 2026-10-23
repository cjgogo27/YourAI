import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const PARTS = {
  tokens: { id: 'tokens', title: 'Token 与嵌入', description: '文本先被分词器映射为离散 token ID，再通过可训练的嵌入表变成向量。图中的小块是教学用 token，六层结构只是大型模型的缩影。', formula: 'x₀ = Embedding(token IDs)' },
  attention: { id: 'attention', title: '因果自注意力', description: '每个位置通过 Query、Key、Value 汇聚当前位置及之前位置的信息。因果掩码遮住未来 token；多头注意力能并行学习不同的关系。', formula: 'Attention(Q,K,V) = softmax(QKᵀ / √dₖ + mask)V' },
  mlp: { id: 'mlp', title: '前馈网络 · MLP', description: '注意力混合不同位置的信息；前馈网络则对每个位置的特征做非线性变换。常见现代模型使用门控前馈层，并结合归一化与残差连接。', formula: 'FFN(x) = W₂ · activation(W₁x)' },
  layer: { id: 'layer', title: 'Transformer 解码器层', description: '同一组基本计算重复堆叠：归一化、因果自注意力、残差连接和前馈网络。图中省略部分算子以便观察；层数与几何大小不代表真实参数量。', formula: 'h′ = h + Attention(Norm(h)); h″ = h′ + FFN(Norm(h′))' },
  output: { id: 'output', title: '输出头与下一个 token', description: '最后一层的隐藏状态投影成词表上的 logits，再转为概率。训练时与真实下一个 token 比较；推理时从分布中选择 token 并接回上下文。', formula: 'p(next token) = softmax(W_out · h / T)' },
  cache: { id: 'cache', title: 'KV Cache', description: '推理时缓存各层已计算的 Key 和 Value。生成新 token 时复用历史 KV，避免重复计算旧 token 的 K、V 投影；新 token 仍需关注历史上下文。', formula: 'K_cache ← [K_cache; k_new], V_cache ← [V_cache; v_new]' },
  residual: { id: 'residual', title: '残差连接与梯度回传', description: '残差连接让输入特征直接传向后续层。训练时梯度沿计算图反向传播，优化器根据梯度更新权重；普通推理只进行前向计算。', formula: 'θ ← θ − η · ∇θ L' },
};

const COLORS = { aqua: 0x4edec6, violet: 0xa193ff, orange: 0xffb06b, blue: 0x6fafff, muted: 0x284b64 };

function accessibleFallback(container, onSelect) {
  const fallback = document.createElement('div');
  fallback.className = 'scene-fallback';
  fallback.style.cssText = 'display:flex;min-height:480px;flex-direction:column;align-items:center;justify-content:center;gap:14px;color:#d9edf4;padding:32px;text-align:center';
  const heading = document.createElement('strong');
  heading.textContent = '三维渲染暂不可用 · 可继续探索计算流程';
  fallback.append(heading);
  const note = document.createElement('p');
  note.textContent = '当前浏览器未提供 WebGL。下面的组件说明与其他实验仍可使用。';
  note.style.cssText = 'max-width:340px;font-size:13px;line-height:1.8;color:#91a8bc';
  fallback.append(note);
  for (const id of ['tokens', 'attention', 'mlp', 'output', 'cache']) {
    const button = document.createElement('button');
    button.textContent = PARTS[id].title;
    button.style.cssText = 'width:260px;padding:12px;border:1px solid #315468;border-radius:8px;background:#112b3b;color:#b2f2e5;cursor:pointer';
    button.addEventListener('click', () => onSelect?.(PARTS[id]));
    fallback.append(button);
  }
  container.append(fallback);
  return { setStage() {}, setPlaying() {}, setSpeed() {}, resetView() {}, setExploded() {}, setFlow() {}, resize() {}, dispose() { fallback.remove(); } };
}

/** A real, locally rendered 3D teaching model. It never loads model weights. */
export function createModelScene(container, { onSelect } = {}) {
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'low-power' });
  } catch {
    return accessibleFallback(container, onSelect);
  }

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(39, 1, 0.1, 100);
  const initialCamera = new THREE.Vector3(11.8, 8.0, 14.4);
  camera.position.copy(initialCamera);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.8));
  renderer.setClearColor(0x061320, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.35;
  const canvas = renderer.domElement;
  canvas.style.cssText = 'display:block;width:100%;height:100%;touch-action:none;outline-offset:-4px';
  canvas.tabIndex = 0;
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', '可交互三维 Transformer 模型。拖动旋转，滚轮缩放，点击组件查看说明。键盘方向键旋转，加减号缩放，Enter 逐个查看组件。');
  container.append(canvas);
  const controls = new OrbitControls(camera, canvas);
  controls.target.set(0.5, 0.05, 0);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.enablePan = true;
  controls.minDistance = 7;
  controls.maxDistance = 31;
  controls.maxPolarAngle = Math.PI * 0.87;
  controls.minPolarAngle = 0.12;
  controls.update();

  scene.add(new THREE.AmbientLight(0xc2e5ff, 1.5));
  const keyLight = new THREE.DirectionalLight(0xb1d8ff, 3);
  keyLight.position.set(5, 9, 8);
  scene.add(keyLight);
  const edgeLight = new THREE.DirectionalLight(0x49e7c2, 2);
  edgeLight.position.set(-5, 1, -4);
  scene.add(edgeLight);

  const model = new THREE.Group();
  scene.add(model);
  const pickable = [];
  const labels = [];
  const disposable = new Set();
  const layers = [];
  const animatedNodes = [];
  const wires = [];
  const clock = new THREE.Clock();
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  let playing = !reducedMotion.matches;
  let speed = 1;
  let stage = 1;
  let flow = 'forward';
  let exploded = false;
  let expansion = 0;
  let phase = 0;
  let disposed = false;
  let frame = 0;
  let keyboardPart = 0;

  function track(resource) { disposable.add(resource); return resource; }
  const nodeGeometry = track(new THREE.SphereGeometry(0.066, 12, 8));
  const particleGeometry = track(new THREE.SphereGeometry(0.043, 8, 6));
  const plateGeometry = track(new THREE.BoxGeometry(4.3, 0.065, 2.25));
  const plateEdges = track(new THREE.EdgesGeometry(plateGeometry));
  const cubeGeometry = track(new THREE.BoxGeometry(0.64, 0.18, 0.55));

  function material(color, extra = {}) {
    return track(new THREE.MeshStandardMaterial({ color, roughness: 0.35, metalness: 0.2, ...extra }));
  }

  function choose(object, id) {
    object.userData.part = id;
    pickable.push(object);
    return object;
  }

  function label(text, color, width = 1.5, height = 0.26) {
    const bitmap = document.createElement('canvas');
    bitmap.width = 768;
    bitmap.height = 108;
    const ctx = bitmap.getContext('2d');
    ctx.font = '500 76px "Microsoft YaHei", "PingFang SC", sans-serif';
    bitmap.width = Math.ceil(ctx.measureText(text).width + 30);
    ctx.clearRect(0, 0, bitmap.width, bitmap.height);
    ctx.font = '500 76px "Microsoft YaHei", "PingFang SC", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle = color;
    ctx.fillText(text, bitmap.width / 2, 54);
    const texture = track(new THREE.CanvasTexture(bitmap));
    texture.colorSpace = THREE.SRGBColorSpace;
    const sprite = new THREE.Sprite(track(new THREE.SpriteMaterial({ map: texture, transparent: true, depthTest: false, depthWrite: false })));
    sprite.scale.set(width, height, 1);
    sprite.renderOrder = 12;
    labels.push(sprite);
    return sprite;
  }

  function line(points, color, opacity = 0.22, parent = model) {
    const geometry = track(new THREE.BufferGeometry().setFromPoints(points.map(p => Array.isArray(p) ? new THREE.Vector3(...p) : p)));
    const lineMaterial = track(new THREE.LineBasicMaterial({ color, transparent: true, opacity, depthWrite: false }));
    const result = new THREE.Line(geometry, lineMaterial);
    parent.add(result);
    return result;
  }

  function curve(from, via, to, color, opacity = 0.25, parent = model) {
    const path = new THREE.QuadraticBezierCurve3(new THREE.Vector3(...from), new THREE.Vector3(...via), new THREE.Vector3(...to));
    const result = line(path.getPoints(32), color, opacity, parent);
    return { path, line: result };
  }

  const grid = new THREE.GridHelper(22, 44, 0x224458, 0x122b3f);
  grid.position.y = -4.2;
  grid.material.transparent = true;
  grid.material.opacity = 0.24;
  track(grid.geometry);
  track(grid.material);
  scene.add(grid);

  const baseMaterial = material(0x0b2433, { transparent: true, opacity: 0.52 });
  const base = new THREE.Mesh(track(new THREE.CylinderGeometry(3.5, 3.5, 0.055, 64)), baseMaterial);
  base.position.set(0, -4.08, 0);
  model.add(base);
  const ring = new THREE.Mesh(track(new THREE.TorusGeometry(3.5, 0.008, 4, 80)), material(COLORS.aqua, { emissive: COLORS.aqua, emissiveIntensity: 0.4, transparent: true, opacity: 0.3 }));
  ring.rotation.x = Math.PI / 2;
  ring.position.y = -4.03;
  model.add(ring);

  const tokenGroup = new THREE.Group();
  tokenGroup.position.set(0, -3.4, 0);
  model.add(tokenGroup);
  const words = ['让', '模型', '理解', '世界'];
  const tokenMeshes = [];
  words.forEach((word, index) => {
    const cube = choose(new THREE.Mesh(cubeGeometry, material(COLORS.aqua, { emissive: COLORS.aqua, emissiveIntensity: 0.1, transparent: true, opacity: 0.65 })), 'tokens');
    cube.position.set(-1.35 + index * 0.9, 0, 0);
    tokenMeshes.push(cube);
    tokenGroup.add(cube);
    const caption = label(word, '#c7fff2', 0.7, 0.21);
    caption.position.copy(cube.position).add(new THREE.Vector3(0, 0.23, 0));
    tokenGroup.add(caption);
    line([[cube.position.x, 0.12, 0], [cube.position.x, 0.75, 0]], COLORS.aqua, 0.35, tokenGroup);
  });
  const inputTitle = label('TOKEN EMBEDDING', '#74bdb8', 2.3, 0.23);
  inputTitle.position.set(0, -0.45, 0.6);
  tokenGroup.add(inputTitle);

  for (let index = 0; index < 6; index++) {
    const layer = new THREE.Group();
    const y = -2.45 + index * 0.88;
    layer.position.y = y;
    layer.userData.baseY = y;
    model.add(layer);
    layers.push(layer);

    const plate = choose(new THREE.Mesh(plateGeometry, material(0x14394d, { transparent: true, opacity: 0.19, depthWrite: false })), 'layer');
    layer.add(plate);
    const outline = new THREE.LineSegments(plateEdges, track(new THREE.LineBasicMaterial({ color: index === 3 ? COLORS.aqua : COLORS.muted, transparent: true, opacity: index === 3 ? 0.7 : 0.5 })));
    layer.add(outline);
    layer.userData.outline = outline;
    const indexLabel = label(`L${String(index + 1).padStart(2, '0')}`, '#6e9aae', 0.65, 0.22);
    indexLabel.position.set(-2.65, 0.1, 0.3);
    layer.add(indexLabel);

    const attnMaterial = material(COLORS.aqua, { emissive: COLORS.aqua, emissiveIntensity: 0.35 });
    const mlpMaterial = material(COLORS.violet, { emissive: COLORS.violet, emissiveIntensity: 0.27 });
    const attentionNodes = [];
    const feedNodes = [];
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col < 3; col++) {
        const node = choose(new THREE.Mesh(nodeGeometry, attnMaterial), 'attention');
        node.position.set(-1.6 + col * 0.47, 0.14, -0.67 + row * 0.67);
        layer.add(node);
        attentionNodes.push(node.position.clone());
        const feed = choose(new THREE.Mesh(nodeGeometry, mlpMaterial), 'mlp');
        feed.position.set(0.45 + col * 0.48, 0.14, -0.67 + row * 0.67);
        layer.add(feed);
        feedNodes.push(feed.position.clone());
      }
    }
    for (let row = 0; row < 3; row++) {
      for (let col = 0; col <= row; col++) {
        line([attentionNodes[row], attentionNodes[col + 3], attentionNodes[row + 6]], COLORS.aqua, 0.22, layer);
      }
      for (let next = 0; next < 3; next++) {
        line([feedNodes[row * 3], feedNodes[next * 3 + 1], feedNodes[row * 3 + 2]], COLORS.violet, 0.22, layer);
      }
      line([[-0.6, 0.14, -0.67 + row * 0.67], [0.45, 0.14, -0.67 + row * 0.67]], COLORS.aqua, 0.28, layer);
    }
    animatedNodes.push({ material: attnMaterial, layer: index, type: 'attention' }, { material: mlpMaterial, layer: index, type: 'mlp' });
    // Invisible but generous component hit areas make the small nodes easy to inspect.
    for (const [id, x] of [['attention', -1.15], ['mlp', 0.95]]) {
      const hitbox = choose(new THREE.Mesh(track(new THREE.BoxGeometry(1.65, 0.25, 1.8)), material(0xffffff, { transparent: true, opacity: 0, depthWrite: false })), id);
      hitbox.position.set(x, 0.15, 0);
      layer.add(hitbox);
    }
  }

  const attentionLabel = label('因果注意力', '#78e5cf', 1.7, 0.3);
  attentionLabel.position.set(-1.1, 2.55, 1.4);
  model.add(attentionLabel);
  const feedLabel = label('前馈网络', '#b3a7ff', 1.6, 0.3);
  feedLabel.position.set(1.1, 2.55, 1.4);
  model.add(feedLabel);

  for (const x of [-2.0, 2.0]) {
    const wire = line([[x, -2.5, -1], [x, 2.2, -1]], COLORS.blue, 0.38);
    wires.push(wire);
    const residualTube = choose(new THREE.Mesh(track(new THREE.CylinderGeometry(0.025, 0.025, 4.7, 6)), material(COLORS.blue, { transparent: true, opacity: 0.15 })), 'residual');
    residualTube.position.set(x, -0.15, -1);
    model.add(residualTube);
  }

  const outputGroup = new THREE.Group();
  outputGroup.position.set(0, 3.05, 0);
  model.add(outputGroup);
  const probabilities = [0.65, 1.0, 0.43, 0.22, 0.14];
  probabilities.forEach((probability, index) => {
    const height = probability * 0.62;
    const bar = choose(new THREE.Mesh(track(new THREE.BoxGeometry(0.33, height, 0.34)), material(index === 1 ? COLORS.orange : COLORS.aqua, { emissive: index === 1 ? COLORS.orange : COLORS.aqua, emissiveIntensity: index === 1 ? 0.35 : 0.07, transparent: true, opacity: index === 1 ? 0.95 : 0.42 })), 'output');
    bar.position.set(-1 + index * 0.5, height / 2, 0);
    outputGroup.add(bar);
  });
  const outputTitle = label('下一个 TOKEN', '#ffcc93', 2.1, 0.27);
  outputTitle.position.set(0, 1.0, 0);
  outputGroup.add(outputTitle);
  for (let i = 0; i < 3; i++) line([[-0.6 + i * 0.6, 2.1, 0], [-0.6 + i * 0.6, 3.05, 0]], COLORS.aqua, 0.3);

  const cacheGroup = new THREE.Group();
  cacheGroup.position.set(3.75, -0.35, -0.4);
  model.add(cacheGroup);
  const cacheMaterials = [];
  for (let i = 0; i < 4; i++) {
    const blockMaterial = material(COLORS.violet, { transparent: true, opacity: 0.14, emissive: COLORS.violet, emissiveIntensity: 0.05, depthWrite: false });
    cacheMaterials.push(blockMaterial);
    const block = choose(new THREE.Mesh(track(new THREE.BoxGeometry(0.7, 0.34, 1.12)), blockMaterial), 'cache');
    block.position.y = i * 0.48;
    cacheGroup.add(block);
    const outline = new THREE.LineSegments(track(new THREE.EdgesGeometry(block.geometry)), track(new THREE.LineBasicMaterial({ color: COLORS.violet, transparent: true, opacity: 0.3 })));
    outline.position.copy(block.position);
    cacheGroup.add(outline);
    const caption = label(`K · V  ${i + 1}`, '#9ba4cd', 0.9, 0.18);
    caption.position.set(0.1, i * 0.48 + 0.03, 0.75);
    cacheGroup.add(caption);
  }
  const cacheTitle = label('KV CACHE', '#ada1ee', 1.7, 0.26);
  cacheTitle.position.set(0, 2.1, 0);
  cacheGroup.add(cacheTitle);
  const cacheConnector = curve([1.9, 0.2, -0.4], [3.0, 0.2, -0.4], [3.75, 0.2, -0.4], COLORS.violet, 0.22);
  const cacheReturn = curve([3.75, 0.7, -0.4], [3.0, 1.5, -0.4], [1.9, 1.5, -0.4], COLORS.violet, 0.18);

  const particles = [];
  for (let i = 0; i < 24; i++) {
    const color = i % 3 === 0 ? COLORS.orange : COLORS.aqua;
    const mesh = new THREE.Mesh(particleGeometry, material(color, { emissive: color, emissiveIntensity: 2 }));
    model.add(mesh);
    particles.push({ mesh, offset: i / 24, channel: i % 4 });
  }
  const cacheParticle = new THREE.Mesh(particleGeometry, material(COLORS.violet, { emissive: COLORS.violet, emissiveIntensity: 2 }));
  model.add(cacheParticle);

  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  let pointerDown = null;
  function pointerPosition(event) {
    const rect = canvas.getBoundingClientRect();
    pointer.set(((event.clientX - rect.left) / rect.width) * 2 - 1, -((event.clientY - rect.top) / rect.height) * 2 + 1);
    raycaster.setFromCamera(pointer, camera);
    return raycaster.intersectObjects(pickable, false);
  }
  function onPointerDown(event) { pointerDown = { x: event.clientX, y: event.clientY }; }
  function onPointerUp(event) {
    if (!pointerDown || Math.hypot(event.clientX - pointerDown.x, event.clientY - pointerDown.y) > 6) { pointerDown = null; return; }
    pointerDown = null;
    const hit = pointerPosition(event)[0];
    if (hit) onSelect?.(PARTS[hit.object.userData.part]);
  }
  function onPointerMove(event) {
    if (pointerDown) return;
    canvas.style.cursor = pointerPosition(event).length ? 'pointer' : 'grab';
  }
  function onKeyDown(event) {
    if (event.key === 'Enter') {
      const ids = ['tokens', 'attention', 'mlp', 'layer', 'output', 'cache', 'residual'];
      onSelect?.(PARTS[ids[keyboardPart++ % ids.length]]);
      event.preventDefault();
      return;
    }
    const offset = camera.position.clone().sub(controls.target);
    const spherical = new THREE.Spherical().setFromVector3(offset);
    if (event.key === 'ArrowLeft') spherical.theta -= 0.12;
    else if (event.key === 'ArrowRight') spherical.theta += 0.12;
    else if (event.key === 'ArrowUp') spherical.phi -= 0.1;
    else if (event.key === 'ArrowDown') spherical.phi += 0.1;
    else if (event.key === '+' || event.key === '=') spherical.radius *= 0.9;
    else if (event.key === '-') spherical.radius *= 1.1;
    else return;
    spherical.phi = THREE.MathUtils.clamp(spherical.phi, controls.minPolarAngle, controls.maxPolarAngle);
    spherical.radius = THREE.MathUtils.clamp(spherical.radius, controls.minDistance, controls.maxDistance);
    camera.position.copy(controls.target).add(new THREE.Vector3().setFromSpherical(spherical));
    controls.update();
    event.preventDefault();
  }
  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointerup', onPointerUp);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('keydown', onKeyDown);

  function resize() {
    if (disposed) return;
    const { width, height } = container.getBoundingClientRect();
    if (width < 1 || height < 1) return;
    renderer.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }
  const observer = new ResizeObserver(resize);
  observer.observe(container);
  resize();

  function animate() {
    if (disposed) return;
    frame = requestAnimationFrame(animate);
    const dt = Math.min(clock.getDelta(), 0.08);
    if (playing) phase += dt * speed;
    const targetExpansion = exploded ? 1 : 0;
    expansion = reducedMotion.matches ? targetExpansion : THREE.MathUtils.lerp(expansion, targetExpansion, Math.min(dt * 5, 1));
    layers.forEach((layer, index) => {
      layer.position.y = layer.userData.baseY + (index - 2.5) * expansion * 0.28;
      layer.children[0].material.opacity = stage === 0 ? 0.09 : 0.19;
      layer.userData.outline.material.opacity = stage === 0 ? 0.2 : index === Math.floor(phase % 6) ? 0.85 : 0.4;
    });
    tokenGroup.position.y = -3.4 - expansion * 0.6;
    outputGroup.position.y = 3.05 + expansion * 0.65;
    attentionLabel.position.y = 2.55 + expansion * 0.65;
    feedLabel.position.y = 2.55 + expansion * 0.65;
    const backward = flow === 'backward';
    const startY = -3.3 - expansion * 0.6;
    const endY = 3.3 + expansion * 0.65;
    particles.forEach(({ mesh, offset, channel }) => {
      const t = (phase * 0.18 + offset) % 1;
      const pathT = backward ? 1 - t : t;
      mesh.position.set(-1.35 + channel * 0.9, THREE.MathUtils.lerp(startY, endY, pathT), Math.sin(pathT * Math.PI * 12 + channel) * 0.28);
      mesh.visible = stage !== 0 || mesh.position.y < -2.4;
      mesh.material.color.setHex(backward ? COLORS.orange : COLORS.aqua);
      mesh.material.emissive.setHex(backward ? COLORS.orange : COLORS.aqua);
    });
    animatedNodes.forEach(({ material: nodeMat, layer, type }) => {
      const wave = Math.max(0, Math.sin(phase * 2.4 - (backward ? 5 - layer : layer) * 0.8));
      nodeMat.emissiveIntensity = stage === 0 ? 0.04 : 0.18 + wave * (type === 'attention' ? 0.85 : 0.5);
    });
    tokenMeshes.forEach((mesh, index) => { mesh.material.emissiveIntensity = stage === 0 ? 0.3 + Math.max(0, Math.sin(phase * 2 - index)) * 0.6 : 0.15; });
    const cacheActive = stage === 4 || flow === 'cache';
    cacheMaterials.forEach((mat, index) => {
      mat.opacity = cacheActive ? 0.35 : 0.1;
      mat.emissiveIntensity = cacheActive ? 0.3 + Math.max(0, Math.sin(phase * 2 - index)) * 0.45 : 0.03;
    });
    cacheConnector.line.material.opacity = cacheReturn.line.material.opacity = cacheActive ? 0.6 : 0.12;
    cacheParticle.visible = cacheActive;
    const cacheTime = (phase * 0.65) % 2;
    cacheParticle.position.copy((cacheTime < 1 ? cacheConnector.path : cacheReturn.path).getPoint(cacheTime % 1));
    controls.update();
    renderer.render(scene, camera);
  }
  animate();

  return {
    setStage(index) {
      stage = THREE.MathUtils.clamp(Number(index) || 0, 0, 4);
      flow = stage === 4 ? 'cache' : 'forward';
    },
    setPlaying(value) { playing = Boolean(value); },
    setSpeed(value) { speed = THREE.MathUtils.clamp(Number(value) || 1, 0.1, 5); },
    resetView() { camera.position.copy(initialCamera); controls.target.set(0.5, 0.05, 0); controls.update(); },
    setExploded(value) { exploded = Boolean(value); },
    setFlow(value) { if (['forward', 'backward', 'cache'].includes(value)) flow = value; },
    resize,
    dispose() {
      disposed = true;
      cancelAnimationFrame(frame);
      observer.disconnect();
      controls.dispose();
      canvas.removeEventListener('pointerdown', onPointerDown);
      canvas.removeEventListener('pointerup', onPointerUp);
      canvas.removeEventListener('pointermove', onPointerMove);
      canvas.removeEventListener('keydown', onKeyDown);
      disposable.forEach(resource => resource.dispose());
      renderer.dispose();
      canvas.remove();
    },
  };
}
