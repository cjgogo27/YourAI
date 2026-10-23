# YourAI Lab · 大模型全流程交互实验室

用可旋转、可拆解的三维模型，探索一个大语言模型如何从数据中学习、接受后训练，再逐步生成回答。网站以中文解释为主，基于 **Three.js、Vite 和原生 ES Modules** 构建，可作为静态网站部署到 GitHub Pages。

- [在线实验室](https://cjgogo27.github.io/YourAI/)
- [GitHub 仓库](https://github.com/cjgogo27/YourAI)
- 维护者：[cjgogo27](https://github.com/cjgogo27)

## 五个学习阶段

| 阶段 | 观察什么 | 可以操作什么 |
| --- | --- | --- |
| 01 · 数据准备 | 筛选、去重、示意分词与训练批次 | 输入语料，调整质量阈值，切换去重，查看保留样本和 token ID |
| 02 · 预训练 | next-token 目标、前向传播、交叉熵、反向传播与参数更新 | 调整学习率和数据质量，推进示意训练，探索一次真实梯度计算 |
| 03 · 指令微调（SFT） | 对话示范、助手回答与损失掩码 | 切换指令样例，选择是否只对助手回答计入损失，播放监督更新 |
| 04 · 偏好对齐 | DPO 或基于 PPO 的 RLHF，以及独立评估与部署关卡 | 选择偏好回答，切换优化路线，调整 DPO 系数与相对概率差，查看评估说明 |
| 05 · 推理生成 | Prefill、逐 token Decode、概率分布与 KV Cache | 修改提示词，调整温度与 Top-p，单步生成，查看缓存容量估算 |

**DPO 和 PPO-based RLHF 是两条可选路线，不是必须连续执行的两个阶段。** DPO 直接从偏好回答对优化策略；典型 PPO-based RLHF 先学习奖励模型，再利用策略采样、奖励及 KL 约束更新策略。后训练后的模型仍需独立评估，检查能力、指令遵循与风险，再决定是否部署。

## 如何探索

- 点击顶部阶段导航，或选择“播放完整流程”自动走过五个阶段。
- **鼠标拖动**旋转真实三维场景，**滚轮**缩放；触屏支持单指旋转、双指缩放。
- 点击模型组件，查看注意力、前馈网络、残差、输出与缓存的解释。
- 使用“展开模型层”观察结构，使用“重置视角”回到初始位置。
- 场景获得键盘焦点后，使用**方向键**旋转、**+ / −**缩放，**Enter**轮流选择组件。
- 暂停、变速或单步推进动画；查看因果注意力热图，理解模型为何不能看到未来 token。
- 切换 1B / 7B / 70B 教学配置，观察配置参数与缓存估算的关系。
- 页面会把当前阶段写入地址 hash。例如 `#stage-0` 对应数据准备，`#stage-4` 对应推理生成；复制地址即可分享当前阶段。**链接只保存阶段，不包含完整实验状态。**
- 点击“导出本次实验”，下载 `yourai-experiment.json`，记录配置、训练示意历史、偏好设置和生成结果。导出文件可能包含你输入的提示词，请按自己的分享意愿保管。

页面适配桌面和移动设备，并尊重系统的“减少动态效果”偏好。三维显示需要支持 WebGL 的现代浏览器；不支持时会显示提示，其余学习面板仍可使用。

## 本地启动

建议使用 Node.js 22，与项目持续集成工作流保持一致。

```bash
git clone https://github.com/cjgogo27/YourAI.git
cd YourAI
npm install
npm run dev
```

打开终端中 Vite 显示的本地地址，默认通常为 `http://127.0.0.1:5173`。依赖已经锁定；复现已有环境或在 CI 中安装时可使用 `npm ci`。

```bash
# 验证计算逻辑
npm test

# 生成可部署的静态文件
npm run build

# 本地检查生产构建
npm run preview
```

构建结果位于 `dist/`。通过 HTTP 服务访问页面，不要直接双击 HTML 文件运行 ES Modules。

## 验证与 GitHub Pages 发布

网站已发布到 [https://cjgogo27.github.io/YourAI/](https://cjgogo27.github.io/YourAI/)。仓库使用 `main` 保存源码，使用 `gh-pages` 保存构建后的静态页面。

源码推送到 `main` 后，[`.github/workflows/ci.yml`](.github/workflows/ci.yml) 会自动使用 Node.js 22 安装锁定依赖，执行测试与生产构建，并上传构建产物。该工作流用于验证改动；发布网站使用下面的命令。

先确认本机 Git 已登录有权推送本仓库的账号、依赖已安装，再在项目目录执行：

```bash
npm test
npm run deploy
```

`npm run deploy` 调用 [`scripts/deploy.mjs`](scripts/deploy.mjs)，先运行 `npm run build`，再将静态产物推送到 `gh-pages`，保留该分支已有提交历史。GitHub 检测到分支更新后，会自动运行 Pages 构建与发布。可以在仓库的 **Actions** 中查看 `pages build and deployment` 的结果。

本仓库已经完成 Pages 配置。若在其他仓库首次启用，进入 **Settings → Pages → Build and deployment**，将 **Source** 设为 **Deploy from a branch**，选择 **gh-pages** 分支和 **/ (root)** 目录。发布使用本机已有的 Git 身份验证，不需要将访问令牌写进源码。

`vite.config.js` 使用相对资源路径，适配 `/YourAI/` 项目子目录。更新发布后，等待 GitHub Pages 工作流成功即可访问新版本。

## 教学模型的边界

本项目是**机制可视化与小规模数值演示**。它不下载十亿参数模型，不执行真实的大模型预训练或 PPO，也不提供通用 AI 问答。

- **三维结构**：场景中的六层是教学抽象；实际模型的层数、注意力结构、位置编码和后训练方法各不相同。1B / 7B / 70B 是教学配置，不能视为某个具体已发布模型的完整规格。
- **分词与生成**：示意分词器使用确定性教学映射，token ID 不属于真实模型词表。生成使用内置小词表、预设 logits 和可复现采样，不代表真实模型对提示词的理解。
- **训练曲线**：曲线由简化函数生成，用于解释学习率、数据质量与训练过程的关系；它们不是实测训练日志或基准结果。“真实梯度计算”演示的是小词表 logits 的交叉熵更新，并非完整 Transformer 的训练。
- **计算与估算**：softmax、Top-p、因果注意力归一化及 DPO 损失使用数学计算。KV Cache 容量按所选配置估算，不包括模型权重、激活或运行时额外开销。
- **训练与推理**：训练可并行计算序列各位置的预测损失；基础自回归推理则逐 token 生成。正常推理复用缓存，但不因此更新模型权重。
- **结果解读**：较低训练损失不等于更可靠的答案；更高温度不等于更聪明。注意力图也不是模型决策的完整因果解释。

所有实验在浏览器端完成，无需模型 API 密钥。输入不会发送给模型服务；下载 JSON 是本地操作。

## 项目结构

```text
YourAI/
├── index.html                 # 页面入口
├── src/
│   ├── main.js                # 阶段、控制台、导航与实验导出
│   ├── scene.js               # Three.js 场景与交互
│   ├── simulation.js          # 数学计算与教学模拟
│   └── style.css              # 响应式视觉样式
├── tests/                     # 计算逻辑测试
├── public/                    # 静态资源
├── scripts/
│   └── deploy.mjs             # 构建并发布至 gh-pages
├── .github/workflows/
│   └── ci.yml                 # main 源码的测试、构建与产物上传
└── vite.config.js             # 构建配置
```

## 参考资料

以下论文与官方文档用于核对机制；本项目的教学数值不来自这些论文的实验结果。

1. [The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116) — 数据过滤与去重。
2. [Hugging Face Tokenizers: The tokenization pipeline](https://huggingface.co/docs/tokenizers/en/pipeline) — 文本到 token 的处理流程。
3. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Transformer 与缩放点积注意力。
4. [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) — 基础语言模型的架构与预训练实例。
5. [Hugging Face: Causal language modeling](https://huggingface.co/docs/transformers/en/tasks/language_modeling) — next-token 目标与因果约束。
6. [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — SFT、奖励模型与 RLHF。
7. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — PPO 优化方法。
8. [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290) — DPO 的偏好优化目标。
9. [Hugging Face: How caching works](https://huggingface.co/docs/transformers/en/cache_explanation) — 自回归推理中的 KV Cache。
10. [Hugging Face: Generation](https://huggingface.co/docs/transformers/en/main_classes/text_generation) — 温度、Top-p 与生成参数。
