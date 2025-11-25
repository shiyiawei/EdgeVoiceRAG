# 🚗 EdgeVoiceRAG

<div align="center">
  <strong>基于边缘端的智能座舱RAG语音交互系统</strong>
  <br/>
  <sub>离线运行 | 低延迟响应 | 模块化设计</sub>
</div>

---

## 📖 项目简介

EdgeVoiceRAG 是一套运行在 RK3576 芯片上的全离线、端到端智能语音交互系统，专为智能座舱、机器人等边缘计算场景设计。系统集成了流式语音识别（ASR）、检索增强生成（RAG）、大语言模型推理（LLM）和语音合成（TTS）四大核心模块，实现 **5 秒内完成语音输入→智能响应→语音输出的完整闭环**。

### 核心特性

- 🎤 **流式语音识别**：集成 Zipformer 模型，VAD 静音阈值优化至 0.4s，模块延迟 ≤0.9s
- 🧠 **智能路由系统**：基于 RAG 的混合响应策略，支持紧急/事实性/复杂/创意四级查询分类
- 🚀 **高性能推理**：DeepSeek-R1-Distill-Qwen-1.5B 模型 w4a16 量化，伪流式传输使首响应时延降低 50%
- 🔊 **零延迟切换**：双缓冲队列实现 ASR-TTS 无缝衔接，消除语音卡顿
- 🔌 **松耦合架构**：ZeroMQ 实现模块间异步通信，支持独立部署与扩展

---

## 🏗️ 系统架构

```
用户语音输入
    ↓
┌─────────────────────────────────────────────────────────┐
│  ASR模块（Zipformer流式识别）                            │
│  - 麦克风音频数据采集                                    │
│  - VAD静音检测（0.4s阈值）                               │
│  - 实时语音转文字                                        │
└────────────────┬────────────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────────────┐
│  意图策略模块                                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ 紧急查询  │  │ 事实查询 │  │  复杂查询 │  │ 创意查询  │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │
└───┬──────────────┬──────────────┬──────────────┬────────┘
    │              │              │              │
    ↓              ↓              ↓              ↓
┌───────┐    ┌───────────────────────────────────────┐
│ RAG   │    │  RAG模块（座舱知识检索）                │
│ 专用  │    │  - 文本处理与向量化                     │
│ 2s    │    │  - 向量相似度检索（Top-K）              │
│       │    │  - 向量库查询                          │
└───┬───┘    └────────────┬──────────────────────────┘
    │                     │              │
    │                     ↓              ↓
    │            ┌─────────────────────────────────────┐
    │            │  DeepSeek-LLM模块                   │
    │            │  - w4a16量化推理                     │
    │            │  - NPU+2小核CPU运行                  │
    │            │  - 伪流式文本分段传输                 │
    │            │  - Prefill：233ms / Decode：7tok/s   │
    │            └──────────┬──────────────────────────┘
    │                       │
    └───────────────────────┴───────────┐
                                        ↓
                            ┌────────────────────────┐
                            │  TTS模块（SummerTTS）   │
                            │  - 伪流式语音合成       │
                            │  - 双缓冲队列处理       │
                            │  - 消息/音频异步并行    │
                            └──────────┬─────────────┘
                                       ↓
                                   语音输出
```

---

## 🎯 性能指标

| 模块 | 指标 | 性能 |
|------|------|------|
| **ASR** | 模块延迟 | ≤0.9s |
| | VAD 阈值 | 0.4s |
| **RAG** | 事实性查询响应 | ~2s |
| | 向量检索时延 | <0.5s |
| **LLM** | Prefill（自然对话） | 233ms |
| | Prefill（座舱场景） | 1100ms |
| | Decode 速度 | 7 tokens/s |
| | 资源限制性能损失 | 仅8% |
| **系统** | 端到端延迟 | <5s |
| | RAG vs 纯LLM | 提速50% |

---

## 🛠️ 技术栈

### 核心技术
- **编程语言**：C++17, Python 3.8+
- **AI框架**：RKNN-Toolkit2 (量化部署), Sherpa-onnx (推理)
- **通信协议**：ZeroMQ (REQ-REP模式)
- **构建工具**：CMake 3.18+

### 关键模型
- **ASR**：Zipformer (流式语音识别)
- **LLM**：DeepSeek-R1-Distill-Qwen-1.5B (w4a16量化)
- **TTS**：SummerTTS (离线语音合成)
- **Embedding**：中文语义向量模型 (768维)

### 硬件平台
- **芯片**：RK3576 (NPU + 8核CPU)
- **内存**：建议 ≥4GB
- **存储**：≥8GB (模型文件)

---

## 📦 快速开始

### 环境准备

```bash
# 1. 安装系统依赖
sudo apt-get update
sudo apt-get install cmake build-essential libzmq3-dev python3-dev

# 2. 安装Python依赖
pip install numpy pybind11 rknn-toolkit2

# 3. 克隆项目
git clone https://github.com/shiyiawei/EdgeVoiceRAG.git
cd EdgeVoiceRAG
```

### 模型下载

```bash
# 下载预训练模型（需要约3GB空间）
bash scripts/download_models.sh
```

### 编译运行

```bash
# 编译项目
mkdir build && cd build
cmake ..
make -j4

```

---

## 📂 项目结构

```
EdgeVoiceRAG/
├── src/
│   ├── asr/              # 语音识别模块
│   ├── rag/              # RAG检索模块
│   ├── llm/              # 大模型推理模块
│   ├── tts/              # 语音合成模块
│   └── common/           # 通用组件（ZeroMQ封装等）
├── models/               # 模型文件
├── config/               # 配置文件
├── scripts/              # 工具脚本
├── docs/                 # 文档
└── CMakeLists.txt
```

---

## 🔧 关键技术实现

### 1. VAD静音阈值优化
通过对比实验，将默认 1.2s 阈值降至 0.4s，在保证准确率的前提下显著降低首响应延迟。

### 2. 智能路由策略
```python
# 查询分类逻辑
if is_emergency(query):      # 紧急查询 → 直接返回预设答案
    return quick_response()
elif is_factual(query):      # 事实查询 → RAG专用
    return rag_search()
elif is_complex(query):      # 复杂查询 → RAG+LLM混合
    return rag_llm_hybrid()
else:                        # 创意查询 → LLM专用
    return llm_generate()
```

### 3. 伪流式传输
将 LLM 生成文本按标点符号分段，每完成一个语义单元立即传输给 TTS，首响应时延降低 50%。

### 4. 资源限制策略
使用 `taskset` 将 LLM 推理绑定到 NPU + 2 个小核 CPU，避免大核争抢，推理速度仅降低 8%。

---

## 🚀 适用场景

- 🚗 **智能座舱**：车载语音助手、车辆控制
- 🤖 **服务机器人**：导览、问答、任务执行
- 🏭 **工业设备**：无网络环境的语音交互
- 🏥 **医疗设备**：隐私敏感场景的离线问答

---

## 📊 测试数据

### 查询类型分布（测试集 500 条）
- 紧急查询：15% (响应时延 <1s)
- 事实性查询：45% (响应时延 ~2s)
- 复杂查询：30% (响应时延 ~3s)
- 创意查询：10% (响应时延 ~4s)

### 系统资源占用
- CPU 使用率：45-60%
- 内存占用：~2.8GB
- NPU 利用率：85-95%

---

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 开启 Pull Request

---

## 📄 开源协议

本项目采用 MIT 协议开源 - 详见 [LICENSE](LICENSE) 文件

---

## 👨‍💻 作者信息

**Song** - Master's Student in Advanced Computer Science, University of Birmingham

---

## 🙏 致谢

- [Sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) - 提供流式 ASR 框架
- [DeepSeek](https://www.deepseek.com/) - 开源高质量语言模型
- [RockChip](https://www.rock-chips.com/) - 提供 NPU 量化工具链

---

<div align="center">
  <sub>如果这个项目对你有帮助，请给一个 ⭐️ Star 支持！</sub>
</div>
