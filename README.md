# DreamRealm Qwen3.5-VL 本地训练

这是一个精简的单机训练仓库，仅保留 Qwen3.5-VL 的 DreamRealm
whole-family 训练链路。核心目标由完整回答 SFT、行方向匹配损失和列方向匹配损失组成，
不使用 rollout 或参考模型。

## 仓库结构

```text
.
├── dreamrealm/              # 数据校验、编码、目标函数、反传和检查点
├── docs/data-format.md      # family JSONL 数据格式
├── scripts/local_smoke.py   # 无需模型和 GPU 的 CPU 冒烟测试
├── tests/                   # 核心单元测试
├── train_dreamrealm.py      # 唯一训练入口：单机单卡
└── requirements.txt
```

## 安装

建议使用 Python 3.10 或 3.11。真实训练需要 Linux、CUDA GPU，以及本机可读取的
Qwen3.5-VL 模型目录。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果本机需要特定 CUDA 版本的 PyTorch，请先按 PyTorch 官方说明安装对应 wheel，
再安装其余依赖。

## 先跑本地冒烟测试

这个命令只需要 CPU，不下载模型，也不读取图片：

```bash
python scripts/local_smoke.py
python -m unittest discover -s tests -v
```

成功时第一条命令会输出 `"status": "PASS"`。

## 单卡训练

模型、数据、图片和输出都使用普通本地路径。JSONL 中的图片路径可以是绝对路径；
若使用相对路径，通过 `--media-root` 指定图片根目录。

```bash
CUDA_VISIBLE_DEVICES=0 python train_dreamrealm.py \
  --model /models/Qwen3.5-9B \
  --data /datasets/dreamrealm/families.jsonl \
  --media-root /datasets/dreamrealm \
  --output /checkpoints/dreamrealm-run-001 \
  --epochs 3 \
  --families-per-step 1 \
  --lr 5e-6 \
  --warmup-ratio 0.03 \
  --lr-scheduler cosine \
  --save-optimizer
```

训练默认冻结视觉 patch embedding 和视觉 blocks，训练语言模型与多模态投影层；优化器
主权重和状态保存在 CPU，以降低单卡显存占用。输出目录必须不存在，避免覆盖已有结果。

在正式训练前，可先验证完整数据和图像编码：

```bash
python train_dreamrealm.py \
  --model /models/Qwen3.5-9B \
  --data /datasets/dreamrealm/families.jsonl \
  --media-root /datasets/dreamrealm \
  --output /tmp/dreamrealm-preflight \
  --preflight-only
```

数据字段与 edited/no-edit 两种 family 结构见
[docs/data-format.md](docs/data-format.md)。
