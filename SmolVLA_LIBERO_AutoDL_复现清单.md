# SmolVLA + LIBERO 全流程落地清单（AutoDL / RTX 4090D 24GB）

> 目标：从一台空白 AutoDL 实例开始，用公共数据集把 `lerobot/smolvla_base` 微调成 LIBERO 多任务策略，并在 LIBERO 仿真里跑出可复现的成功率。
> 版本基线：**LeRobot v0.6.1（2026-08 发布，稳定 tag） + hf-libero 0.1.4 + Python 3.12 + PyTorch 2.7/2.8 (CUDA 12.8)**
> 适用硬件：**1 × RTX 4090D 24GB**（单卡，按量计费）
> 本地端：**Ubuntu 或 Windows 皆可**，每个本地操作都给两套终端命令；服务器端：Ubuntu（AutoDL）
> 说明：所有命令按“复制即可执行”写；`{}` 表示需要你替换的内容。
> 若某个参数报 `unrecognized arguments`，先 `lerobot-train --help | grep 关键词` 确认该 tag 的写法——0.5→0.6 改过若干参数名（如 `eval_freq` → `env_eval_freq`），**以你装的那个 tag 的** **`--help`** **为准**。

***

## 0. 先定验收标准（Definition of Done）

不要用“跑起来了”当成功。按下面 6 个里程碑验收，每达成一个再往下走。

| 里程碑    | 内容      | 验收标准（硬指标）                                                         | 预计耗时            | 预计花费      |
| ------ | ------- | ----------------------------------------------------------------- | --------------- | --------- |
| **M0** | 机器与环境就绪 | `egl_probe` 能列出 GPU 设备；`lerobot-train --help` 正常                  | 1.5–2 h         | \~4 元     |
| **M1** | 模型与数据就绪 | 4 个仓库落地且校验通过（数据集打印 `episodes=1693 / frames=273465 / tasks=40`）    | 0.5–1 h（含下载/上传） | \~1–2 元   |
| **M2** | 链路冒烟    | 300 步训练不报错 + 单任务 3 episode 评测能跑完并打印成功率                            | 1 h             | \~2 元     |
| **M3** | 小试收敛    | 20k 步后 `libero_spatial` 成功率 **≥ 55%**                             | 3–4 h           | \~8 元     |
| **M4** | 正式训练    | 80k 步后四套平均 **≥ 80%**（论文 87.3%，单卡 78–86% 属正常）                      | 10–16 h         | \~25–35 元 |
| **M5** | 结果固化    | 4 套 × 10 任务 × 10 episode = 400 episode 的完整表 + checkpoint + 一键复现脚本 | 1.5–3 h         | \~5 元     |

**总计：约 18–27 小时机时，40–55 元（4090D 按 1.98 元/h 估算，会员价更低）。**

> 达标线建议：平均 ≥80% 算“跑通”，≥85% 算“复现良好”。低于 70% 不要急着加步数，先查数据/评测设置（见 §10）。

***

## 1. 开工前必须建立的 8 个认知

### 1.1 你训的到底是什么

| 项                              | 数值                                                      | 说明                                                |
| ------------------------------ | ------------------------------------------------------- | ------------------------------------------------- |
| 总参数                            | \~450M                                                  | VLM 骨干 `SmolVLM2-500M-Video-Instruct` 取**前 16 层** |
| 动作专家                           | \~100M                                                  | Flow Matching，10 步去噪出一个 action chunk              |
| 视觉 token                       | 每帧 **64** 个                                             | pixel-shuffle 降采样，不是 image tiling                 |
| chunk\_size / n\_action\_steps | 50 / 50                                                 | 一次预测 50 步动作                                       |
| state / action 维度              | 补零到 **32**                                              | 所以 8 维 state、7 维 action 都能直接喂                     |
| 输入分辨率                          | 512×512（padding）                                        | 原始 256×256 会被放大                                   |
| 默认微调策略                         | `train_expert_only=True` + `freeze_vision_encoder=True` | 实际只训 \~100M，batch=8 仅约 **5GB** 显存                 |

**这是全篇最重要的一句话**：SmolVLA 默认只训动作专家，24GB 卡不仅够用，而且很宽裕。网上“4090 跑不动 VLA”的说法，说的是 π0/π0.5（3B+）和 GR00T，不是 SmolVLA。

### 1.2 LIBERO 是什么形状的数据

- 4 个标准套件：`libero_spatial` / `libero_object` / `libero_goal` / `libero_10`（LIBERO-Long），每套 10 个任务。
- `lerobot/libero` 数据集：**1693 episodes / 273,465 frames / 40 tasks**，含以上 4 套。
- 每条样本：2 路相机（`observation.images.image` 主视角 + `observation.images.image2` 腕部）、8 维 state、7 维 action（6 维末端增量 + 1 维夹爪）、1 条英文语言指令、20Hz。
- 所以你要做的是**40 任务的多任务模仿学习**，不是 lifelong learning，也不是“一个任务一个模型”。

### 1.3 三条可选路线，先选对再动手

| 路线                 | 命令要点                                                   | 可训练参数                | 24GB 可行性       | 预期                 | 建议          |
| ------------------ | ------------------------------------------------------ | -------------------- | -------------- | ------------------ | ----------- |
| **A 论文口径**         | `--policy.type=smolvla --policy.load_vlm_weights=true` | 专家 + state/action 投影 | 很轻松            | 论文报 87.3%          | 想严格对标论文时用   |
| **B 官方推荐（本清单主路线）** | `--policy.path=lerobot/smolvla_base`                   | 默认 \~100M            | 很轻松            | 收敛更快，80%+          | ✅ 新手首选      |
| **C 放开 VLM**       | 再加 `--policy.train_expert_only=false`                  | \~306M / 450M        | batch 需降到 8–16 | 有社区实测“提升不明显”       | 只在 B 卡在瓶颈时试 |
| **D LoRA**         | `--peft.method_type=LORA --peft.r=64`                  | 极少                   | 最省             | 官方有示例，LIBERO 上未证更优 | 显存极紧时用      |

> ⚠️ 路线 B 必须加 `--policy.input_features=null`，否则会去加载 base 模型自带的相机名（`camera1/2/3`）而与 LIBERO 的 `image/image2` 冲突，直接 `KeyError`。这是最高频的坑之一。

### 1.4 为什么用 `lerobot/libero` 而不是另一个

| 数据集                            | 大小         | 相机存储           | 结论                   |
| ------------------------------ | ---------- | -------------- | -------------------- |
| **`lerobot/libero`**           | **1.9 GB** | MP4 视频         | ✅ 推荐，下载快 37 倍，加载速度相当 |
| `HuggingFaceVLA/libero`        | 69.9 GB    | PNG in parquet | 仅在装不上视频解码后端时用        |
| `physical-intelligence/libero` | —          | 原始格式           | 需要自己转 key，不建议新手碰     |

### 1.5 训练量级心里有数

- 官方参考：20k 步 ≈ 4 h / 单 A100（batch 64）。4090D 约为 A100 的 60–75% 吞吐。
- SmolVLA 默认 cosine 调度 `decay_steps=30000`；跑 80k 步时学习率会在 30k 后贴地，**这正常**（官方默认就是这么设计的）。
- 想让 LR 全程有效衰减，加 `--policy.scheduler_decay_steps=80000`。

### 1.6 不要做的事

- ❌ 不要自己采数据（现阶段没有 SO-101，LIBERO 数据就是你的“公共数据集”）。
- ❌ 不要一开始就 100k 步。先 300 步验证链路，再 20k 验证收敛，最后长跑。
- ❌ 不要开训练中的在线仿真评测（`--env_eval_freq` 默认 20000），会抢显存、拖慢训练，本流程统一设为 0。
- ❌ 不要把 HuggingFace 缓存留在系统盘（30GB 会被撑爆）。

### 1.7 关键路径与依赖关系

```
租机 → 磁盘/软链 → 镜像源/加速 → conda+torch → apt EGL 库 → lerobot(v0.6.1) → 数据 1.9GB
                                                                        ↓
                                          冒烟训练 300 步 → 单任务评测 → 20k 步 → 四套评测 → 80k 步 → 最终评测
```

### 1.8 时间/成本预算表（按 4090D 1.98 元/h）

| 阶段              | 机时         | 费用         |
| --------------- | ---------- | ---------- |
| 环境安装            | 2 h        | 4 元        |
| 数据下载            | 0.5 h      | 1 元        |
| 冒烟 + 调试         | 2 h        | 4 元        |
| 20k 步训练         | 4 h        | 8 元        |
| 80k 步训练         | 13 h       | 26 元       |
| 评测（400 episode） | 2 h        | 4 元        |
| **合计**          | **\~24 h** | **\~47 元** |

> 数据盘 100GB 另计，约 0.01 元/GB/日（以控制台为准），一个月几块钱。

***

## 2. AutoDL 租机：最佳配置（照这张表选）

### 2.1 配置清单

| 项目         | 选择                                              | 理由 / 注意                                                           |
| ---------- | ----------------------------------------------- | ----------------------------------------------------------------- |
| **GPU**    | **RTX 4090D 24GB × 1**                          | 约 **1.98 元/h**；24GB 对 SmolVLA 默认微调（\~5–14GB）绰绰有余                  |
| 备选 GPU     | RTX 4090 24GB（约 2.19 元/h）                       | 算力略高、库存通常更多；贵 10%，急用时选它                                           |
| 不建议        | A100 40GB / H800                                | 单卡 5–15 元/h，本项目用不上；除非要 batch≥128 或跑 π0.5                          |
| **CPU/内存** | 选 **≥12 核 / ≥48GB** 的主机                         | AutoDL 按 8核/32GB 每卡分配；DataLoader 要 8 workers，核多更稳                 |
| **镜像**     | **基础镜像：PyTorch 2.7+ / CUDA 12.8 / Python 3.12** | 关键：Python 必须 3.12（lerobot 0.6.x 要求 ≥3.12），CUDA 12.8 匹配官方 torch 轮子 |
| **系统盘**    | 30GB（默认，免费）                                     | 只放代码和环境；缓存必须挪走                                                    |
| **数据盘**    | **扩到 100GB**                                    | 默认 50GB 偏紧：数据 2GB + 资产 \~5GB + 10 个 ckpt×1.8GB + 日志/视频            |
| **计费**     | 按量计费                                            | 关机即停费；跑完立刻关机                                                      |
| **地区**     | 选 4090D **空闲数多**的区（西北/内蒙/重庆等）                   | 建好别换区，数据盘绑定地区                                                     |
| **网络**     | 开学术加速                                           | `source /etc/network_turbo`                                       |

> 为什么不是 3090：3090 24GB 便宜（1.66 元/h）但没有 bf16 加速，SmolVLA 用 AMP/bf16 时 4090D 快 40% 以上，总成本反而更低。

### 2.2 控制台操作顺序（别搞反）

> 这一节只能网页操作（AutoDL 没有开机器的 CLI）；从 §3 起全部是终端命令。

1. 注册 → **学生认证**（有会员折扣）→ 充值 100 元（够跑完整个项目）。
2. 算力市场 → 筛选 `RTX 4090D` → 选空闲数 ≥2 的主机 → 选镜像（见上表）→ **数据盘拉到 100GB** → 创建。
3. 实例创建后会**自动开机并开始计费**。第一件事：`nvidia-smi` 确认卡在。
4. 记下 SSH 登录指令（形如 `ssh -p 38xxx root@connect.xxx.gpuhub.com`），本机用 MobaXterm / VS Code Remote / 终端连上。
5. **不要频繁关机再开**：关机后 GPU 不预留，可能被别人占走，再开提示“GPU 不足”。一次开机干完一个阶段最省事。

### 2.3 省钱四招

| 招式            | 做法                                               | 省多少            |
| ------------- | ------------------------------------------------ | -------------- |
| 无卡模式          | 关机后用“无卡模式开机”（0.5 核 / 2GB / 0.1 元/h）做**纯下载、整理文件** | 下载 2 小时省 3.8 元 |
| 定时自动关机        | 用 §5.7 条件关机脚本（训练+评测都成功才 shutdown）                | 防止跑完空转一整夜      |
| 分级训练          | 300 步 → 20k → 80k，不要一上来长跑                        | 避免 30 元级浪费     |
| checkpoint 复用 | `--save_freq=10000`（80k 全程 8 个 ≈ 40G），中断可续       | 避免重跑、也避免写爆磁盘   |

> ⚠️ 无卡模式只有 2GB 内存，**不要**在里面 `pip install torch` 或编译 `egl_probe`，会 OOM。它只适合下载。

***

## 3. 实例初始化（服务器端 Ubuntu，一次做完）

> **终端约定**：服务器（AutoDL）只有 Ubuntu 一种终端。Windows 用户先 `ssh -p 端口 root@主机` 连上去，敲的还是下面这些命令——本节起所有代码块都是 Ubuntu，注释即“这句在干什么”。
> 需要 Windows 本地命令的地方（只有 §5.4 离线下载），会给「Ubuntu / Windows」两套。

### 3.1 先体检

```bash
nvidia-smi                        # 看卡在不在：应出现 RTX 4090D，24564MiB
nproc                             # CPU 核数，≥12 时 DataLoader 才不憋屈
free -g                           # 内存，≥48GB 比较稳
df -h / /root/autodl-tmp          # 系统盘 30G / 数据盘 100G，别搞反
python -V                         # 期望 3.12.x（lerobot 0.6 硬性要求 ≥3.12）
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
                                  # 期望输出 2.7.x 12.8 True 三件套
```

| 命令           | 正常结果                                | 不正常怎么办                        |
| ------------ | ----------------------------------- | ----------------------------- |
| `nvidia-smi` | 有 4090D、显存 24564MiB、驱动 570+         | 驱动太老就换镜像；无卡就是没分配到             |
| `nproc`      | ≥12                                 | 8 核也能跑，但 `num_workers` 只能开到 6 |
| `df -h`      | `/` 约 30G、`/root/autodl-tmp` 约 100G | 数据盘没扩容 → 控制台扩容（要重启）           |
| `python -V`  | 3.12.x                              | 是 3.10/3.11 → 走 §4.2 情况 B     |
| torch 三件套    | 2.7+ / 12.8 / True                  | `False` 说明 CUDA 轮子与驱动不匹配      |

### 3.2 目录规划 + 缓存迁移（必做）

```bash
mkdir -p /root/autodl-tmp/{hf_cache,libero_cache,outputs,logs,src}
# 一次性建好 5 个目录：HF 缓存 / LIBERO 资产缓存 / 输出 / 日志 / 源码
# 四个仓库（smolvla_base / SmolVLM2 / libero 数据 / libero-assets）统一在 SmolVLA/ 下，不再单建 models/ datasets/

mkdir -p /root/.cache
# 确保 ~/.cache 存在，否则下面的软链会失败

ln -sfn /root/autodl-tmp/hf_cache     /root/.cache/huggingface
# -s 建软链，-f 覆盖已有，-n 把软链当文件而不是目录；把 HF 缓存指到数据盘

ln -sfn /root/autodl-tmp/SmolVLA/libero-assets /root/.cache/libero
# LIBERO 仿真运行时会去 ~/.cache/libero 找资产，直接软链到 SmolVLA/libero-assets（资产已在本地）
# 如果 hf-libero 期望子目录层级，再改为 ln -sfn /root/autodl-tmp/SmolVLA /root/.cache/libero

ls -l /root/.cache
# 确认两行都是 “huggingface -> /root/autodl-tmp/...”，箭头方向别反
```

> 为什么必须做：`smolvla_base` 1–2GB、SmolVLM2 基座 \~1GB、LIBERO 资产 1–3GB、数据集 1.9GB，全堆系统盘 30GB 必爆 `No space left on device`。

### 3.3 环境变量（写入 .bashrc，重连自动生效）

```bash
cat >> ~/.bashrc <<'EOF'

# ===== SmolVLA / LIBERO =====
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
EOF
# 追加写入 ~/.bashrc；用 'EOF' 加引号，防止 $ 被当前 shell 提前展开

source ~/.bashrc
# 让当前这个 shell 立即生效；以后新开窗口会自动读取

env | grep -E "HF_|MUJOCO|PYOPENGL"
# 逐条核对 6 个变量是否真的进来了
```

| 变量                          | 作用             | 不设会怎样                      |
| --------------------------- | -------------- | -------------------------- |
| `HF_ENDPOINT`               | 走 hf-mirror 镜像 | 直连 huggingface.co 超时，权重下不来 |
| `HF_HOME`                   | 缓存根目录          | 默认写系统盘，30GB 撑爆             |
| `HF_HUB_ENABLE_HF_TRANSFER` | 多线程下载          | 慢 3–5 倍（需装 `hf_transfer`）  |
| `MUJOCO_GL=egl`             | 无头渲染后端         | `GLFW error` / 需要显示器       |
| `PYOPENGL_PLATFORM=egl`     | OpenGL 绑定层对齐   | 偶发 `EGL_NOT_INITIALIZED`   |
| `TOKENIZERS_PARALLELISM`    | 关分词器多进程告警      | 满屏 warning，不影响结果           |

### 3.4 pip 源 + 下载加速

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# 全局换清华源，装包快 10 倍

pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
# 允许这个 http 源，否则新版 pip 会拒

pip install -U pip setuptools wheel hf_transfer
# 升级构建工具链 + 装 HF 多线程下载器
```

### 3.5 网络加速（GitHub / HF 慢时）

```bash
source /etc/network_turbo
# 开 AutoDL 学术加速，只对当前 shell 有效

env | grep -i proxy
# 确认 http_proxy / https_proxy 已注入

unset http_proxy https_proxy
# 用完关掉，否则会影响正常网络（下国内源反而变慢）
```

## 4. 安装环境（最容易翻车的一节，逐条核对）

### 4.1 系统依赖（EGL 渲染 + 编译链）

```bash
apt-get update
# 刷新软件包索引，装新包前必做

apt-get install -y --no-install-recommends \
  build-essential cmake ninja-build pkg-config git curl \
  libegl1 libgl1 libglib2.0-0 libglew-dev libosmesa6-dev \
  libegl1-mesa-dev libgles2-mesa-dev libgl1-mesa-dev
# 第 1 行：编译链 + 构建工具 + 下载工具
# 第 2 行：EGL/GL 运行库（MuJoCo 离屏渲染必需）与 OSMesa 兜底
# 第 3 行：EGL/GL 开发头文件（编 egl_probe 时要用）
# --no-install-recommends 少装无关推荐包，省系统盘
```

- `libegl1` + `libegl1-mesa-dev`：缺了报 `libEGL.so.1: cannot open shared object file`。
- `libosmesa6-dev`：EGL 彻底不通时的兜底（`MUJOCO_GL=osmesa`，慢 5–10 倍，只用于排查）。
- `cmake` + `build-essential`：编译 `hf-egl-probe` 用。

### 4.2 Python 环境（二选一）

**情况 A：镜像自带 Python 3.12 + torch≥2.7 + CUDA 12.8（推荐，直接用）**

```bash
python -V
# 看是不是 3.12.x

python -c "import torch; print(torch.__version__, torch.version.cuda)"
# 看是不是 2.7+ 与 12.8；两个都满足就跳过本节
```

**情况 B：镜像不满足（如 Python 3.10）**

```bash
conda create -y -n lerobot python=3.12
# 新建独立环境，别在 base 里折腾，坏了直接删

conda activate lerobot
# 之后每条命令都要先激活；忘了激活会报 command not found

pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
# 指定 CUDA 12.8 的官方轮子；直接 pip install torch 会装成 cu130，驱动不认
```

> 驱动版本必须 ≥ 570 才能上 cu128 轮子。用 `nvidia-smi` 右上角 `CUDA Version` 确认。

### 4.3 安装 LeRobot（固定 tag，不用 main）

```bash
cd /root/autodl-tmp/src
# 源码放数据盘，别占系统盘

git clone --depth 1 --branch v0.6.1 https://github.com/huggingface/lerobot.git
# --depth 1 只拉最新一次提交（快）；--branch 锁定 tag，避免 main 漂移

cd lerobot
# 进仓库根目录，后面 pip 的 . 指这里

pip install --no-build-isolation "hf-egl-probe==1.0.2"
# 先单独编 EGL 探测包；关掉构建隔离，才能复用系统的 cmake

pip install --no-build-isolation -e ".[training,smolvla,libero]"
# -e 可编辑安装，改源码立刻生效
# 三个 extra：训练脚本 / SmolVLA 策略 / LIBERO 仿真
```

**为什么是这三个 extra**：

| extra      | 提供                                         | 缺了会怎样                          |
| ---------- | ------------------------------------------ | ------------------------------ |
| `training` | datasets / accelerate / wandb / torchcodec | 没有 `lerobot-train`             |
| `smolvla`  | transformers 5.4–5.6 / num2words           | `Unknown policy type: smolvla` |
| `libero`   | hf-libero（robosuite 1.4 / mujoco / bddl）   | `env.type=libero` 不可用          |

> `hf-libero` 只在 Linux 安装（`sys_platform=='linux'`），AutoDL 是 Linux，没问题。
> **不要**用 `pip install lerobot`：0.6 起 PyPI 包不含训练与数据集依赖。

### 4.4 ffmpeg（视频解码后端）

`lerobot/libero` 用 MP4 存帧，靠 torchcodec 解码，torchcodec 需要 ffmpeg 7.x 动态库。

```bash
conda install -y -c conda-forge ffmpeg=7.1.1
ffmpeg -version | head -1     # 期望 7.1.1
```

若没用 conda（情况 A 直接用镜像 base 环境），也要确保镜像里的 ffmpeg 是 7.x：

```bash
ffmpeg -version | head -1
# 若 <7：conda install -y -c conda-forge ffmpeg=7.1.1  (会装进当前 conda 环境)
```

### 4.5 五项验证（全过才继续）

```bash
# 1) 包能导入
python -c "import torch, lerobot, libero, robosuite, mujoco; print('imports ok')"

# 2) CUDA 可用
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

# 3) EGL 渲染设备可见（最关键的一步）
python -m egl_probe.get_available_devices
# 期望：Device 0 is available for rendering

# 4) 命令行入口存在
lerobot-train --help | head -20
lerobot-eval  --help | head -20

# 5) 版本对齐
pip list | grep -Ei "lerobot|torch|transformers|mujoco|robosuite|hf-libero|torchcodec"
```

期望关键版本：

```
lerobot        0.6.1
torch          2.7.x / 2.8.x
transformers   5.4.x - 5.6.x
mujoco         3.x
robosuite      1.4.0
hf-libero      0.1.4
torchcodec     0.3-0.11
```

### 4.6 安装期报错速查

| 报错关键词                                                          | 原因                      | 处理                                                       |
| -------------------------------------------------------------- | ----------------------- | -------------------------------------------------------- |
| `Command '['cmake', '--version']' returned non-zero`           | 构建隔离环境里没有 cmake         | 加 `--no-build-isolation`；先 `apt install cmake`           |
| `libEGL.so.1: cannot open shared object file`                  | 缺 EGL 运行库               | `apt install libegl1 libegl1-mesa-dev`                   |
| `Unknown policy type: smolvla`                                 | 没装 `[smolvla]`          | `pip install -e ".[smolvla]"`                            |
| `No module named 'libero'`                                     | 没装 `[libero]` / 非 Linux | 确认 `.[libero]`，确认系统是 Linux                               |
| `Could not load libtorchcodec`                                 | torchcodec 与 torch 2.8.0+cu128 不兼容（本机实测 0.8/0.8.1/0.9.1/0.11.1 加载即 std::bad_alloc 崩溃）             | `pip uninstall torchcodec && pip install av`，全清单统一 `--dataset.video_backend=pyav`（lerobot 自动将 pyav 设为默认解码器）                             |
| `A module compiled using NumPy 1.x cannot be run in NumPy 2.x` | torch/numpy ABI 错配      | 重装匹配的 torch；或 `pip install "numpy<2"` 前先确认 lerobot 要求 ≥2 |
| `torch.cuda.is_available() == False`                           | 驱动/轮子 CUDA 版本不匹配        | 看 `nvidia-smi` 的 CUDA Version，重装对应 cu 轮子                 |

***

## 5. 获取模型与数据（两条路：服务器直下 / 本地下再上传）

### 5.1 先决策：走哪条

| 你的情况                             | 推荐                 | 原因                    |
| -------------------------------- | ------------------ | --------------------- |
| AutoDL 到 hf-mirror 速度正常（>5 MB/s） | **方案 A：服务器直下**     | 少一次上行，命令最少，5 分钟搞定     |
| 本地有代理 / 已经下好了 / 需要断点续传           | **方案 B：本地下再上传**    | 可控、可校验、可复用            |
| 家里上行 <20 Mbps                    | 方案 A               | 2GB 传上去要 15 分钟以上，反而更慢 |
| 要复用到多台实例 / 长期留存                  | 方案 B + AutoDL 文件存储 | 一次上传，同地区多机共享          |

> 结论：**默认走 A**；只有“服务器下载老失败”或“你本地已有文件”时才走 B。B 不是更优解，只是更可控。

### 5.2 要下哪 4 个仓库（第 2 个最容易漏）

| 用途                  | 仓库                                           | 大约体积   | 实际落地路径（4 仓库统一在 SmolVLA/ 下）                              |
| ------------------- | -------------------------------------------- | ------ | ------------------------------------------------------- |
| 策略权重                | `lerobot/smolvla_base`                       | 1–2 GB | `/root/autodl-tmp/SmolVLA/smolvla_base`                 |
| **VLM 基座**（实例化时还要用） | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | \~1 GB | `/root/autodl-tmp/SmolVLA/SmolVLM2-500M-Video-Instruct` |
| 训练数据                | `lerobot/libero`（dataset）                    | 1.9 GB | `/root/autodl-tmp/SmolVLA/libero`                       |
| 仿真资产                | `lerobot/libero-assets`（dataset）             | 1–3 GB | `/root/autodl-tmp/SmolVLA/libero-assets`                |

> 第 2 行是最容易漏的：`smolvla_base` 的 `config.json` 里 `vlm_model_name` 写的是基座仓库名，**实例化模型时会再去 Hub 拉一次**。离线场景下必须显式指到本地目录，否则前功尽弃。
>
> **📂 当前项目目录结构**：4 个仓库统一放在 `/root/autodl-tmp/SmolVLA/` 下，不再分散到 `models/`、`datasets/`、`libero_cache/assets/`。本清单后续所有路径均以此为准。

### 5.3 方案 A：服务器直接下载（推荐）

```bash
source /etc/network_turbo
# 可选：开学术加速；hf-mirror 一般不用也能跑

export HF_ENDPOINT=https://hf-mirror.com
# 当前 shell 生效；已在 .bashrc 里的话这句可省

hf download lerobot/smolvla_base \
  --local-dir /root/autodl-tmp/SmolVLA/smolvla_base
# 下策略权重（1–2GB），--local-dir 指定平铺目录，不用 HF 缓存那套软链

hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --local-dir /root/autodl-tmp/SmolVLA/SmolVLM2-500M-Video-Instruct
# 下 VLM 基座（~1GB）；最容易漏的一个，实例化时还会用它

hf download lerobot/libero \
  --repo-type=dataset \
  --local-dir /root/autodl-tmp/SmolVLA/libero
# 下训练数据（1.9GB）；--repo-type=dataset 不能少，否则当成模型仓库

hf download lerobot/libero-assets \
  --repo-type=dataset \
  --local-dir /root/autodl-tmp/SmolVLA/libero-assets
# 下仿真资产（1–3GB）；也可留给第一次 eval 自动下

du -sh /root/autodl-tmp/SmolVLA/{smolvla_base,SmolVLM2-500M-Video-Instruct,libero,libero-assets}
# 核对体积，和上面表格对得上才继续
```

期望落地结构：

```
/root/autodl-tmp/SmolVLA/libero/         meta/  data/  videos/
/root/autodl-tmp/SmolVLA/smolvla_base/   config.json  model.safetensors  ...
/root/autodl-tmp/SmolVLA/libero-assets/   articulated_objects/ ... scenes/
```

固定 revision（可复现）：

```bash
hf download lerobot/libero --repo-type=dataset --revision main
# 先看 main 当前指向哪个 commit，再去网页取 sha

# 训练时加：--dataset.revision=<commit-sha>
# 数据集被重传过也能对上同一份数据
```

> 数据集被重新上传过，成功率就只能跟同一 revision 对比。写实验记录时把 sha 记下来。

### 5.4 方案 B：本地下再上传（Ubuntu / Windows 双套）

**适用**：服务器直下反复失败 / 你本地已有文件 / 要在多台实例间复用。
**代价**：多一次家庭宽带上行。2GB 在 30 Mbps 上行约 10 分钟，10 Mbps 约 30 分钟。

两种本地终端只有 5 处不一样，其余命令逐字相同：

| 事项    | Ubuntu                   | Windows（PowerShell）                    |
| ----- | ------------------------ | -------------------------------------- |
| 设环境变量 | `export HF_ENDPOINT=...` | `$env:HF_ENDPOINT = "..."`             |
| 本地目录  | `~/smolvla`              | `D:\smolvla`                           |
| 算哈希   | `sha256sum x.tar`        | `Get-FileHash x.tar -Algorithm SHA256` |
| 断点续传  | `rsync -avP`             | 无内置 → WSL 里用 rsync，或 WinSCP            |
| 路径分隔符 | `/`                      | `\`（PowerShell 也认 `/`，混用不报错）           |

#### 5.4.1 本地准备

**Ubuntu 本地**

```bash
python3 --version                  # 期望 3.10+；没有就 sudo apt install python3 python3-pip
python3 -m pip --version           # 确认 pip 可用
python3 -m pip install -U "huggingface_hub[cli]" hf_transfer   # 装 hf 命令 + 多线程下载器
hf --version                       # 验证 hf 能跑；报错说明 Scripts/bin 不在 PATH
which ssh scp tar rsync            # 四个都要有；rsync 缺了就 sudo apt install rsync
mkdir -p ~/smolvla                 # 建工作目录
df -h ~/smolvla                    # 看剩余空间，要求 ≥15GB
```

**Windows 本地**

```powershell
python --version                   # 期望 3.10+；没有就去 python.org 装并勾选 Add to PATH
pip --version                      # 确认 pip 可用
pip install -U "huggingface_hub[cli]" hf_transfer   # 装 hf 命令 + 多线程下载器
hf --version                       # 验证；报“无法识别”见 §9 G2
Get-Command ssh, scp, tar | Select-Object Name, Source   # 三个都应指向 System32
New-Item -ItemType Directory -Force D:\smolvla | Out-Null  # 建工作目录
Get-PSDrive D | Select-Object @{n='Free(GB)';e={[math]::Round($_.Free/1GB,1)}}  # 剩余空间 ≥15GB
```

> Windows 若 `scp` 缺失，用**管理员** PowerShell 装一次：`Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0`。
> 目录三条铁律：**不在 OneDrive 同步目录、路径不含中文、剩余 ≥15GB**（OneDrive 占位符会传成 0 字节）。

#### 5.4.2 下载 4 个仓库

**Ubuntu 本地**

```bash
export HF_ENDPOINT=https://hf-mirror.com      # 走国内镜像，避免直连超时
export HF_HUB_ENABLE_HF_TRANSFER=1            # 开多线程下载（配合 hf_transfer）
cd ~/smolvla                                  # 统一在 4 个仓库的父目录下操作

hf download lerobot/smolvla_base --local-dir ./smolvla_base
# 策略权重；--local-dir 生成平铺目录（真实文件），不是 HF 那套 blobs+snapshots

hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct --local-dir ./SmolVLM2-500M-Video-Instruct
# VLM 基座；最容易被漏掉的一个

hf download lerobot/libero --repo-type=dataset --local-dir ./libero
# 训练数据；--repo-type=dataset 必须写

hf download lerobot/libero-assets --repo-type=dataset --local-dir ./libero-assets
# 仿真资产；文件多但都要
```

**Windows 本地**

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"    # 走国内镜像（只对当前窗口有效）
$env:HF_HUB_ENABLE_HF_TRANSFER = "1"          # 开多线程下载
Set-Location D:\smolvla                       # 统一在父目录下操作

hf download lerobot/smolvla_base --local-dir .\smolvla_base
# 策略权重

hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct --local-dir .\SmolVLM2-500M-Video-Instruct
# VLM 基座

hf download lerobot/libero --repo-type=dataset --local-dir .\libero
# 训练数据

hf download lerobot/libero-assets --repo-type=dataset --local-dir .\libero-assets
# 仿真资产
```

> 想长期生效（Windows）：`[Environment]::SetEnvironmentVariable("HF_ENDPOINT","https://hf-mirror.com","User")`，然后重开窗口。

#### 5.4.3 校验 + 打包

**Ubuntu 本地**

```bash
cd ~/smolvla                                   # 进入父目录

ls -lh smolvla_base                            # 必须看到 config.json 与 model.safetensors
test -f libero/meta/info.json && echo OK       # 输出 OK 说明数据集完整
ls libero-assets                               # 应有 5 个资产目录
du -sh *                                       # 看 4 个目录各多大，总计 5–8GB

rm -rf ./*/.cache 2>/dev/null || true          # 删每个仓库里的 HF 元数据（几十 MB，传了没用）

tar -cf smolvla_base.tar  smolvla_base
tar -cf smolvlm2.tar      SmolVLM2-500M-Video-Instruct
tar -cf libero_data.tar   libero
tar -cf libero_assets.tar libero-assets
# 打成 4 个 tar：数据集 ~3400 个 mp4、资产上万个贴图，逐个传会慢到崩
# 不要加 -z：mp4 已压缩，gzip 只烧 CPU

sha256sum *.tar | tee sha256.txt
# 生成哈希清单并存盘，后面和服务器逐行比对，专治“传到一半”
```

**Windows 本地**

```powershell
Set-Location D:\smolvla                        # 进入父目录

Get-ChildItem .\smolvla_base | Select-Object Name, Length   # 必须看到 config.json 与 model.safetensors
Test-Path .\libero\meta\info.json              # True 说明数据集完整
Get-ChildItem .\libero-assets -Directory       # 应有 5 个资产目录

$dirs = "smolvla_base","SmolVLM2-500M-Video-Instruct","libero","libero-assets"
foreach ($d in $dirs) {                        # 逐个算体积，总计应在 5–8GB
  $gb = (Get-ChildItem $d -Recurse -File | Measure-Object Length -Sum).Sum / 1GB
  "{0,-32} {1,6:N2} GB" -f $d, $gb
}

Get-ChildItem -Directory | ForEach-Object {     # 删每个仓库里的 HF 元数据
  $c = Join-Path $_.FullName ".cache"
  if (Test-Path $c) { Remove-Item -Recurse -Force $c }
}

tar -cf smolvla_base.tar  smolvla_base
tar -cf smolvlm2.tar      SmolVLM2-500M-Video-Instruct
tar -cf libero_data.tar   libero
tar -cf libero_assets.tar libero-assets
# Windows 10/11 自带 tar（bsdtar），不用装 7-Zip；同样不要加 -z

Get-FileHash *.tar -Algorithm SHA256 | Tee-Object -FilePath sha256.txt
# 生成哈希清单并写文件，格式与 sha256sum 不同，但哈希值本身可比
```

#### 5.4.4 上传

**Ubuntu 本地**

```bash
scp -P 38123 *.tar root@connect.westb.seetacloud.com:/root/autodl-tmp/
# -P 是端口（ssh 才是小写 -p）；一次传 4 个，断了要重传

rsync -avP -e "ssh -p 38123" ./*.tar root@connect.westb.seetacloud.com:/root/autodl-tmp/
# 推荐：-a 保留属性、-v 显示进度、-P 断点续传；中断后重跑同一条即可接着传
```

**Windows 本地**

```powershell
scp -P 38123 .\*.tar root@connect.westb.seetacloud.com:/root/autodl-tmp/
# 端口是大写 -P；首次连接问 yes/no，输 yes；密码粘贴时不显示字符
```

> Windows 没有内置 `rsync`。要断点续传，二选一：
> ① 在 WSL2 里 `rsync -avP -e "ssh -p 38123" /mnt/d/smolvla/*.tar root@host:/root/autodl-tmp/`；
> ② 用 WinSCP 图形界面拖（支持 `.filepart` 续传）。
> 多实例复用：控制台→文件存储上传，服务器上 `cp /root/autodl-fs/*.tar /root/autodl-tmp/`（同机秒级）。

#### 5.4.5 服务器解包 + 校验（Ubuntu）

```bash
mkdir -p /root/autodl-tmp/SmolVLA/{smolvla_base,SmolVLM2-500M-Video-Instruct,libero,libero-assets}
# 确保四个落地目录都存在

cd /root/autodl-tmp
# 所有 tar 都在这里

sha256sum *.tar
# 输出 4 行哈希，与本地 sha256.txt / Get-FileHash 的结果逐行对；不一致就是传坏了

tar -xf smolvla_base.tar  -C SmolVLA/
# -x 解包 -f 指定文件 -C 指定目标目录

tar -xf smolvlm2.tar      -C SmolVLA/
tar -xf libero_data.tar   -C SmolVLA/

tar -xf libero_assets.tar -C SmolVLA/
# --strip-components=1 去掉顶层 libero-assets/，让 5 个资产子目录直接落在 SmolVLA/libero-assets/ 下
# 这一步不做，LIBERO 会认为资产不存在，重新联网整包下载

ls SmolVLA/smolvla_base/config.json SmolVLA/smolvla_base/model.safetensors
# 两个文件都在才算模型完整

ls SmolVLA/libero/meta/info.json
# 数据集完整性的最小判据

for d in articulated_objects stable_scanned_objects turbosquid_objects stable_hope_objects scenes; do
  [ -d "SmolVLA/libero-assets/$d" ] && echo "OK   $d" || echo "MISS $d"
done
# 逐个检查 5 个资产目录；有 MISS 就说明解包层级错了

du -sh SmolVLA/{smolvla_base,SmolVLM2-500M-Video-Instruct,libero,libero-assets}
# 核对体积，和本地一致

rm -f ./*.tar
# 确认以上全 OK 后再删，能省 10GB
```

#### 5.4.6 全离线训练命令（Ubuntu）

```bash
export HF_HUB_OFFLINE=1          # 禁止任何 Hub 请求
export TRANSFORMERS_OFFLINE=1    # 禁止 transformers 联网拉 tokenizer / 权重

lerobot-train \
  --policy.path=/root/autodl-tmp/SmolVLA/smolvla_base \
  --policy.vlm_model_name=/root/autodl-tmp/SmolVLA/SmolVLM2-500M-Video-Instruct \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.use_amp=false \
  --dataset.repo_id=lerobot/libero \
  --dataset.root=/root/autodl-tmp/SmolVLA/libero \
  --dataset.video_backend=pyav \
  --output_dir=/root/autodl-tmp/outputs/libero_smolvla \
  --batch_size=32 --steps=80000 --save_freq=10000 --log_freq=50 \
  --num_workers=8 --env_eval_freq=0 --seed=1000
```

> 多行命令里的 `\` 后面**不能写注释**（会把续行截断），所以参数说明列在下表：

| 参数                              | 作用                                | 不写会怎样                    |
| ------------------------------- | --------------------------------- | ------------------------ |
| `--policy.path`                 | 指向本地模型**目录**                      | 指向文件会报找不到 config.json    |
| `--policy.vlm_model_name`       | 基座也指本地                            | 仍去 Hub 拉，离线直接报错          |
| `--policy.input_features=null`  | 从数据集推断输入特征                        | 相机名不匹配，`KeyError`        |
| `--policy.output_features=null` | 推断动作维度（7）                         | 防御性写法，建议保留               |
| `--policy.push_to_hub=false`    | 关掉自动上传                            | 默认 true，会要求登录 HF         |
| `--policy.use_amp=false`         | 关闭 AMP（本机必须）                       | SmolVLA 权重为 bf16，use_amp 会走 fp16 GradScaler 并崩溃（`not implemented for BFloat16`）；关闭后仍以 bf16 计算 |
| `--dataset.root`                | 数据真实路径                            | 只给 repo\_id 会去联网下载       |
| `--dataset.video_backend`       | MP4 解码后端                          | 装不上 torchcodec 时改 `pyav` |
| `--save_freq=10000`             | 每 10k 步存盘，全程 8 个 ckpt（≈40G，磁盘装得下） | 断了要重跑（默认 20k 才存）         |
| `--env_eval_freq=0`             | 关训练中仿真评测                          | 抢显存、拖慢训练                 |
| `--num_workers=8`               | 数据加载进程数                           | `data_s` 变高，GPU 空转       |

两个必须记住的点：

1. `--policy.vlm_model_name` **必须指本地目录**。不指，即使权重都在本地，实例化时还是会去 Hub 找基座，报 `OSError: We couldn't connect to 'https://huggingface.co'`。
2. 这个值会被写进你新 checkpoint 的 `config.json`，**评测和续训时也要带同样的覆盖**（`lerobot-eval` 和 `lerobot-train --resume` 都支持 `--policy.vlm_model_name=...`）。尤其换机器后，checkpoint 里固化的原路径在新机器上不存在，必须用覆盖参数指到新机器的本地基座目录，否则报 `OSError`。

### 5.5 校验数据形状（写进实验记录）

```bash
python - <<'PY'
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("lerobot/libero", root="/root/autodl-tmp/SmolVLA/libero")
print("episodes:", ds.num_episodes, "frames:", ds.num_frames)
print("tasks:", len(ds.meta.tasks))
for k, v in ds.features.items():
    print(f"  {k}: {v.get('dtype')} {v.get('shape')}")
s = ds[0]
print("sample keys:", list(s.keys()))
PY
```

期望：

```
episodes: 1693  frames: 273465  tasks: 40
observation.images.image    video  [3,256,256]
observation.images.image2   video  [3,256,256]
observation.state           float32 [8]
action                      float32 [7]
```

### 5.6 用可视化确认“数据没坏”

```bash
# 需先装可视化 extra：pip install --no-build-isolation -e ".[dataset_viz]"
lerobot-dataset-viz --repo-id lerobot/libero --episode-index 0
# 本地无浏览器时，用 SSH 端口转发到本机看，或直接跳过
```

看一眼：主视角 + 腕部两路画面是否正常、语言指令是否合理、动作是否连续。**数据看起来不对，训练一定不对。**

### 5.7 训练命令标准模板（通用，后续所有训练命令复用）

> 下面 4 段代码块（快照 → 执行 → 监控 → 关机）构成**一次完整训练 run** 的标准流程。§6.1 冒烟、§7.1 20k、§7.2 80k 训练都遵循这套模板，只改"命令本体"里的具体参数。

**① 参数快照**（训练前执行一次，把完整上下文固化到 `output_dir/run_info/`）

```bash
# ===== 训练前快照（每次 run 必做）=====
META=/root/autodl-tmp/outputs/_runmeta/<你的run名>   # 元数据旁路目录
OUTPUT_DIR=/root/autodl-tmp/outputs/<你的run名>       # 训练产物目录（lerobot 创建）
mkdir -p "$META/run_info"

{
  echo "===== 时间与机器 ====="
  date "+%Y-%m-%d %H:%M:%S  %Z"
  hostname

  echo ""
  echo "===== lerobot git commit ====="
  cd /root/autodl-tmp/src/lerobot && git rev-parse HEAD && git log -1 --oneline

  echo ""
  echo "===== 关键依赖版本 ====="
  pip list | grep -Ei "lerobot|torch|transformers|mujoco|robosuite|hf-libero|torchcodec|accelerate"

  echo ""
  echo "===== GPU 信息 ====="
  nvidia-smi

  echo ""
  echo "===== 关键环境变量 ====="
  env | grep -E "^(HF_|MUJOCO|PYOPENGL|CUDA|TOKENIZERS|OMP|MKL)" | sort

  echo ""
  echo "===== 模型/数据体积 ====="
  du -sh /root/autodl-tmp/SmolVLA/{smolvla_base,SmolVLM2-500M-Video-Instruct,libero,libero-assets}

  echo ""
  echo "===== 磁盘空间 ====="
  df -h / /root/autodl-tmp
} | tee "$META/run_info/snapshot.txt"
# snapshot.txt 就是以后你想复盘"那次训练到底用了什么环境"时要翻的第一份文件
```

> ⚠️ **实测坑（v0.6.1）**：`lerobot-train` 要求 `--output_dir` **不存在**，提前 `mkdir` 会 `FileExistsError`。所以快照、`train_cmd.sh`、`train.log` 一律放旁路目录 `META=/root/autodl-tmp/outputs/_runmeta/<run名>`，`OUTPUT_DIR` 交给 lerobot 自己创建。

**② 训练执行（tmux 前台跑：终端实时打印 + tee 完整留档）**

```bash
# ===== 训练主命令（完整参数写进脚本并留档）=====
cd /root/autodl-tmp/src/lerobot

cat > "$META/run_info/train_cmd.sh" <<'CMDEOF'
# 把下面的 lerobot-train 完整命令原封不动粘进这里；保存后这个文件就是可复现脚本
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.use_amp=false \
  --dataset.repo_id=lerobot/libero \
  --dataset.root=/root/autodl-tmp/SmolVLA/libero \
  --dataset.video_backend=pyav \
  --output_dir=/root/autodl-tmp/outputs/<你的run名> \
  --job_name=<你的run名> \
  --batch_size=32 \
  --steps=20000 \
  --save_freq=5000 \
  --log_freq=50 \
  --num_workers=8 \
  --env_eval_freq=0 \
  --seed=1000
CMDEOF
# ⚠️ 上面命令的 --output_dir / --job_name / --steps / --save_freq 要和你的 run 对应

chmod +x "$META/run_info/train_cmd.sh"

# ===== tmux 前台执行：核心日志直接打在控制终端，同时 tee 落盘留档 =====
tmux new -s <你的run名>     # 进入 tmux 会话（断线后 tmux attach -t <你的run名> 回来）
# --- 以下在 tmux 会话内执行；step/loss/mem_gb/grdn 等核心行实时打印在当前终端 ---
META=/root/autodl-tmp/outputs/_runmeta/<你的run名>
bash "$META/run_info/train_cmd.sh" 2>&1 | tee "$META/train.log"
echo "EXIT=${PIPESTATUS[0]}" | tee "$META/train.exit"   # 训练真实退出码（取管道第一段的）
# 运行中可随时 Ctrl+B, D detach，训练不受影响；重连 attach 后继续看实时日志
# 不用 nohup：tmux 已提供断线保护，且前台运行才能让日志直接打在控制终端
```

**③ 定期断点策略**（已由 `--save_freq` 自动完成，这是说明）

| 训练阶段   | 总步数   | `--save_freq` | 预估存盘间隔    | 产物                      |
| ------ | ----- | ------------- | --------- | ----------------------- |
| 冒烟     | 300   | **300**       | 训练结束时 1 次 | 1 个 checkpoint          |
| 20k 验证 | 20000 | **5000**      | \~75 min  | 4 个 checkpoint（每 5k 步）  |
| 80k 正式 | 80000 | **10000**     | \~150 min | 8 个 checkpoint（每 10k 步） |

> 🎞️ **解码后端（本机实测结论）**：torchcodec 与 torch 2.8.0+cu128 不兼容（各版本 dlopen 阶段 std::bad_alloc 崩溃），已卸载；统一使用 **pyav**（av 15.1.0，lerobot 自动设为默认解码器）。§5.5 已实测 pyav 正常解码 LIBERO MP4。

> 💾 **磁盘预算（保存距离为什么拉长）**：单个 checkpoint ≈ **1.3G**（实测：权重 865M + 优化器状态 394M；SmolVLA 默认 train\_expert\_only=True，只有 action expert 参与优化）。数据盘可用 \~90G：80k 全程 8 个 ≈ 10.4G；20k→80k 同目录续训累计 4+6=10 个 ≈ 13G，加冒烟和评测视频峰值 \~20G，余量充裕。**不要**把 save\_freq 调回 2000/5000——累计 22 个 ckpt ≈ 110G，必爆。代价：80k 阶段中断最多损失 \~150 分钟机时（约 5 元），可接受。

> 自动保存的 checkpoint 在 `$OUTPUT_DIR/checkpoints/XXXXXX/`（**6 位数字目录名，无 step\_ 前缀**，如 `010000`），**每个都包含权重（`pretrained_model/`）+ 优化器/调度器/RNG/步数（`training_state/`）**，随时可以 `--config_path=$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json --resume=true` 接着跑。注意：`train_config.json` 只存在于 checkpoint 内部，run 根目录没有。`last` 是指向最新 checkpoint 的符号链接，续训、评测都走它即可。

**④ 监控与条件关机**（分开写，避免训练失败时误关机）

```bash
# ===== 实时监控 =====
# 方式 A（推荐）：回到训练所在 tmux 会话，直接看终端实时打印
tmux attach -t <你的run名>
# 方式 B：另开一个终端，看落盘日志与硬件状态
META=/root/autodl-tmp/outputs/_runmeta/<你的run名>
tail -f "$META/train.log"                               # 实时看 loss / mem_gb
watch -n 10 "nvidia-smi | head -20"                     # 每 10 秒看显存与利用率
watch -n 30 "ls -lh $OUTPUT_DIR/checkpoints/"           # 每 30 秒看新 checkpoint 有没有出来

# ===== 训练完成后：条件关机脚本（只在训练 + 评测都成功才执行）=====
# 等训练写出退出码文件（tmux 前台模式没有可 wait 的后台 PID，改为轮询 train.exit）
META=/root/autodl-tmp/outputs/_runmeta/<你的run名>
while [ ! -f "$META/train.exit" ]; do sleep 60; done
TRAIN_EXIT=$(cut -d= -f2 "$META/train.exit")

if [ $TRAIN_EXIT -eq 0 ]; then
  echo "✅ 训练成功结束（exit_code=0），准备自动关机..."
  # 关机前先跑一次评测（可选，不想等评测就把下面这段注释掉）
  # lerobot-eval ... --output_dir="$OUTPUT_DIR/eval_final" || { echo "⚠️ 评测失败但训练已完成"; }
  # 60 秒宽限，防止脚本卡住；shutdown 执行后还有 60 秒倒计时
  shutdown -h +1 "SmolVLA 训练完成，60 秒后关机（run: $OUTPUT_DIR）"
else
  echo "❌ 训练异常退出（exit_code=$TRAIN_EXIT）！保留现场，请手动排查"
  echo "   最后 50 行日志："
  tail -50 "$META/train.log"
  # 不关机！错误排查完再手动 shutdown
fi
```

> ⚠️ **千万别用** **`; shutdown`** 直接跟在训练命令后面——分号不看退出码，训练崩溃时也会照样关机，bug 现场全丢。一定要用上面的 `if [ $? -eq 0 ]` 条件包装。
>
> ⚠️ AutoDL 的 `shutdown -h` 会**释放实例**（下次再开可能被别人占走）。如果只是想暂停但保留 GPU，用控制台点"无卡模式开机"，或者确认后再 `shutdown`。

***

## 6. 冒烟测试（M2：2 小时，别省）

### 6.1 300 步训练

> 走**离线路线**的，把下面命令里的 `--policy.path` 换成 `/root/autodl-tmp/SmolVLA/smolvla_base`，并补上 `--policy.vlm_model_name=/root/autodl-tmp/SmolVLA/SmolVLM2-500M-Video-Instruct`，同时在 shell 里 `export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`（完整版见 §5.4.4）。
>
> 严格遵循 §5.7 标准模板：先跑**①参数快照**（`OUTPUT_DIR=/root/autodl-tmp/outputs/smoke_300`），再把下面命令原封不动粘进**②训练执行**的 `train_cmd.sh`，最后在 tmux 会话里前台启动（日志实时可见）。冒烟训练**不自动关机**——结束还要跑 §6.3 评测。

**快照阶段**：

```bash
OUTPUT_DIR=/root/autodl-tmp/outputs/smoke_300   # ← 这行和下面 train_cmd.sh 里的 --output_dir 必须一致
# 跑 §5.7 ① 的快照代码块
```

**训练命令本体**（粘进 `train_cmd.sh`）：

```bash
lerobot-train \
  --policy.path=/root/autodl-tmp/SmolVLA/smolvla_base \
  --policy.vlm_model_name=/root/autodl-tmp/SmolVLA/SmolVLM2-500M-Video-Instruct \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.use_amp=false \
  --dataset.repo_id=lerobot/libero \
  --dataset.root=/root/autodl-tmp/SmolVLA/libero \
  --dataset.video_backend=pyav \
  --output_dir=/root/autodl-tmp/outputs/smoke_300 \
  --job_name=smoke_300 \
  --batch_size=16 \
  --steps=300 \
  --save_freq=300 \
  --log_freq=20 \
  --num_workers=4 \
  --env_eval_freq=0 \
  --seed=1000
```

> ⚠️ **离线必做（实测）**：`smolvla_base/policy_preprocessor.json` 里的 `tokenizer_name` 是 repo id `HuggingFaceTB/SmolVLM2-500M-Video-Instruct`，`HF_HUB_OFFLINE=1` 时无法解析。把该字段改成本地路径即可：
>
> ```bash
> python - <<'PY'
> import json
> p='/root/autodl-tmp/SmolVLA/smolvla_base/policy_preprocessor.json'
> d=json.load(open(p))
> for s in d['steps']:
>     c=s.get('config',{})
>     if isinstance(c,dict) and 'tokenizer_name' in c:
>         c['tokenizer_name']='/root/autodl-tmp/SmolVLA/SmolVLM2-500M-Video-Instruct'
> json.dump(d, open(p,'w'), indent=2, ensure_ascii=False)
> PY
> ```
>
> （`--policy.vlm_model_name` 只覆盖模型配置，覆盖不到 processor 里固化的 tokenizer 名。）

> 冒烟训练的 `--save_freq=300` 意味着**只有 1 个 checkpoint**（训练结束时存一次）。如果冒烟成功需要续训但又意外断开，没关系——重跑一次命令就行，或者先手动把 save\_freq 改成 100。

**逐参数解释（后面正式训练会复用）**

| 参数                                   | 作用                    | 新手注意                                                                                   |
| ------------------------------------ | --------------------- | -------------------------------------------------------------------------------------- |
| `--policy.path=lerobot/smolvla_base` | 加载 450M 预训练权重         | 首次会下载 \~1.5GB 到 `HF_HOME`                                                              |
| `--policy.input_features=null`       | **强制从数据集推断输入特征**      | 不加会因相机名不匹配报 KeyError                                                                   |
| `--policy.output_features=null`      | 从数据集推断动作维度（7）         | 防御性写法                                                                                  |
| `--policy.push_to_hub=false`         | 默认是 **true**，会上传并要求登录 | 不加会报 “no token”                                                                        |
| `--policy.use_amp=false`              | 关闭 AMP                  | **实测开启会崩**：SmolVLA 权重是 bf16，而 use_amp 走 fp16 GradScaler，报 `_amp_foreach_non_finite_check_and_unscale_cuda not implemented for BFloat16`；关闭后模型仍以 bf16 计算 |
| `--dataset.video_backend=pyav`       | MP4 解码后端              | **本机实测 torchcodec 与 torch 2.8.0+cu128 不兼容（dlopen 阶段 std::bad\_alloc 崩溃），已卸载并统一用 pyav** |
| `--output_dir`                       | 必须在**数据盘**            | 目录已存在会 `FileExistsError`                                                               |
| `--env_eval_freq=0`                  | 关闭训练中仿真评测             | 默认 20000，会抢显存                                                                          |

### 6.2 看日志，别看 loss 一个数

每 `--log_freq` 步会打印一行，重点看四个字段：

```
step:300 smp/s:... data_s:0.0xx updt_s:0.3xx mem_gb:12.4 lr:9.9e-05 loss:0.8xx
```

| 字段       | 正常              | 异常信号                                    |
| -------- | --------------- | --------------------------------------- |
| `mem_gb` | 12–20（batch 16） | >22.5 → 降 batch                         |
| `data_s` | 明显小于 `updt_s`   | 接近 `updt_s` → 数据加载是瓶颈，加 `--num_workers` |
| `updt_s` | 0.2–0.6 s/step  | 用它算总时长：`steps × updt_s / 3600` 小时       |
| `loss`   | 从 \~1.0 缓慢下降    | 300 步内剧烈震荡正常，别慌                         |

### 6.3 单任务评测（验证仿真真的能跑）

```bash
cd /root/autodl-tmp/src/lerobot

lerobot-eval \
  --policy.path=/root/autodl-tmp/outputs/smoke_300/checkpoints/last/pretrained_model \
  --policy.vlm_model_name=/root/autodl-tmp/SmolVLA/SmolVLM2-500M-Video-Instruct \
  --env.type=libero \
  --env.task=libero_spatial \
  --env.task_ids='[0]' \
  --eval.batch_size=1 \
  --eval.n_episodes=3 \
  --env.max_parallel_tasks=1 \
  --policy.n_action_steps=10
```

> ⚠️ **首次评测前置（实测）**：① `printf 'n\n' | python -c "from libero.libero import benchmark"` 预生成 `~/.libero/config.yaml`（否则 `EOFError`）；② `libero` 的 `get_assets_path()` 只认包内 `assets/`，需软链到本地资产：`ln -s /root/autodl-tmp/SmolVLA/libero-assets /root/miniconda3/lib/python3.12/site-packages/libero/libero/assets`；③ 跑 `lerobot-eval` 也要带 `--policy.vlm_model_name=<本地VLM目录>`。

**这一步真正的目的是**：触发 LIBERO 资产下载（首次会拉 `lerobot/libero-assets` 到 `~/.cache/libero`，已软链到数据盘），并验证 EGL 渲染通路。

期望：3 个 episode 都跑完，打印 `pc_success`（300 步的模型大概率 0%，**这是正常的**，只要不报错）。

### 6.4 M2 通过标准

- [ ] 训练 300 步无异常退出，`outputs/smoke_300/checkpoints/last/pretrained_model` 存在
- [ ] 评测 3 episode 全部完成，输出成功率与耗时
- [ ] `~/.cache/libero` 已有文件，`du -sh` 不为 0
- [ ] 显存峰值记录下来了（`nvidia-smi` 或日志 `mem_gb`）

> ✅ **本机实测（2026-10-22）**：训练 300 步 1m53s（3.05 step/s），`mem_gb 4.52`，峰值显存 ~5.1G，loss 1.72→0.755；checkpoint 1.3G（权重 865M + 状态 394M）；评测 3 episode 75.6s（25.2s/ep），`pc_success 0.0%`（预期），生成 3 段 MP4。训练日志末尾会有 `DataLoader worker ... Aborted` 的 teardown 噪声，但发生在 `End of training` 之后且**退出码为 0**，不影响结果。

***

## 7. 正式训练（M3 → M4）

### 7.1 第一段：20k 步（约 3–4 h，验证收敛）

> 严格遵循 §5.7 标准模板：先跑**①参数快照**（`OUTPUT_DIR=/root/autodl-tmp/outputs/libero_smolvla`），再把下面命令粘进**②训练执行**的 `train_cmd.sh`。
>
> **20k 训练结束后不自动关机**——要先跑评测确认 spatial ≥ 55%，再决定要不要续到 80k。用 §5.7 ④ 里的条件关机脚本（记得改 `OUTPUT_DIR`），但把 shutdown 那两行**暂时注释掉**，等评测过了再手动取消注释执行。

**快照阶段**：

```bash
OUTPUT_DIR=/root/autodl-tmp/outputs/libero_smolvla
# 跑 §5.7 ① 的快照代码块
```

**训练命令本体**（粘进 `train_cmd.sh`）：

```bash
lerobot-train \
  --policy.path=/root/autodl-tmp/SmolVLA/smolvla_base \
  --policy.vlm_model_name=/root/autodl-tmp/SmolVLA/SmolVLM2-500M-Video-Instruct \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.use_amp=false \
  --policy.scheduler_decay_steps=80000 \
  --dataset.repo_id=lerobot/libero \
  --dataset.root=/root/autodl-tmp/SmolVLA/libero \
  --dataset.video_backend=pyav \
  --output_dir=/root/autodl-tmp/outputs/libero_smolvla \
  --job_name=libero_smolvla \
  --batch_size=32 \
  --steps=20000 \
  --save_freq=5000 \
  --log_freq=50 \
  --num_workers=8 \
  --env_eval_freq=0 \
  --seed=1000
```

> **定期断点**：`--save_freq=5000` → 每 \~75 分钟存一次 checkpoint，总共 4 个（`005000`/`010000`/`015000`/`020000`，6 位数字目录名）。中途挂了损失最多 \~75 分钟，用 `checkpoints/last/pretrained_model/train_config.json --resume=true` 接着跑。

跑完立刻评（先只看 spatial，10 episode）：

```bash
lerobot-eval \
  --policy.path=/root/autodl-tmp/outputs/libero_smolvla/checkpoints/last/pretrained_model \
  --env.type=libero --env.task=libero_spatial \
  --eval.batch_size=1 --eval.n_episodes=10 --env.max_parallel_tasks=1 \
  --policy.n_action_steps=10
```

判定：

| spatial 成功率 | 判断    | 动作                       |
| ----------- | ----- | ------------------------ |
| < 30%       | 链路有问题 | 回到 §10 查（多半是特征/归一化/评测配置） |
| 30–55%      | 欠拟合   | 继续训到 80k                 |
| **55–75%**  | 正常    | ✅ 进入 7.2                 |
| > 75%       | 很好    | 直接长跑                     |

### 7.2 第二段：跑到 80k 步

> 这是最后的长跑训练（**预计 10–13 小时**），**必须配自动关机**（训练完没人盯着容易浪费钱）。严格遵循 §5.7 标准模板。
>
> **关键**：续训前先把上一轮 20k 的 `snapshot.txt` 和 `train_cmd.sh` 复制过来——续训命令本身简短，但参数快照要**完整记录训练现场**，方便以后复盘"这次 80k 和上次 20k 到底差在哪"。

**快照阶段**（沿用 20k 的快照 + 续训特定信息）：

```bash
OUTPUT_DIR=/root/autodl-tmp/outputs/libero_smolvla   # 续训用同一个 output_dir，新建 run_info/continue/
mkdir -p "$OUTPUT_DIR/run_info/continue"

# 复制 20k 的快照过来（作为续训的基线）
cp "$OUTPUT_DIR/run_info/snapshot.txt" "$OUTPUT_DIR/run_info/continue/snapshot_20k.txt"

{
  echo "===== 续训时间 ====="
  date "+%Y-%m-%d %H:%M:%S  %Z"

  echo ""
  echo "===== 20k 训练的 checkpoint 列表（续训起点）====="
  ls -lh "$OUTPUT_DIR/checkpoints/"

  echo ""
  echo "===== 20k 训练的 train_config.json 摘要 ====="
  TRAIN_CFG="$OUTPUT_DIR/checkpoints/last/pretrained_model/train_config.json"   # v0.6.1 只存在 checkpoint 内部
  python -c "import json; c=json.load(open('$TRAIN_CFG')); print(json.dumps({k:c[k] for k in ['policy','dataset','batch_size','steps','save_freq','num_workers','env_eval_freq','seed']}, indent=2, default=str))" 2>/dev/null || echo "(train_config.json 格式不同，直接 cat 一下)"
  cat "$TRAIN_CFG" | head -40

  echo ""
  echo "===== 当前磁盘空间（长跑前再确认一次）====="
  df -h / /root/autodl-tmp
  du -sh "$OUTPUT_DIR"

} | tee "$OUTPUT_DIR/run_info/continue/snapshot.txt"
```

**方式 A（推荐，简单）：直接续训**

```bash
lerobot-train \
  --config_path=/root/autodl-tmp/outputs/libero_smolvla/checkpoints/last/pretrained_model/train_config.json \
  --resume=true \
  --output_dir=/root/autodl-tmp/outputs/libero_smolvla \
  --dataset.root=/root/autodl-tmp/SmolVLA/libero \
  --policy.vlm_model_name=/root/autodl-tmp/SmolVLA/SmolVLM2-500M-Video-Instruct \
  --steps=80000 \
  --save_freq=10000
```

- 上面这行**粘进** **`train_cmd.sh`** 时，tee 要改成追加模式：`bash train_cmd.sh 2>&1 | tee -a "$META/train.log"`（用 `tee -a` 不用 `tee`，保留 20k 阶段的日志）
- `--config_path` **必须指到 checkpoint 内部**的 `train_config.json`：v0.6.1 只把它存在 `checkpoints/<步数>/pretrained_model/` 里，run 根目录没有这个文件，指过去会直接 `FileNotFoundError`。
- `--resume=true` 会从 `checkpoints/last` 恢复权重、优化器、调度器和归一化统计。
- 中间 3 个覆盖参数是**跨机续训保险**（同一台 AutoDL 实例续训可省略；本地离线训练 → 上传 → 云端续训**必须带**）：`train_config.json` 里固化的是原始机器路径——不带 `--dataset.root` 会按旧路径找数据集；不带 `--output_dir` 会把新 checkpoint 写进旧路径目录；不带 `--policy.vlm_model_name` 会按 checkpoint `config.json` 里的旧路径实例化 VLM 基座，报 `OSError`。换机后三个路径都要指到新机器的实际位置。
- 注意：调度器 `decay_steps` 默认 30000，续训后 LR 会在 30k 处已贴地，属正常；若想要全程衰减，在续训命令里加 `--policy.scheduler_decay_steps=80000` 覆盖。

**方式 B（更干净）：从 20k checkpoint 重新起一次 60k**

```bash
lerobot-train \
  --policy.path=/root/autodl-tmp/outputs/libero_smolvla/checkpoints/last/pretrained_model \
  --policy.input_features=null --policy.output_features=null \
  --policy.push_to_hub=false --policy.device=cuda --policy.use_amp=false \
  --policy.scheduler_decay_steps=60000 \
  --dataset.repo_id=lerobot/libero \
  --dataset.root=/root/autodl-tmp/SmolVLA/libero \
  --output_dir=/root/autodl-tmp/outputs/libero_smolvla_80k \
  --batch_size=32 --steps=60000 --save_freq=10000 --log_freq=50 \
  --num_workers=8 --env_eval_freq=0 --seed=1000
```

> 二选一即可。方式 B 的 LR 曲线更漂亮，但要重新预热 1000 步。方式 B 是**全新 run**，快照里的 `OUTPUT_DIR` 要改成 `.../libero_smolvla_80k`。

**定期断点策略（80k 阶段）**：

- `--save_freq=10000` → 每 \~150 分钟存一次 checkpoint，总共 8 个（磁盘预算见 §5.7③）
- `checkpoints/010000/` → 第一次保存（训练跑完最后是 `080000/`）
- 中途挂了用最新 ckpt 的 `train_config.json --resume=true` 续训，最多损失 \~150 分钟

**条件关机脚本（长跑必须配）**：

> 💡 **所有训练都在 tmux 会话里前台跑**（§5.7 ② 的方式）：日志直接打在 pane 上，SSH 断线后 `tmux attach -t <run名>` 回来继续看。条件关机脚本放**另一个 pane**（`Ctrl+B %` 分屏），它轮询 `train.exit` 拿退出码。`Ctrl+B, D` detach 后训练与关机监控都不受影响。

```bash
# ===== 第一步：tmux 里先开训练 =====
tmux new -s smolvla                       # 进入 tmux 会话
# （在 tmux 的 pane 1 里跑训练——§5.7 ② 的 tee 那几行，日志直接打在 pane 上）

# ===== 第二步：分一个新 pane 跑条件关机监控 =====
# Ctrl+B % 分左右 pane，或 Ctrl+B " 分上下 pane
# （在 tmux 的 pane 2 里跑下面的条件关机脚本）
OUTPUT_DIR=/root/autodl-tmp/outputs/libero_smolvla   # 方式 B 改成 .../libero_smolvla_80k
META=/root/autodl-tmp/outputs/_runmeta/libero_smolvla  # 方式 B 改成 _runmeta/libero_smolvla_80k
while [ ! -f "$META/train.exit" ]; do sleep 60; done
TRAIN_EXIT=$(cut -d= -f2 "$META/train.exit")

if [ $TRAIN_EXIT -eq 0 ]; then
  echo "✅ 80k 训练完成（exit_code=0），先跑完整评测再关机..."
  # 完整评测放在关机前，跑完直接 shutdown
  lerobot-eval \
    --policy.path="$OUTPUT_DIR/checkpoints/last/pretrained_model" \
    --env.type=libero \
    --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
    --eval.batch_size=1 --eval.n_episodes=10 --env.max_parallel_tasks=1 \
    --policy.n_action_steps=10 \
    --output_dir="$OUTPUT_DIR/eval_final" \
    && echo "✅ 评测成功！" \
    || echo "⚠️ 评测有错误但训练已完成，继续关机"

  shutdown -h +1 "SmolVLA 80k 训练+评测完成，60 秒后关机（run: $OUTPUT_DIR）"
else
  echo "❌ 80k 训练异常退出（exit_code=$TRAIN_EXIT）！保留现场"
  tail -50 "$META/train.log"
  # 不关机！手动排查
fi
```

> 这个脚本**必须单独跑在另一个 pane/终端**——训练在 pane 1 前台跑，它在 pane 2 轮询 `train.exit`，训练一结束它就能拿到退出码并决定关机与否。

### 7.3 显存与吞吐调参表（4090D 24GB 实测区间）

| 配置                                          | batch | 峰值显存       | 相对速度 | 说明         |
| ------------------------------------------- | ----- | ---------- | ---- | ---------- |
| 默认（expert only）                             | 32    | \~12–16 GB | 1.0× | ✅ 推荐起点     |
| 默认                                          | 64    | \~18–22 GB | 1.6× | 数据加载跟得上就用它 |
| 默认                                          | 8     | \~5–7 GB   | 0.4× | 只用于调试      |
| `train_expert_only=false`                   | 8–16  | \~14–20 GB | 0.5× | 想放开 VLM 时  |
| `freeze_vision_encoder=false` 且 expert only | 16–24 | \~18–22 GB | 0.8× | 视觉适配专用任务时试 |

调法：先 `--batch_size=32`，看日志 `mem_gb`；若 <18 且 `data_s` 远小于 `updt_s`，升到 64；若 OOM，降一半。

### 7.4 监控与日志

```bash
watch -n 5 nvidia-smi
# 每 5 秒刷新一次显卡状态；重点看显存占用与 GPU 利用率

tail -f /root/autodl-tmp/outputs/libero_smolvla/train.log
# 实时跟踪日志（-f）；Ctrl+C 退出，不影响训练

ls -lh /root/autodl-tmp/outputs/libero_smolvla/checkpoints/
# 看已存了哪些 checkpoint、各多大（每个约 5G：权重 ~1.8G + 优化器状态 ~3.2G）

du -sh /root/autodl-tmp/outputs/libero_smolvla
# 看整个输出目录占多少，数据盘快满时先删旧 ckpt
```

想用 wandb（可选）：

```bash
pip install wandb        # 装实验记录工具
wandb login              # 粘贴一次 API key，之后免登录

# 训练命令里加这两个参数：
#   --wandb.enable=true --wandb.project=smolvla-libero
```

### 7.5 训练期常见现象判读

| 现象                    | 含义                          | 处理                                         |
| --------------------- | --------------------------- | ------------------------------------------ |
| loss 在 0.6–1.2 之间震荡不降 | 正常（flow matching 的 loss 尺度） | 看评测，不看 loss                                |
| loss 突然 NaN           | AMP 溢出 / 学习率过大              | 关 `use_amp`，或 `--policy.optimizer_lr=5e-5` |
| `data_s` 持续上升         | DataLoader 跟不上              | `--num_workers=12`，确认数据在本地盘                |
| GPU 利用率长期 <50%        | CPU 瓶颈                      | 同上；检查是否在读网络盘                               |
| 日志里 `lr` 很早变 2.5e-6   | 调度器 30k 衰减                  | 正常；或设 `scheduler_decay_steps`              |
| 训练中途机器重启              | 实例被释放/掉线                    | 用 `--resume` 续                             |

### 7.6 M3/M4 通过标准

- [x] 20k checkpoint 在 spatial 上 ≥55%（**实测 74.0%**，100 ep，task 5 最弱 40%）
- [ ] 80k（或 60k）checkpoint 四套平均 ≥80%（80k 训练已完成，评测待跑）
- [x] 至少保留 3 个 checkpoint（20k / 40k / last）方便对比（已存 020000 / 040000 / 080000，last→080000）
- [x] 训练 config 与日志已存档（`_runmeta/libero_smolvla_80k/run_info/` 含 snapshot + train_cmd.sh，train.log + train.exit）

***

## 8. 正式评测（M5：400 episode 标准口径）

### 8.1 标准评测命令

```bash
cd /root/autodl-tmp/src/lerobot
# 必须在仓库根目录执行，否则找不到 env 与策略插件

lerobot-eval \
  --policy.path=/root/autodl-tmp/outputs/libero_smolvla/checkpoints/last/pretrained_model \
  --env.type=libero \
  --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 \
  --eval.n_episodes=10 \
  --env.max_parallel_tasks=1 \
  --policy.n_action_steps=10 \
  --output_dir=/root/autodl-tmp/logs/eval_final
```

| 参数                         | 作用                                                | 调法                        |
| -------------------------- | ------------------------------------------------- | ------------------------- |
| `--policy.path`            | 指向 `.../checkpoints/last/pretrained_model` **目录** | 只拷 `.safetensors` 会缺预处理配置 |
| `--env.task`               | 4 个套件逗号分隔                                         | 只测一套就写一个名字                |
| `--eval.n_episodes`        | 每个任务跑几个 episode                                   | 正式 10；粗筛 3                |
| `--eval.batch_size`        | 同时跑几个环境                                           | 单卡固定 1，稳                  |
| `--env.max_parallel_tasks` | 任务级并行                                             | 单卡固定 1                    |
| `--policy.n_action_steps`  | 每次推理执行几步                                          | 10 更跟手；抖动大再试 50           |
| `--output_dir`             | 结果 JSON 与日志落盘位置                                   | 放数据盘                      |

- 4 套 × 10 任务 × 10 episode = **400 episode**，与论文口径一致。
- 单卡 `batch_size=1` + `max_parallel_tasks=1` 最稳；4090D 上 `batch_size=2` 可提速但偶尔有渲染进程竞争，先用 1。
- 预计 **1.5–3 小时**。想快：先 `--eval.n_episodes=3` 粗测。

### 8.2 读结果

终端会打印每个任务的 `pc_success` 和 95% Wilson 区间 `pc_success_ci95`，末尾给总平均；同时写入 `eval_info.json`。

**别忽略置信区间**：10 episode 的任务，9/10 的区间约 60%–98%；两个策略区间重叠就不算有差异。要写报告，建议 3 个 seed 取平均。

### 8.3 结果记录表（直接抄）

| 模型 / 步数                | Spatial | Object | Goal   | LIBERO-10 | 平均       | 备注         |
| ---------------------- | ------- | ------ | ------ | --------- | -------- | ---------- |
| smolvla\_base（0 步，零样本） | 未测 | 未测 | 未测 | 未测 | 未测 | 基线；按清单流程 80k 完成后再评完整四套 |
| 20k                    | **74.0**（task 5 最弱 40%） | **60.0**（弱 task0/3 各 30%） | **70.0**（弱 task3/6 各 20%） | **31.0**（7/10 任务 ≤30%） | **58.8** | loss 0.353, lr 2.5e-6, mem 7.67G, 2.1 step/s, 2h46m, ckpt 005000–020000；四套同 seed=1000 |
| 40k                    | 57.0 | — | — | — | — | 仅测 spatial（同 seed 快评），找峰值用 |
| 50k                    | 66.0 | — | — | — | — | 仅测 spatial（同 seed 快评），找峰值用 |
| 80k                    | **65.0**（弱 task0/2/3 各 50%） | **60.0**（弱 task3 20%、task0/1 各 30%） | **70.0**（弱 task9 10%、task6 30%） | **44.0**（7/10 任务 <60%） | **59.8** | ❌ **未达标**（2026-10-25 评测，400 回合，四套 EXIT=0）。训练本身正常（loss 0.202，lr 走到底），但 spatial 较 20k 的 74.0 **退步**，libero_10 全面偏弱——疑似过拟合/评测方差，见 §8.3 下方分析 |
| 论文 SmolVLA-0.45B       | 90      | 96     | —      | —         | **87.3** | 论文口径，仅参考   |
| π0.5（LeRobot 复现）       | 97.0    | 99.0   | 98.0   | 96.0      | **97.5** | 3B 模型，不同量级 |

> 单卡 4090D 复现 78–86% 属正常。若你到 85%+，已经可以写“复现成功”。

#### V2 训练结果（batch 64, 单条余弦 80k, 无 LR 跳变）

| 模型 / 步数 | Spatial | Object | Goal | LIBERO-10 | 平均 | 备注 |
|---|---|---|---|---|---|---|
| smolvla_base（0 步，零样本） | — | — | — | — | — | 未测 |
| **020000**（20k） | **59.0** | **36.0** | **63.0** | **28.0** | **46.5** | batch 64, 实测 ~1.12 s/step, 显存 15.2G；2026-10-27 评测，四套 EXIT=0 |
| **040000**（40k） | **65.0** | **52.0** | **80.0** | **49.0** | **61.5** | 同 v2 训练；spatial/goal 上升，libero_10 明显改善 |
| **060000**（60k） | **75.0** | **55.0** | **78.0** | **57.0** | **66.2** | ✅ **V2 best**；libero_10 = 57.0 远超 V1 的 44.0（单余弦对 OOD 提升最大） |
| **080000**（80k） | 71.0 | 47.0 | 73.0 | 54.0 | 61.2 | 80k 四套全回落，轻微过拟合，与 V1 同现象；余弦 LR 贴 0 后不再改善 |
| 论文 SmolVLA-0.45B | 90 | 96 | — | — | **87.3** | 论文口径，仅参考 |

> V2 配方：`--batch_size=64 --steps=80000 --policy.scheduler_decay_steps=80000`，单条余弦全程 80k，消除 v1 分段续训导致的 LR 跳变（20k 处 2.5e-6 → 8.6e-5）。训练量 512 万样本 = 18.7 epochs（v1 20k 仅 64 万样本 = 2.3 epochs）。

<!-- V3_TABLE_START -->
#### V3 训练结果（batch 64, 单条余弦 100k, RandomShift ±16px 增强 + weight decay 1e-2）

| 模型 / 步数 | Spatial | Object | Goal | LIBERO-10 | 平均 | 备注 |
|---|---|---|---|---|---|---|
| **050000**（50k） | **64.0** | **76.0** | **83.0** | **47.0** | **67.5** | 治过拟合配方 run |
| **060000**（60k） | **66.0** | **81.0** | **79.0** | **45.0** | **67.8** | V2 峰值同位对照 |
| **070000**（70k） | **75.0** | **77.0** | **75.0** | **38.0** | **66.2** |  |
| **080000**（80k） ⭐ | **70.0** | **75.0** | **81.0** | **49.0** | **68.8** | V2 终点同位对照 |
| **100000**（100k） | **60.0** | **79.0** | **82.0** | **46.0** | **66.8** | 论文对齐步数 |
| 论文 SmolVLA-0.45B | 90 | 96 | — | — | **87.3** | 论文口径，仅参考 |

> V3 配方：在 V2 基础上三项针对性改动——① 自定义 `RandomShift` 几何增强（±16px，crop+边缘复制，仅训练时启用）治"背题"过拟合；② `--policy.optimizer_weight_decay=1e-2`（V2 实为 1e-10 ≈ 零正则）；③ 步数对齐论文 100k + warmup 100。评测 5 ckpt × 4 套件 = 20 组。**判定**：❌ 未达标（best 68.8 < 80）。对照基准：V2 best 060000 = 66.2。
<!-- V3_TABLE_END -->

### 8.4 加速评测（可选）

```bash
# 只测一套、每任务 5 episode，用于 checkpoint 之间快速比较
lerobot-eval --policy.path=<ckpt> --env.type=libero --env.task=libero_10 \
  --eval.batch_size=1 --eval.n_episodes=5 --env.max_parallel_tasks=1
```

***

## 9. 新手最常踩的 46 个坑（按阶段分类）

### A. 租机与存储（5 个）

| #  | 现象                                            | 原因                    | 解决                               |
| -- | --------------------------------------------- | --------------------- | -------------------------------- |
| A1 | `OSError: [Errno 28] No space left on device` | HF 缓存/权重默认写系统盘 30GB   | 做 §3.2 软链；`du -sh ~/.cache/*` 排查 |
| A2 | 关机再开提示“GPU 不足”                                | 按量计费关机不预留 GPU         | 等释放 / 克隆实例 / 换同区主机；重要阶段一次开完      |
| A3 | 数据盘没扩容，checkpoint 存不下                         | 每个 ckpt \~5G × 8–10 个 | 建实例时就扩到 100GB                    |
| A4 | 无卡模式下 pip 装 torch 卡死/OOM                      | 无卡只有 0.5 核 2GB        | 无卡只做下载；编译安装用正常模式                 |
| A5 | 跑完忘关机，一夜烧掉 20 元                               | 按开关机时长计费              | 用 §5.7 条件关机模板（`&&` 条件判断，成功才关）    |

### B. 安装与环境（8 个）

| #  | 现象                                            | 原因                                                                           | 解决                                                                                                                                      |
| -- | --------------------------------------------- | ---------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| B1 | `GLFW error` / 需要显示器                          | 没设 `MUJOCO_GL=egl`                                                           | `export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`                                                                                            |
| B2 | `libEGL.so.1: cannot open shared object file` | 缺 EGL 库                                                                      | `apt install libegl1 libegl1-mesa-dev`                                                                                                  |
| B3 | `cmake --version returned non-zero`           | pip 构建隔离里没有 cmake                                                            | `pip install --no-build-isolation ...`                                                                                                  |
| B4 | `Unknown policy type: smolvla`                | 只装了核心包                                                                       | `pip install -e ".[smolvla]"`                                                                                                           |
| B5 | `No module named libero`                      | 没装 `[libero]` 或非 Linux                                                       | 确认 Linux + `.[libero]`                                                                                                                  |
| B6 | `Could not load libtorchcodec` / 解码失败         | torchcodec 与 torch 2.8.0+cu128 不兼容（本机实测 0.8/0.8.1/0.9.1/0.11.1 加载即 std::bad_alloc 崩溃）                                                                  | `pip uninstall torchcodec && pip install av`，训练命令用 `--dataset.video_backend=pyav`                                                                                             |
| B7 | `pip install lerobot` 后没有 `lerobot-train`     | 0.6 起 PyPI 包不含训练依赖                                                           | 用源码 `-e ".[training,...]"`                                                                                                              |
| B8 | 首次 `import libero` 卡住/报 `EOFError`，屏幕上等你输入    | 没有 `~/.libero/config.yaml` 时，`libero/__init__.py` 会 `input()` 问你“是否自定义数据集路径” | 先手动 `mkdir -p ~/.libero`，交互式跑一次 `python -c "import libero"` 并回答 `N`；或 `printf 'n\n' > /tmp/ans && python -c "import libero" < /tmp/ans` |

### C. 数据（4 个）

| #  | 现象                                           | 原因               | 解决                                                                |
| -- | -------------------------------------------- | ---------------- | ----------------------------------------------------------------- |
| C1 | 下载卡在 0% / 超时                                 | 直连 HF            | `HF_ENDPOINT=https://hf-mirror.com`；或 `source /etc/network_turbo` |
| C2 | `LocalEntryNotFoundError`                    | 缓存半截 / 离线标志      | 删掉对应缓存目录重下；确认没设 `HF_HUB_OFFLINE=1`                                |
| C3 | 训练报 `KeyError: 'observation.images.camera1'` | 用了 base 自带特征，没置空 | 加 `--policy.input_features=null`                                  |
| C4 | 换了数据集后成功率对不上论文                               | 数据集 revision 变了  | 固定 `--dataset.revision=<sha>`                                     |

### D. 训练（9 个）

| #  | 现象                                    | 原因                               | 解决                                                                                        |
| -- | ------------------------------------- | -------------------------------- | ----------------------------------------------------------------------------------------- |
| D1 | `FileExistsError: ... already exists` | `output_dir` 已存在且非 resume        | 换目录名，或加 `--resume=true`                                                                   |
| D2 | `401`/`no token`/要求登录                 | `push_to_hub` 默认 True            | `--policy.push_to_hub=false`                                                              |
| D3 | `CUDA out of memory`                  | batch 太大 / 开了在线评测                | 降 batch；`--env_eval_freq=0`                                                               |
| D4 | 训练极慢，GPU 利用率低                         | DataLoader 瓶颈 / 数据在网络盘           | `--num_workers=12`，数据放 `/root/autodl-tmp`                                                 |
| D5 | loss NaN                              | AMP 溢出 / lr 太大                   | 关 `use_amp`；`--policy.optimizer_lr=5e-5`                                                  |
| D6 | 80k 步后 LR 一直是 2.5e-6                  | 默认 `scheduler_decay_steps=30000` | 加 `--policy.scheduler_decay_steps=80000`                                                  |
| D7 | 跑一半断线，白跑                              | 没设 `save_freq` / 没日志             | `--save_freq=10000`；tmux 里 `... 2>&1 | tee train.log`                                        |
| D8 | 想从 ckpt 续训但报配置冲突                      | 传了 `--policy.path` 又传了别的冲突字段     | 用 `--config_path=<run>/checkpoints/last/pretrained_model/train_config.json --resume=true` |
| D9 | 只用 2 路相机却报相机数不符                       | 直接沿用 base 的 3 相机配置               | 同样加 `--policy.input_features=null`（必要时 `--policy.empty_cameras=1`）                        |

### E. 评测（6 个）

| #  | 现象                      | 原因                               | 解决                                              |
| -- | ----------------------- | -------------------------------- | ----------------------------------------------- |
| E1 | 评测启动就 `EGLError`        | 没设渲染后端                           | 命令前加 `MUJOCO_GL=egl`                            |
| E2 | 评测卡住不动                  | 并行 env 抢 GPU / 渲染进程死锁            | `batch_size=1`、`max_parallel_tasks=1`           |
| E3 | 成功率异常低（<10%）但训练 loss 正常 | `n_action_steps` 与训练不一致 / 控制模式不对 | 试 `--policy.n_action_steps=10/50`；确认 `relative` |
| E4 | 动作幅度巨大、机械臂乱飞            | 归一化统计没用上                         | 用训练产出的 `pretrained_model` 目录整体加载，别只拷权重          |
| E5 | 资产下载失败                  | 首次评测要拉 `lerobot/libero-assets`   | 设 `HF_ENDPOINT`；确认 `~/.cache/libero` 软链可写       |
| E6 | 400 episode 跑了半天        | 单卡串行                             | 先 `n_episodes=3` 粗筛，最后再跑全量                      |

### F. 方法与认知（6 个）

| #  | 误区                 | 正解                                             |
| -- | ------------------ | ---------------------------------------------- |
| F1 | “必须自己采数据”          | LIBERO 公共数据集就是标准训练集；真机数据是后期的事                  |
| F2 | “batch 越大越好”       | 24GB 上 batch 32–64 足够；再大只增加等待，不提升效果            |
| F3 | “loss 不降就是坏了”      | flow matching 的 loss 尺度与分类任务不同，**看评测成功率**      |
| F4 | “步数越多越好”           | 40k–80k 通常到平台；100k+ 收益递减，先查数据和评测               |
| F5 | “随便哪个 4090 教程都能照抄” | π0/GR00T 的 80GB 教程不适用；版本、CLI 参数变化很快，以 tag 文档为准 |
| F6 | “跑出 60% 就是复现失败”    | 单卡、单 seed、10 episode 的方差很大；先跑 3 seed、看置信区间     |

### G. 离线部署：本地下再上传（Ubuntu / Windows，8 个）

| #  | 现象                                          | 原因                                                         | 解决                                                                             |
| -- | ------------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------ |
| G1 | `scp : 无法将“scp”项识别为 cmdlet`                 | Windows 没装 OpenSSH 客户端                                     | 管理员 PowerShell：`Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0` |
| G2 | `hf : 无法将“hf”项识别为 cmdlet`                   | Python 的 `Scripts` 目录不在 PATH                               | `python -m pip show -f huggingface_hub` 找到 `Scripts\hf.exe`，把该目录加进 PATH 后重开终端  |
| G3 | 上传的 tar 只有几 KB，或服务器解出来是空文件                  | 文件在 OneDrive 目录下，是“仅在线”占位符                                 | 移到 `D:\smolvla` 这类非同步目录，或右键“始终保留在此设备上”                                         |
| G4 | 训练报 `FileNotFoundError: .../meta/info.json` | tar 解包多套了一层目录                                              | `ls` 看层级；必要时 `--strip-components=1` 重解                                         |
| G5 | 已离线，仍报 `couldn't connect to huggingface.co` | 漏了 VLM 基座，或没设 `--policy.vlm_model_name` / `HF_HUB_OFFLINE` | 补齐基座目录 + 显式覆盖                                                                  |
| G6 | 每次评测还在下载资产                                  | `assets/` 下 4 个子目录不齐，触发了整包重下                               | 检查 `articulated_objects` 等 4 个目录                                               |
| G7 | `scp -r` 传几千个小文件，慢且中断                       | 没打包                                                        | 先 `tar -cf`，传单个文件                                                              |
| G8 | 删了 tar 才发现解包不全 / 哈希对不上                      | 没校验就清理                                                     | 先 `sha256sum` 与 `Get-FileHash` 比对，再跑 §5.4.5 三条校验，最后 `rm *.tar`                 |

> 补一条心法：**离线方案的价值是“可控”，不是“更快”**。服务器能直下就别折腾；只有当直下反复失败、或你要在多个实例间复用同一份文件时，才值得走 G 组。

***

## 10. 故障排查决策树（按症状走）

**训练起不来**

1. `lerobot-train: command not found` → 没激活环境 / 没装 `[training]`
2. `Unknown policy type` → 没装 `[smolvla]`
3. `KeyError: observation.images.*` → 没加 `--policy.input_features=null`
4. `401 / token` → 没加 `--policy.push_to_hub=false`
5. `FileExistsError` → 换 `output_dir`

**训练能起但很慢**

1. 看 `mem_gb`：<10 说明 batch 太小，升到 32/64
2. 看 `data_s` vs `updt_s`：接近 → 加 `num_workers`，数据移到本地盘
3. `nvidia-smi` 利用率 <50% → CPU/IO 瓶颈，同上
4. 用了 `osmesa` 渲染 → 换回 `egl`

**评测跑不出成功率**

1. `EGLError` → 环境变量
2. 卡住 → 降并行
3. 0% → 先用 20k+ 的 ckpt，确认不是 smoke ckpt；再查 `n_action_steps`
4. 波动大 → 提高 `n_episodes`，跑多 seed

***

## 11. 后期接 SO-101 真机（现在只需预留设计）

现在不用做，但训练时把下面几件事做对，后期迁移会非常顺：

### 11.1 数据格式对齐（最重要）

| 项         | LIBERO（现在）                            | SO-101（将来）                   | 处理                           |
| --------- | ------------------------------------- | ---------------------------- | ---------------------------- |
| 数据集格式     | LeRobot v3.0                          | 也用 LeRobot 录制                | 一致，直接复用                      |
| 相机键名      | `observation.images.image` / `image2` | 录制时自定义（front/wrist）          | 用 `--rename_map` 映射到模型期望键名   |
| state 维度  | 8                                     | 6（SO-101 关节）                 | SmolVLA 补零到 32，天然兼容          |
| action 维度 | 7                                     | 6                            | 同上，`output_features` 会按数据集覆盖 |
| 语言指令      | 数据集自带                                 | 录制时的 `--dataset.single_task` | **必须与推理时完全一致**               |

### 11.2 采集建议（照 SmolVLA 论文的做法）

- **50 条起步**，且要“每个变化重复多次”：例如 5 个方块位置 × 每个 10 条。
- 25 条通常不够（官方明确说效果差）。
- 光照、背景、初始位姿要有变化，但任务描述必须统一。
- 相机固定、对焦清晰、帧率稳定 30fps。

### 11.3 迁移策略

| 策略                  | 做法                                   | 适用                  |
| ------------------- | ------------------------------------ | ------------------- |
| 从 LIBERO ckpt 继续微调  | `--policy.path=<你的 LIBERO ckpt>`     | 想保留“抓取/放置”先验，数据少时更稳 |
| 从 `smolvla_base` 重训 | `--policy.path=lerobot/smolvla_base` | 真机数据 ≥50 条，且形态差异大   |
| 两个都跑，取好的            | 各 20k 步对比                            | 最稳，成本也就多几块钱         |

> 真机推理命令在 0.6.x 是 `lerobot-rollout`（策略部署与录制已解耦）；具体参数以你届时版本的 `docs/source/smolvla.mdx` 为准，别照抄旧版 `lerobot-record` 教程。

***

## 12. 附录

### 12.1 一键安装脚本（保存为 `setup.sh`）

```bash
#!/usr/bin/env bash
set -euo pipefail

echo "== 0. 体检 =="
nvidia-smi
nproc; free -g; df -h / /root/autodl-tmp
python -V

echo "== 1. 目录与缓存软链 =="
mkdir -p /root/autodl-tmp/{hf_cache,libero_cache,datasets,outputs,logs,src}
mkdir -p /root/.cache
ln -sfn /root/autodl-tmp/hf_cache     /root/.cache/huggingface
ln -sfn /root/autodl-tmp/libero_cache /root/.cache/libero

echo "== 2. 环境变量 =="
grep -q "HF_ENDPOINT" ~/.bashrc || cat >> ~/.bashrc <<'EOF'

# ===== SmolVLA / LIBERO =====
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0
EOF
source ~/.bashrc

echo "== 3. pip 源与工具 =="
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
pip config set global.trusted-host pypi.tuna.tsinghua.edu.cn
pip install -U pip setuptools wheel hf_transfer

echo "== 4. 系统依赖 =="
apt-get update
apt-get install -y --no-install-recommends \
  build-essential cmake ninja-build pkg-config git curl \
  libegl1 libgl1 libglib2.0-0 libglew-dev libosmesa6-dev \
  libegl1-mesa-dev libgles2-mesa-dev libgl1-mesa-dev

echo "== 5. LeRobot v0.6.1 =="
cd /root/autodl-tmp/src
[ -d lerobot ] || git clone --depth 1 --branch v0.6.1 https://github.com/huggingface/lerobot.git
cd lerobot
pip install --no-build-isolation "hf-egl-probe==1.0.2"
pip install --no-build-isolation -e ".[training,smolvla,libero]"
conda install -y -c conda-forge ffmpeg=7.1.1

echo "== 6. 验证 =="
python -c "import torch, lerobot, libero, robosuite, mujoco; print('imports ok')"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -m egl_probe.get_available_devices
lerobot-train --help | head -5

echo "== 7. 模型与数据（已下好可跳过） =="
mkdir -p /root/autodl-tmp/SmolVLA/{smolvla_base,SmolVLM2-500M-Video-Instruct,libero,libero-assets}
hf download lerobot/smolvla_base \
  --local-dir /root/autodl-tmp/SmolVLA/smolvla_base
hf download HuggingFaceTB/SmolVLM2-500M-Video-Instruct \
  --local-dir /root/autodl-tmp/SmolVLA/SmolVLM2-500M-Video-Instruct
hf download lerobot/libero --repo-type=dataset \
  --local-dir /root/autodl-tmp/SmolVLA/libero
hf download lerobot/libero-assets --repo-type=dataset \
  --local-dir /root/autodl-tmp/SmolVLA/libero-assets

du -sh /root/autodl-tmp/SmolVLA/{smolvla_base,SmolVLM2-500M-Video-Instruct,libero,libero-assets}

echo "DONE"
```

### 12.2 命令速查

> 所有训练命令都按 **§5.7 标准模板** 执行：先参数快照 → 再写 train\_cmd.sh → tmux 前台启动（tee 留档）→ 条件关机。下面只列出 `train_cmd.sh` 里的**命令本体**，外围包装（快照/tmux/条件关机）严格遵循 §5.7。

**训练命令本体**（粘进 `train_cmd.sh` 的 lerobot-train 部分）：

```bash
# === 冒烟训练（300 步）===
lerobot-train --policy.path=/root/autodl-tmp/SmolVLA/smolvla_base \
  --policy.vlm_model_name=/root/autodl-tmp/SmolVLA/SmolVLM2-500M-Video-Instruct \
  --policy.input_features=null --policy.output_features=null \
  --policy.push_to_hub=false --policy.device=cuda --policy.use_amp=false \
  --dataset.repo_id=lerobot/libero --dataset.root=/root/autodl-tmp/SmolVLA/libero \
  --dataset.video_backend=pyav \
  --output_dir=/root/autodl-tmp/outputs/smoke_300 \
  --job_name=smoke_300 \
  --batch_size=16 --steps=300 --save_freq=300 --log_freq=20 \
  --num_workers=4 --env_eval_freq=0 --seed=1000
# ⚠️ 冒烟不关机，还要跑评测

# === 20k 验证训练 ===
lerobot-train --policy.path=/root/autodl-tmp/SmolVLA/smolvla_base \
  --policy.vlm_model_name=/root/autodl-tmp/SmolVLA/SmolVLM2-500M-Video-Instruct \
  --policy.input_features=null --policy.output_features=null \
  --policy.push_to_hub=false --policy.device=cuda --policy.use_amp=false \
  --policy.scheduler_decay_steps=80000 \
  --dataset.repo_id=lerobot/libero --dataset.root=/root/autodl-tmp/SmolVLA/libero \
  --dataset.video_backend=pyav \
  --output_dir=/root/autodl-tmp/outputs/libero_smolvla \
  --job_name=libero_smolvla \
  --batch_size=32 --steps=20000 --save_freq=5000 --log_freq=50 \
  --num_workers=8 --env_eval_freq=0 --seed=1000
# ⚠️ 20k 也不自动关机，先看评测

# === 80k 正式训练（续训方式 A；后 3 个覆盖参数同机可省、跨机必带）===
lerobot-train --config_path=/root/autodl-tmp/outputs/libero_smolvla/checkpoints/last/pretrained_model/train_config.json \
  --resume=true --steps=80000 --save_freq=10000 \
  --output_dir=/root/autodl-tmp/outputs/libero_smolvla \
  --dataset.root=/root/autodl-tmp/SmolVLA/libero \
  --policy.vlm_model_name=/root/autodl-tmp/SmolVLA/SmolVLM2-500M-Video-Instruct
# ⚠️ 续训日志用追加模式：bash train_cmd.sh 2>&1 | tee -a train.log

# === 80k 正式训练（方式 B：新 run）===
lerobot-train --policy.path=/root/autodl-tmp/outputs/libero_smolvla/checkpoints/last/pretrained_model \
  --policy.input_features=null --policy.output_features=null \
  --policy.push_to_hub=false --policy.device=cuda --policy.use_amp=false \
  --policy.scheduler_decay_steps=60000 \
  --dataset.repo_id=lerobot/libero --dataset.root=/root/autodl-tmp/SmolVLA/libero \
  --dataset.video_backend=pyav \
  --output_dir=/root/autodl-tmp/outputs/libero_smolvla_80k \
  --batch_size=32 --steps=60000 --save_freq=10000 --log_freq=50 \
  --num_workers=8 --env_eval_freq=0 --seed=1000

# === 训练（离线版：模型已本地下好）===
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
lerobot-train --policy.path=/root/autodl-tmp/SmolVLA/smolvla_base \
  --policy.vlm_model_name=/root/autodl-tmp/SmolVLA/SmolVLM2-500M-Video-Instruct \
  --policy.input_features=null --policy.output_features=null \
  --policy.push_to_hub=false --policy.device=cuda --policy.use_amp=false \
  --dataset.repo_id=lerobot/libero --dataset.root=/root/autodl-tmp/SmolVLA/libero \
  --dataset.video_backend=pyav \
  --output_dir=/root/autodl-tmp/outputs/run1 \
  --batch_size=32 --steps=80000 --save_freq=10000 --log_freq=50 \
  --num_workers=8 --env_eval_freq=0 --seed=1000
```

**评测命令**（直接跑，不用模板）：

```bash
# 全量评测（4 套 × 10 任务 × 10 episode = 400）
lerobot-eval --policy.path=/root/autodl-tmp/outputs/run1/checkpoints/last/pretrained_model \
  --env.type=libero --env.task=libero_spatial,libero_object,libero_goal,libero_10 \
  --eval.batch_size=1 --eval.n_episodes=10 --env.max_parallel_tasks=1 \
  --policy.n_action_steps=10 \
  --output_dir=/root/autodl-tmp/outputs/run1/eval_final

# 单任务快速评测
lerobot-eval --policy.path=<ckpt> --env.type=libero --env.task=libero_spatial \
  --env.task_ids='[0]' --eval.batch_size=1 --eval.n_episodes=5 --env.max_parallel_tasks=1

# 离线环境下评测（同样要带 vlm_model_name，并开 OFFLINE）
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
lerobot-eval --policy.path=/root/autodl-tmp/outputs/run1/checkpoints/last/pretrained_model \
  --policy.vlm_model_name=/root/autodl-tmp/SmolVLA/SmolVLM2-500M-Video-Instruct \
  --env.type=libero --env.task=libero_spatial \
  --eval.batch_size=1 --eval.n_episodes=10 --env.max_parallel_tasks=1
```

**条件关机模板**（❌ 别再用 `; shutdown` 这种裸命令了！）：

```bash
# 条件关机：训练 + 评测都过才关机，失败保留现场
OUTPUT_DIR=/root/autodl-tmp/outputs/run1
META=/root/autodl-tmp/outputs/_runmeta/run1
while [ ! -f "$META/train.exit" ]; do sleep 60; done; [ "$(cut -d= -f2 $META/train.exit)" = "0" ] && \
  lerobot-eval ... --output_dir="$OUTPUT_DIR/eval_final" && \
  shutdown -h +1 "SmolVLA 完成（run: $OUTPUT_DIR），60 秒后关机" \
  || { echo "❌ 失败/中断，保留现场"; tail -50 "$META/train.log"; }
# ⚠️ 这个脚本要在**另一个 pane/终端**单独跑
# ⚠️ 冒烟和 20k 阶段先不要 shutdown，把 shutdown 那几行注释掉
```

**定期检查 checkpoint 有没有在存**（另开终端）：

```bash
# 每 30 秒看一下有没有新 checkpoint 出来（说明定期断点正常工作）
watch -n 30 "ls -lh /root/autodl-tmp/outputs/run1/checkpoints/"
```

### 12.3 目录结构（建好后应是这样）

```
/root/autodl-tmp/
├── SmolVLA/                    # 📂 4 个仓库统一放这里（本项目根目录）
│   ├── smolvla_base/           #   lerobot/smolvla_base 策略权重
│   ├── SmolVLM2-500M-Video-Instruct/  #   HuggingFaceTB/SmolVLM2 VLM 基座
│   ├── libero/                 #   lerobot/libero 数据集（meta/ data/ videos/）
│   └── libero-assets/          #   lerobot/libero-assets 仿真资产（articulated_objects/ ...）
├── hf_cache/                   # HF 权重与数据集缓存（HF_HOME）
├── libero_cache/               # LIBERO 资产（~/.cache/libero 软链目标）
├── outputs/
│   ├── smoke_300/
│   └── libero_smolvla/
│       ├── checkpoints/{last,020000,...}/pretrained_model/   # train_config.json 也在这里面
│       └── train.log
├── logs/
└── src/lerobot/                # 源码（v0.6.1）
```

### 12.4 实验记录模板（每次 run 复制一份）

```yaml
run_id: libero_smolvla_80k_seed1000
date: 2026-10-xx
git: lerobot v0.6.1
model_base: lerobot/smolvla_base
dataset: lerobot/libero @ sha=<commit>
trainable: expert_only (default) | full
batch: 32  steps: 80000  lr: 1e-4  amp: true
gpu: RTX 4090D 24GB x1
wall_clock: 12h40m
peak_vram_gb: 18.2
eval: n_episodes=10/task, batch=1, n_action_steps=10, seed=1000
result: spatial 82 | object 90 | goal 84 | long 68 | avg 81.0
notes: 前 5k 步 data_s 偏高，num_workers 8->12 后正常
```

### 12.5 参考链接

- LeRobot 仓库（v0.6.1）：<https://github.com/huggingface/lerobot/tree/v0.6.1>
- LIBERO 文档（含数据集对比、评测口径）：`docs/source/libero.mdx`
- SmolVLA 文档：`docs/source/smolvla.mdx`
- 模型：<https://huggingface.co/lerobot/smolvla_base>
- 数据集：<https://huggingface.co/datasets/lerobot/libero> （镜像 `hf-mirror.com`）
- 论文：SmolVLA: A vision-language-action model for affordable and efficient robotics (arXiv 2506.01844)
- LIBERO 论文：Benchmarking Knowledge Transfer for Lifelong Robot Learning (arXiv 2306.03310)
- AutoDL 文档：<https://www.autodl.com/docs/> （计费 / 无卡模式 / 数据盘）

### 12.6 一键下载脚本（Ubuntu / Windows 双套，可选）

两个脚本做同一件事：下载 4 个仓库 → 校验 → 清 `.cache` → 打 4 个 tar → 输出 SHA256。**不上传、不删你的文件**（重复运行只覆盖同名 tar），输出全英文以避免终端编码问题。

**Ubuntu**

```bash
bash local_download.sh                # 默认落到 ~/smolvla
bash local_download.sh /data/smolvla  # 也可指定目录
```

**Windows**

```powershell
cd <脚本所在目录>
powershell -ExecutionPolicy Bypass -File .\local_win_download.ps1 -Root D:\smolvla
```

| 脚本                       | 系统      | 默认目录         | 说明                                     |
| ------------------------ | ------- | ------------ | -------------------------------------- |
| `local_download.sh`      | Ubuntu  | `~/smolvla`  | 需要 `bash`、`hf`、`tar`、`scp`、`sha256sum` |
| `local_win_download.ps1` | Windows | `D:\smolvla` | `-ExecutionPolicy Bypass` 只对本进程生效      |

- 想手工逐步做，就按 §5.4.2 → §5.4.3 走，效果完全一致。
- 脚本跑完只打印 scp 模板，**不会**自动登录服务器。

### 12.7 Ubuntu / Windows 命令对照表

只列本项目真正用到的。服务器端永远是 Ubuntu，Windows 只影响「本地那一段」。

| 操作       | Ubuntu                                            | Windows（PowerShell）                                          |
| -------- | ------------------------------------------------- | ------------------------------------------------------------ |
| 设临时环境变量  | `export HF_ENDPOINT=...`                          | `$env:HF_ENDPOINT = "..."`                                   |
| 设永久环境变量  | `echo 'export X=1' >> ~/.bashrc`                  | `[Environment]::SetEnvironmentVariable("X","1","User")`      |
| 建目录      | `mkdir -p ~/smolvla`                              | `New-Item -ItemType Directory -Force D:\smolvla`             |
| 进目录      | `cd ~/smolvla`                                    | `Set-Location D:\smolvla`                                    |
| 看剩余空间    | `df -h ~/smolvla`                                 | `Get-PSDrive D`                                              |
| 看目录体积    | `du -sh *`                                        | `Get-ChildItem \| Measure-Object Length -Sum`                |
| 列文件      | `ls -lh`                                          | `Get-ChildItem`                                              |
| 判断文件存在   | `test -f x && echo OK`                            | `Test-Path .\x`                                              |
| 删目录      | `rm -rf ./*/.cache`                               | `Remove-Item -Recurse -Force .\.cache`                       |
| 打包       | `tar -cf x.tar dir`                               | `tar -cf x.tar dir`                                          |
| 解包到指定目录  | `tar -xf x.tar -C dst`                            | `tar -xf x.tar -C dst`                                       |
| 算 SHA256 | `sha256sum x.tar`                                 | `Get-FileHash x.tar -Algorithm SHA256`                       |
| 下载 HF 仓库 | `hf download repo --local-dir .`                  | 同左                                                           |
| 登录服务器    | `ssh -p 38123 root@host`                          | `ssh -p 38123 root@host`                                     |
| 传文件      | `scp -P 38123 x.tar root@host:/path`              | `scp -P 38123 x.tar root@host:/path`                         |
| 断点续传     | `rsync -avP -e "ssh -p 38123" ./ root@host:/path` | 无内置 → WSL2 里的 rsync，或 WinSCP                                 |
| 实时看日志    | `tail -f train.log`                               | `Get-Content train.log -Wait -Tail 20`                       |
| 看进程      | `ps aux \| grep lerobot`                          | `Get-Process python`                                         |
| 杀进程      | `kill -9 <PID>`                                   | `Stop-Process -Id <PID> -Force`                              |
| 定时关机     | `shutdown -h now`                                 | `Stop-Computer`                                              |
| 后台跑长任务   | `nohup cmd > log 2>&1 &`                          | `Start-Process -NoNewWindow cmd -RedirectStandardOutput log` |
| 端口转发看网页  | `ssh -L 8080:localhost:8080 -p ...`               | 同左（Windows 也自带 ssh）                                          |

> 一句话记法：**除了「设变量、建目录、算哈希、续传」这四件事，其余命令两边逐字相同**；而服务器端只认 Ubuntu，你在 Windows 上 SSH 进去后敲的还是左边那列。

***

## 13. 一句话行动清单

1. 租 4090D + 100GB 数据盘 + PyTorch2.7/CUDA12.8/Py3.12 基础镜像
2. 软链缓存到数据盘，写好 5 个环境变量
3. apt 装 EGL，pip 装 `.[training,smolvla,libero]`（`--no-build-isolation`）
4. 下 4 个仓库（`smolvla_base` / `SmolVLM2` 基座 / `libero` 数据 / `libero-assets`），或本地下好再上传，跑通 `egl_probe`
5. 300 步冒烟 + 单任务 3 episode 评测
6. 20k 步 → spatial ≥55% → 续到 80k
7. 400 episode 四套评测 → 填结果表 → 存档
8. 关机

