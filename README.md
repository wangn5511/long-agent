# model-agent：OpenHands 驱动的 rightArm MLOps 循环

本仓库以定制版 OpenHands 为主体，同时交付中央 MLOps 循环、任务 Agent 规范、
rightArmPerception 模型代码和统一运行配置。

最近一次真实 Run 的数据、基线、四轮训练方案、失败根因和审计时间线：

**[手势模型持续改进 Run 报告](docs/reports/run_d6026a6b4e1bc3912b170e98-summary.html)**

## 核心代码导航

本仓库包含一份可独立构建的 OpenHands fork，因此文件数量会明显多于普通训练项目。
日常开发主要集中在以下目录：

| 层次 | 目录 | 作用 |
| --- | --- | --- |
| Agent Harness | `openhands/app_server/mlops_task/`、`skills/` | 工具边界、任务规范、长时作业观察与 Agent 执行环境 |
| Model Loop Engine | `openhands/mlops_loop/` | Run 状态机、事件账本、调度、恢复、模型晋级和回滚 |
| Model Adapter | `workspace/rightArmPerception/rightarm_mlops/` | 数据、训练、评测、导出和模型特定协议 |
| Runtime | `containers/model-agent/`、`docker-compose.yml` | OpenHands 与 Loop Controller 的隔离运行环境 |
| Configuration | `configs/` | 非秘密镜像、作业和运行参数 |
| Reports | `docs/reports/` | 可分享的实验汇总；原始产物不进入 Git |

### 上游运行时

以下内容主要来自 OpenHands 或模型训练依赖。它们不是 Model Agent 的业务入口，
但当前仍用于从源码构建运行镜像或复现实验，因此暂不删除：

- `frontend/`：构建 OpenHands 网页；
- `openhands/` 中除上述定制模块外的代码：OpenHands 应用运行时；
- `workspace/rightArmPerception/training/`：手势模型和 MMAction2 训练实现；
- `enterprise/`、`openhands-ui/`、`kind/`：上游快照内容，后续可在拆分
  Runtime 仓库时移出主仓库。

## 目录概览

```text
model-agent/
├── openhands/
│   ├── app_server/mlops_task/       # Agent Harness
│   └── mlops_loop/                  # Model Loop Engine
├── workspace/rightArmPerception/
│   └── rightarm_mlops/              # 手势模型 Adapter
├── skills/                          # Agent 行动规范
├── configs/                         # 非秘密运行配置
├── containers/model-agent/          # 定制运行镜像
├── docs/reports/                    # 实验汇总报告
├── docker-compose.yml               # OpenHands + Loop Controller
├── model-agent                      # data/train/full/export/improve 入口
├── setup.sh                         # 首次准备
└── start.sh                         # 一键启动
```

## 首次启动

```bash
git clone https://git.cyto.one/wangning/model-agent.git
cd model-agent
./start.sh
```

`setup.sh` 会为当前 Linux 用户创建独立端口、状态目录和控制器令牌，并在本机
没有镜像时构建定制 OpenHands 镜像。`start.sh` 同时启动：

1. OpenHands 网页和 Agent；
2. 中央 MLOps Loop Controller。

网页只绑定服务器的 `127.0.0.1`。使用 VSCode SSH 时，在“端口”面板转发脚本
打印的端口，然后打开对应的 `http://localhost:<端口>`。首次进入 Settings，
配置自己的 LLM 模型、Base URL 和 API Key。

## 中央循环的边界

- OpenHands Agent 不直接持有宿主机 Docker socket。
- 只有 Loop Controller 可以执行经过白名单约束的模型作业。
- 数据、checkpoint、PKL、ONNX、RKNN、COS 密钥和 LLM 密钥都不进入 Git。
- 默认 `MLOPS_TRIGGER_MODE=manual`，因此初次启动不会误触发训练。
- 完成基线制品注册和镜像 digest 锁定后，可切换为 `local` 或 `command` 触发；
  默认轮询周期是 86400 秒，即 24 小时。
- 生产部署任务继续保留人工确认门，不会因为 Agent 的自然语言回答自动发布。

中央循环规范位于 `openhands/mlops_loop/specs/`，确定性模型作业位于
`workspace/rightArmPerception/rightarm_mlops/`。

## 常用操作

```bash
./status.sh
./stop.sh
```

## 四种人工入口

启动服务并在 OpenHands Settings 中配置 LLM 后，可以从仓库根目录按阶段创建
一个可审计 Run：

```bash
# COS 下载、完整性校验、清洗、生成并合并 PKL
./model-agent trigger data \
  --cos-uri cos://bucket/prefix/batch-001/

# 从现有 PKL 开始训练，随后评估和导出
./model-agent trigger train \
  --pkl /path/to/merged-gesture-11.pkl

# 从 COS 开始执行数据到模型导出的完整循环
./model-agent trigger full \
  --cos-uri cos://bucket/prefix/batch-001/

# 用现有 checkpoint 重新评估并导出 ONNX/RKNN
./model-agent trigger export \
  --checkpoint /path/to/best.pth

# 查看最近的 Run、状态和当前步骤
./model-agent status
```

`train` 所需基线模型、`export` 所需评估 PKL，以及 `data/full` 所需三个
RTM ONNX 文件可以直接追加命令参数；也可以在私有 `.env` 中设置
`MLOPS_BASE_MODEL_PATH`、`MLOPS_DEFAULT_DATASET_PATH` 和三个
`MLOPS_RTM_*_PATH`。`setup.sh` 会发现这台服务器上已有的兼容文件。

执行前可追加 `--dry-run`，只检查参数和输入，不复制文件、不创建 Run，
也不会启动训练。正式触发时，本地文件先进入当前用户的私有导入区并经过
SHA-256 校验，然后由控制器封存；OpenHands 只看到制品 ID。COS 凭据仅以
只读方式挂载给控制器，绝不进入 Agent 会话或 Git。
