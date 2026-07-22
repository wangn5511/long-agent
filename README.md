# model-agent：OpenHands 驱动的 rightArm MLOps 循环

本仓库以定制版 OpenHands 为主体，同时交付中央 MLOps 循环、任务 Agent 规范、
rightArmPerception 模型代码和统一运行配置。

## 目录

```text
model-agent/
├── openhands/                       # OpenHands 后端
│   └── mlops_loop/                  # 中央 MLOps 循环
├── frontend/                        # OpenHands 网页
├── workspace/rightArmPerception/    # 数据、训练、评估与导出代码
├── skills/                          # OpenHands 任务规范
├── configs/                         # 非秘密运行配置
├── docker-compose.yml               # OpenHands + Loop Controller
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
