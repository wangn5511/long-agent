# model-agent 配置

本目录只保存可以提交到 Git 的非秘密配置。LLM API Key 在 OpenHands 网页的
Settings 中由每个用户单独保存；腾讯云密钥、控制器令牌和其他凭据必须放在
`.model-agent/secrets/` 或管理员提供的只读 secret mount 中。

- `model-agent.env.example`：由根目录 `setup.sh` 解析为私有 `.env`。
- `pipeline.example.yaml`：数据、训练、评估、导出和部署策略的交付模板。
- `jobs/external-jobs.json`：允许中央控制器调用的外部命令前缀。
- `jobs/docker-images.example.json`：生产作业镜像目录模板；所有镜像必须锁定
  到真实的 `sha256` digest。
- `generated/`：本机解析后的镜像目录和制品引用，永远不提交到 Git。

中央循环的协议级 LoopSpec 和 RunSpec 位于
`openhands/mlops_loop/specs/`，任务 Agent 规范位于
`skills/agentic-mlops-task-agent/SKILL.md`。
