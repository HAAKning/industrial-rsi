# runtime-v5 企业赛整改

此目录为镜像 3 的选手代码及复现依据。实际构建镜像为 `team/harness:runtime-v5`，SDK 1.0.5，Linux/amd64。

- `app/` 是参与构建镜像的选手入口、策略与父检查点审计代码。
- `baseline_train.patch` 是相对于官方 B 训练包 `baselines/train.py` 的修改；模型、数据、预测和各配置沿用当前官方 A/B/C 包。为了遵守原仓库的第三方再分发边界，基线、SDK 与 NumPy wheel 请从赛事平台及其官方来源取得。
- 完整离线构建上下文和复现工具以队伍提交的 `07_harness_source_runtime_v5.zip` 为准。归档 SHA-256 `2b2ade2fc3d4dd7cc840a862b149853687a4c716f8de76d68cdc91cf51f0595d`。
- 容器 TAR SHA-256 `53981b27dbcdfde392b707c7b6e608b0a7b3d86ad0352c475974d309f133b82f`。平台镜像编号 3；本地源码提交 `cf78e47bdde79f29ed7fd72f02217e0251ca8a65`。
- [系统运行说明](./04_系统运行说明_runtime_v5.pdf)、[项目说明](./05_项目说明书_runtime_v5.pdf)、[本地实验记录](./evidence/08_实验记录_runtime_v5.pdf)。

现有实验使用官方公开数据和官方 SDK Host Mock，训练代码真实执行但模型调用为模拟回执，并非真实 PyTRiO 调用或平台认证。正式平台试运行和评测结果待取得。[历史 SwanLab 记录](https://swanlab.cn/@HAAKning/industrial-rsi)不包含本次 Mock 验收。

旧版路径存在时跳过 SDK 父文件校验且将回滚副本记为新候选。新版通过 SDK 按父描述符验证文件，核对实际加载字节和继承状态；未改善时选择已有父 ID。平台未指出具体违规轮次，不能断言这些问题是唯一触发原因。
