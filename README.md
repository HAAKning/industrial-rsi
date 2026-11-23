# 智训引擎工业模型自进化

本仓库用于参赛方案的源码审阅。核心是通过官方Host网关请求续训策略，对策略做白名单校验，调用官方工业小模型引擎续训，并依据当前公开验证指标采纳候选或回滚。

## 源码与复现

- `policy.py` 与 `contest_submission.py` 是供直接浏览的核心原创模块，与已核验的v2镜像对应文件逐字节一致。
- `original_source_v2.zip` 保留项目目录结构，包含参赛队代码、Dockerfile、测试、复现脚本及公开数据实验结果。下载并解压后使用其中的目录结构运行，不能只把网页上两个模块当作完整运行环境。
- 官方 `contest-harness-sdk 1.0.4`、官方 `baselines/` 和数据包不在此公开仓库中转载。复现者需从赛事平台取得相同版本，将基线放入 `submission/app/baselines/`，将SDK压缩包放入 `submission/contest-harness-sdk-1.0.4.tar.gz`。
- 离线构建还需 NumPy 1.26.4 的 CPython 3.12 musllinux x86_64 wheel，放入 `submission/vendor/`。基础镜像须事先缓存。具体操作见压缩包内README和Dockerfile。
- 赛事指定源码附件另附完整构建上下文；公开仓库与比赛附件的第三方文件范围不同。

## 验证状态

v2镜像为Linux/amd64，A/B/C各通过一轮断网、根文件系统只读、2核与4GiB内存限制下的Host替身协议检查，并对实际上传的检查点各进行3条预测检查。

该检查使用预设模型回复，不代表真实PyTRiO调用成功。48组网格结果是公开验证集调参结果，不是平台正式成绩。真实模型回执、Token usage与隐藏测试成绩仍待平台验证。回滚不能保证隐藏集泛化。

镜像TAR SHA256：`f7964d69aa59c9c655e7821249c17eb30ebe2827a18cd512b3f088e1fa27795a`。

历史公开实验：https://swanlab.cn/@HAAKning/industrial-rsi 。51条历史记录为48组公开网格和3组Mock闭环；v2新增本地检查尚未追加到该历史项目。

## 运行约定

基础模型标识为 `Qwen/Qwen3.8-27B`。容器环境变量必须为模块名 `CONTEST_SUBMISSION=contest_submission`。回归函数返回 `{"value": 非负有限数值}`，SDK包装外层outputs。stdout只输出协议JSONL，诊断写stderr。

密钥仅在赛事平台填写，不写入本仓库或比赛源码附件。

## 第三方组件

NumPy采用BSD-3-Clause许可。官方SDK和基线由赛事平台提供，其再分发许可以主办方规定为准。本仓库不对参赛队代码授予额外开源许可；代码公开供赛事审阅。
