# 来源与组件

本实现从 HAAK runtime-v4 选手代码整改，保留官方工业模型实现和继承语义。
`baselines/model.py`、`data.py`、`predict.py` 和各任务配置沿用官方包。
`baselines/train.py` 在共用 B 回归兼容版上增加可选的实读摘要与恢复状态校验。
官方基线来源为 RSI 训练起点 A/B/C 当前公开包，完整逐文件摘要保存在实验记录。

NumPy 1.26.4 wheel 内含组件许可和第三方声明，核心采用 BSD 3-Clause。
Python 3.12 基础镜像沿用 Python Software Foundation 许可及镜像组件声明。
contest-harness SDK 1.0.5 来自主办方，随源码保留官方压缩包、原始文档和文件。
原始 SDK 包未单列 LICENSE 文件，不据此替主办方额外授予许可。
训练数据和起点请从官方平台另行取得；不包含隐藏测试集、密钥或兑换码。
