# 本地兼容补丁

官方源码固定到 `d91ea855110dae533f0aada947b2b7d78cc8a4e1`。

只修改 `python/katago/train/model_pytorch.py` 中非 1x1 输入卷积的构造：

```diff
- kernel_size=3
+ kernel_size=config.get("initial_conv_kernel_size", 3)
```

历史 TF b6c96 第一层是 5x5；转换后的 checkpoint 配置设置
`initial_conv_kernel_size=5`，保留原参数的全部空间范围。
其他模型继续使用上游默认值 3，包括未来独立部署的 tf3-b11。
`scripts/bootstrap.py` 可幂等重建这个补丁；不会覆盖已经改动的训练 checkpoint。

转换脚本另有 Windows 只读 `pread` 兼容函数，作用仅限旧 checkpoint 提取进程。
旧读取器在独立 Python 3.10 环境运行，不向训练环境安装 TensorFlow。
