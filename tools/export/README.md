# pnnx 导出边界

源码仓库提供端侧 C++ 运行时、模型布局检查和模型打包工具；转换完成的 ncnn
资产单独发布在
[Coderdw/Youtu-VL-4B-Instruct-ncnn](https://huggingface.co/Coderdw/Youtu-VL-4B-Instruct-ncnn/tree/main)，
不重复写入 Git 历史。

转换对象固定为 `tencent/Youtu-VL-4B-Instruct` revision
`8d30a0e49662a1d628a472b12df264dbcd768753`。转换边界包括：

- Vision embedding、27 个 Vision block 和 post layer norm；
- VLPatchMerger；
- 40 个 LLM layer，每层拆为 Attention Input、Attention Output 和 MLP；
- Final RMSNorm、17 个 LM-head shard、text embedding 和 RoPE 常量；
- C++ ByteLevel-BPE 使用的 `tokenizer.bin`。

pnnx/ncnn 图执行线性层、归一化、激活和投影；动态图像网格、Vision
RoPE/mask、图文 token 融合、KV Cache、Prefill/Decode 调度和文本解码
保留在 C++ 中。

生成运行资产后，先检查布局：

```bash
python3 tools/check_model_layout.py --model-root /path/to/export-workspace
```

再复制运行必需文件并生成 SHA-256：

```bash
python3 tools/package_model.py \
  --source-root /path/to/export-workspace \
  --output-root /path/to/youtu-vl-ncnn-model
```
