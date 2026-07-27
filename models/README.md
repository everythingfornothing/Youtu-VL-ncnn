# Model Layout

模型权重不存入源码 Git 仓库，统一发布在
[Coderdw/Youtu-VL-4B-Instruct-ncnn](https://huggingface.co/Coderdw/Youtu-VL-4B-Instruct-ncnn/tree/main)。
下载、校验和解压命令见项目根目录的 [`README.md`](../README.md)。

解压后的 `model/` 是 `--model-root` 应指向的目录：

```text
model/
|- artifacts/
|  |- vision_embedding/
|  |- vision_layer0_masked_core/ ... vision_layer26_masked_core/
|  |- vision_post_layernorm/
|  |- llm_layer0_three_part/ ... llm_layer39_three_part/
|  |- llm_final_head_ncnn/
|  |- text_embedding/embed_tokens_weight.npy
|  `- llm_rope_inv_freq.npy
|- models/
|  |- youtu_merger.ncnn.param
|  `- youtu_merger.ncnn.bin
|- tokenizer/
|  `- tokenizer.bin
`- checksums.sha256
```

每个 LLM layer 包含三个子图：

```text
part_a_attention_input/part_a_attention_input.ncnn.{param,bin}
part_b_attention_output/part_b_attention_output.ncnn.{param,bin}
part_c_mlp/part_c_mlp.ncnn.{param,bin}
```

Final head 包含 `final_norm` 和 `lm_head_shard_00` 至
`lm_head_shard_16`。`tokenizer.bin` 是 C++ ByteLevel-BPE 运行时资产，
不引入 Python 运行时依赖。

推理前执行：

```bash
python3 tools/check_model_layout.py --model-root /path/to/model
(cd /path/to/model && sha256sum -c checksums.sha256)
```
