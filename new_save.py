from pathlib import Path
from safetensors import safe_open

save_dir = Path(
    "/data/stage3_code/model/student-v4-test"
)

save_dir.mkdir(parents=True, exist_ok=True)

# 假设 Transformer 已完成三个 LoRA 的融合与卸载
# 先检查保存前的参数状态
state = transformer.state_dict()

print("保存前 Tensor 数量:", len(state))

print(
    "保存前理论大小 GB:",
    sum(
        t.numel() * t.element_size()
        for t in state.values()
    ) / 1e9
)

# 采用新目录保存，不覆盖原文件
transformer.save_pretrained(
    str(save_dir),
    safe_serialization=True,
    max_shard_size="4GB",
)

# 检查所有保存的分片
for file in sorted(save_dir.glob("*.safetensors")):
    with safe_open(
        str(file),
        framework="pt",
        device="cpu",
    ) as f:
        print(
            file.name,
            "Tensor 数量:",
            len(list(f.keys()))
        )

print("保存完成，所有权重分片均可解析")
