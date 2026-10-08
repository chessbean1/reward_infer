
from safetensors import safe_open
from pathlib import Path
from collections import Counter

model_dir = Path("/data/stage3_code/model/student-v4")
model_path = model_dir / "diffusion_pytorch_model.safetensors"

print("文件存在:", model_path.exists())

if model_path.exists():
    print("文件大小 (GB):", model_path.stat().st_size / 1e9)

    try:
        with safe_open(str(model_path), framework="pt", device="cpu") as f:
            keys = list(f.keys())

            print("文件读取成功")
            print("Tensor 数量:", len(keys))
            print("前 10 个参数名称:")
            for key in keys[:10]:
                print("  ", key)

            dtype_counts = Counter()
            total_params = 0
            total_bytes = 0

            dtype_bytes = {
                "F64": 8,
                "F32": 4,
                "BF16": 2,
                "F16": 2,
                "I64": 8,
                "I32": 4,
                "I16": 2,
                "I8": 1,
                "U8": 1,
                "BOOL": 1,
                "F8_E4M3": 1,
                "F8_E5M2": 1,
            }

            for key in keys:
                tensor = f.get_slice(key)
                shape = tensor.get_shape()
                dtype = tensor.get_dtype()

                numel = 1
                for dim in shape:
                    numel *= dim

                total_params += numel
                dtype_counts[dtype] += 1
                total_bytes += numel * dtype_bytes.get(dtype, 0)

            print("\n参数 dtype 统计:", dtype_counts)
            print("总参数量:", total_params)
            print("理论存储大小 (GB):", total_bytes / 1e9)

    except Exception as e:
        print("读取失败:", repr(e))
