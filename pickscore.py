# import
from transformers import AutoProcessor, AutoModel
from PIL import Image
import torch
import os


# =========================
# load model
# =========================
device = "cuda"

processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
model_pretrained_name_or_path = "yuvalkirstain/PickScore_v1"

processor = AutoProcessor.from_pretrained(processor_name_or_path)

model = AutoModel.from_pretrained(
    model_pretrained_name_or_path
).eval().to(device)


# =========================
# 原来的打分函数，基本不改
# =========================
def calc_probs(prompt, images):

    # preprocess images
    image_inputs = processor(
        images=images,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    # preprocess text
    text_inputs = processor(
        text=prompt,
        padding=True,
        truncation=True,
        max_length=77,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():

        # image embedding
        image_embs = model.get_image_features(**image_inputs)
        image_embs = image_embs / torch.norm(
            image_embs,
            dim=-1,
            keepdim=True
        )

        # text embedding
        text_embs = model.get_text_features(**text_inputs)
        text_embs = text_embs / torch.norm(
            text_embs,
            dim=-1,
            keepdim=True
        )

        # score
        scores = model.logit_scale.exp() * (
            text_embs @ image_embs.T
        )[0]

        # teacher/student 两张图之间做 softmax
        probs = torch.softmax(scores, dim=-1)

    return probs.cpu().tolist()


# =========================
# 新增：寻找包含 keyword 的图片
# =========================
def find_image(folder, keyword):

    image_exts = (
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp"
    )

    for filename in sorted(os.listdir(folder)):

        if (
            keyword in filename
            and filename.lower().endswith(image_exts)
        ):
            return os.path.join(folder, filename)

    return None


# =========================
# main
# =========================
if __name__ == "__main__":

    txt_path = "./prompts.txt"

    teacher_dir = "./teacher"
    student_dir = "./student"

    output_path = "./pickscore_results.txt"

    # 读取所有 prompt
    with open(txt_path, "r", encoding="utf-8") as f:
        prompts = [
            line.strip()
            for line in f
            if line.strip()
        ]

    # 输出文件
    with open(output_path, "w", encoding="utf-8") as fout:

        # 表头
        fout.write(
            "prompt\tteacher_score\tstudent_score\n"
        )

        for prompt in prompts:

            # -------------------------
            # 找 teacher/student 图片
            # -------------------------
            teacher_path = find_image(
                teacher_dir,
                prompt
            )

            student_path = find_image(
                student_dir,
                prompt
            )

            # -------------------------
            # 检查图片是否存在
            # -------------------------
            if teacher_path is None:
                print(
                    f"[Warning] teacher image not found: {prompt}"
                )
                continue

            if student_path is None:
                print(
                    f"[Warning] student image not found: {prompt}"
                )
                continue

            # -------------------------
            # 打开图片
            # -------------------------
            teacher_image = Image.open(
                teacher_path
            ).convert("RGB")

            student_image = Image.open(
                student_path
            ).convert("RGB")

            # 顺序一定是：
            # 0 -> teacher
            # 1 -> student
            pil_images = [
                teacher_image,
                student_image
            ]

            # -------------------------
            # PickScore 打分
            # -------------------------
            probs = calc_probs(
                prompt,
                pil_images
            )

            teacher_score = probs[0]
            student_score = probs[1]

            # -------------------------
            # 控制台输出
            # -------------------------
            print(
                f"{prompt} | "
                f"teacher: {teacher_score:.6f} | "
                f"student: {student_score:.6f}"
            )

            # -------------------------
            # 保存
            # -------------------------
            fout.write(
                f"{prompt}\t"
                f"{teacher_score:.6f}\t"
                f"{student_score:.6f}\n"
            )

            # 防止中途程序出错导致结果没有写入
            fout.flush()

            teacher_image.close()
            student_image.close()

    print(
        f"\nDone! Results saved to: {output_path}"
    )
