import os
import csv
from q_align import QAlignScorer, QAlignAestheticScorer
from PIL import Image


# ============================================================
# 配置
# ============================================================

txt_path = "./prompts.txt"

teacher_dir = "./teacher"
student_dir = "./student"

output_csv = "./qalign_scores.csv"


# ============================================================
# 同时加载两个模型
# ============================================================

# 图像质量
quality_scorer = QAlignScorer()

# 图像美学
aesthetic_scorer = QAlignAestheticScorer()


# ============================================================
# 根据关键词寻找图片
# ============================================================

def find_image(folder, keyword):

    image_exts = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    )

    for filename in sorted(os.listdir(folder)):

        if (
            keyword in filename
            and filename.lower().endswith(image_exts)
        ):
            return os.path.join(folder, filename)

    return None


# ============================================================
# 读取 txt
# ============================================================

with open(txt_path, "r", encoding="utf-8") as f:

    prompts = [
        line.strip()
        for line in f
        if line.strip()
    ]


# ============================================================
# 遍历、打分并保存
# ============================================================

with open(
    output_csv,
    "w",
    newline="",
    encoding="utf-8-sig"
) as fout:

    writer = csv.writer(fout)

    # CSV 表头
    writer.writerow([
        "prompt",
        "teacher_quality",
        "student_quality",
        "teacher_aesthetic",
        "student_aesthetic"
    ])

    for prompt in prompts:

        # ----------------------------------------------------
        # 找 teacher 和 student 图片
        # ----------------------------------------------------

        teacher_path = find_image(
            teacher_dir,
            prompt
        )

        student_path = find_image(
            student_dir,
            prompt
        )


        # ----------------------------------------------------
        # 检查图片
        # ----------------------------------------------------

        if teacher_path is None:
            print(f"[Warning] teacher image not found: {prompt}")
            continue

        if student_path is None:
            print(f"[Warning] student image not found: {prompt}")
            continue


        # ----------------------------------------------------
        # 打开图片
        # ----------------------------------------------------

        teacher_img = Image.open(
            teacher_path
        ).convert("RGB")

        student_img = Image.open(
            student_path
        ).convert("RGB")


        # 顺序：
        # 0 -> teacher
        # 1 -> student
        img_list = [
            teacher_img,
            student_img
        ]


        # ----------------------------------------------------
        # 图像质量打分
        # ----------------------------------------------------

        quality_scores = quality_scorer(
            img_list
        ).tolist()

        teacher_quality = quality_scores[0]
        student_quality = quality_scores[1]


        # ----------------------------------------------------
        # 图像美学打分
        # ----------------------------------------------------

        aesthetic_scores = aesthetic_scorer(
            img_list
        ).tolist()

        teacher_aesthetic = aesthetic_scores[0]
        student_aesthetic = aesthetic_scores[1]


        # ----------------------------------------------------
        # 控制台输出
        # ----------------------------------------------------

        print(
            f"{prompt} | "
            f"Quality - "
            f"teacher: {teacher_quality:.6f}, "
            f"student: {student_quality:.6f} | "
            f"Aesthetic - "
            f"teacher: {teacher_aesthetic:.6f}, "
            f"student: {student_aesthetic:.6f}"
        )


        # ----------------------------------------------------
        # 写入 CSV
        # ----------------------------------------------------

        writer.writerow([
            prompt,
            teacher_quality,
            student_quality,
            teacher_aesthetic,
            student_aesthetic
        ])

        fout.flush()


        # ----------------------------------------------------
        # 关闭图片
        # ----------------------------------------------------

        teacher_img.close()
        student_img.close()


print(f"\nDone! Results saved to: {output_csv}")
