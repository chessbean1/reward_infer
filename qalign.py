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
# 选择模型
# ============================================================

# 图像质量评分
scorer = QAlignScorer()

# 如果你想测试美学质量，则把上面一行注释掉，使用下面这一行
# scorer = QAlignAestheticScorer()


# ============================================================
# 根据关键词寻找图片
# ============================================================

def find_image(folder, keyword):
    """
    在 folder 中寻找文件名包含 keyword 的图片
    """

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
# 遍历并打分
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
        "teacher_score",
        "student_score"
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
        # 检查是否找到
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # 打开图片
        # ----------------------------------------------------

        teacher_img = Image.open(
            teacher_path
        ).convert("RGB")

        student_img = Image.open(
            student_path
        ).convert("RGB")


        # ----------------------------------------------------
        # 两张图片一次性打分
        #
        # scores[0] = teacher
        # scores[1] = student
        # ----------------------------------------------------

        img_list = [
            teacher_img,
            student_img
        ]

        scores = scorer(img_list).tolist()

        teacher_score = scores[0]
        student_score = scores[1]


        # ----------------------------------------------------
        # 控制台输出
        # ----------------------------------------------------

        print(
            f"{prompt} | "
            f"teacher: {teacher_score:.6f} | "
            f"student: {student_score:.6f}"
        )


        # ----------------------------------------------------
        # 写入 CSV
        # ----------------------------------------------------

        writer.writerow([
            prompt,
            teacher_score,
            student_score
        ])

        # 防止运行到一半出错导致之前结果未写入
        fout.flush()


        # 关闭图片
        teacher_img.close()
        student_img.close()


print(f"\nDone! Results saved to: {output_csv}")
