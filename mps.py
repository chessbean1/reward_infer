import os
import csv

txt_path = "prompts.txt"
teacher_dir = "teacher"
student_dir = "student"
output_path = "scores.csv"

# 你的 overall condition
condition = (
    "light, color, clarity, tone, style, ambiance, artistry, shape, "
    "face, hair, hands, limbs, structure, instance, texture, quantity, "
    "attributes, position, number, location, word, things."
)


def find_image(folder, keyword):
    """
    在 folder 中寻找文件名包含 keyword 的图片。
    找到第一个就返回完整路径。
    """
    image_exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

    for filename in os.listdir(folder):
        if keyword in filename and filename.lower().endswith(image_exts):
            return os.path.join(folder, filename)

    return None


with open(txt_path, "r", encoding="utf-8") as f, \
     open(output_path, "w", newline="", encoding="utf-8") as out_f:

    writer = csv.writer(out_f)
    writer.writerow([
        "prompt",
        "teacher_image",
        "student_image",
        "teacher_score",
        "student_score"
    ])

    for line in f:
        prompt = line.strip()

        # 跳过空行
        if not prompt:
            continue

        teacher_img = find_image(teacher_dir, prompt)
        student_img = find_image(student_dir, prompt)

        if teacher_img is None or student_img is None:
            print(
                f"[skip] {prompt} | "
                f"teacher={teacher_img} | student={student_img}"
            )
            continue

        probs = infer_example(
            [teacher_img, student_img],
            prompt,
            condition,
            model,
            image_processor,
            tokenizer,
            device
        )

        teacher_score = probs[0]
        student_score = probs[1]

        print(
            f"{prompt} | "
            f"teacher: {teacher_score:.6f} | "
            f"student: {student_score:.6f}"
        )

        writer.writerow([
            prompt,
            teacher_img,
            student_img,
            teacher_score,
            student_score
        ])
