import os
import csv
from io import BytesIO

from transformers import CLIPImageProcessor, AutoTokenizer
from PIL import Image
import torch


# ============================================================
# load model
# ============================================================

device = "cuda"

processor_name_or_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"

image_processor = CLIPImageProcessor.from_pretrained(
    processor_name_or_path
)

tokenizer = AutoTokenizer.from_pretrained(
    processor_name_or_path,
    trust_remote_code=True
)

model_ckpt_path = "outputs/MPS_overall_checkpoint.pth"

model = torch.load(model_ckpt_path)
model.eval().to(device)


# ============================================================
# 原来的推理函数
# ============================================================

def infer_example(
    images,
    prompt,
    condition,
    clip_model,
    clip_processor,
    tokenizer,
    device
):

    def _process_image(image):

        if isinstance(image, dict):
            image = image["bytes"]

        if isinstance(image, bytes):
            image = Image.open(BytesIO(image))

        if isinstance(image, str):
            image = Image.open(image)

        image = image.convert("RGB")

        pixel_values = clip_processor(
            image,
            return_tensors="pt"
        )["pixel_values"]

        return pixel_values


    def _tokenize(caption):

        input_ids = tokenizer(
            caption,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids

        return input_ids


    image_inputs = torch.concatenate([
        _process_image(images[0]).to(device),
        _process_image(images[1]).to(device)
    ])

    text_inputs = _tokenize(prompt).to(device)
    condition_inputs = _tokenize(condition).to(device)


    with torch.no_grad():

        text_features, image_0_features, image_1_features = clip_model(
            text_inputs,
            image_inputs,
            condition_inputs
        )

        image_0_features = (
            image_0_features
            / image_0_features.norm(dim=-1, keepdim=True)
        )

        image_1_features = (
            image_1_features
            / image_1_features.norm(dim=-1, keepdim=True)
        )

        text_features = (
            text_features
            / text_features.norm(dim=-1, keepdim=True)
        )


        image_0_scores = (
            clip_model.logit_scale.exp()
            * torch.diag(
                torch.einsum(
                    'bd,cd->bc',
                    text_features,
                    image_0_features
                )
            )
        )

        image_1_scores = (
            clip_model.logit_scale.exp()
            * torch.diag(
                torch.einsum(
                    'bd,cd->bc',
                    text_features,
                    image_1_features
                )
            )
        )


        scores = torch.stack(
            [
                image_0_scores,
                image_1_scores
            ],
            dim=-1
        )

        probs = torch.softmax(
            scores,
            dim=-1
        )[0]


    return probs.cpu().tolist()


# ============================================================
# 新增：在文件夹中找名称包含 keyword 的图片
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
            return os.path.join(
                folder,
                filename
            )

    return None


# ============================================================
# main
# ============================================================

if __name__ == "__main__":

    # txt 文件
    txt_path = "./prompts.txt"

    # teacher / student 文件夹
    teacher_dir = "./teacher"
    student_dir = "./student"

    # 输出 CSV
    output_csv = "./mps_scores.csv"


    # overall condition
    condition = (
        "light, color, clarity, tone, style, ambiance, artistry, "
        "shape, face, hair, hands, limbs, structure, instance, "
        "texture, quantity, attributes, position, number, "
        "location, word, things."
    )


    # ========================================================
    # 读取 txt
    # ========================================================

    with open(
        txt_path,
        "r",
        encoding="utf-8"
    ) as f:

        prompts = [
            line.strip()
            for line in f
            if line.strip()
        ]


    # ========================================================
    # 遍历并保存结果
    # ========================================================

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

            # ------------------------------------------------
            # 找对应 teacher / student 图片
            # ------------------------------------------------

            teacher_path = find_image(
                teacher_dir,
                prompt
            )

            student_path = find_image(
                student_dir,
                prompt
            )


            # ------------------------------------------------
            # 找不到则跳过
            # ------------------------------------------------

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


            # ------------------------------------------------
            # 打分
            #
            # images[0] -> teacher
            # images[1] -> student
            # ------------------------------------------------

            probs = infer_example(
                [
                    teacher_path,
                    student_path
                ],
                prompt,
                condition,
                model,
                image_processor,
                tokenizer,
                device
            )


            teacher_score = probs[0]
            student_score = probs[1]


            # ------------------------------------------------
            # 控制台输出
            # ------------------------------------------------

            print(
                f"{prompt} | "
                f"teacher: {teacher_score:.6f} | "
                f"student: {student_score:.6f}"
            )


            # ------------------------------------------------
            # 写 CSV
            # ------------------------------------------------

            writer.writerow([
                prompt,
                teacher_score,
                student_score
            ])

            # 防止程序中途停止导致结果没有落盘
            fout.flush()


    print(
        f"\nDone! Results saved to: {output_csv}"
    )
