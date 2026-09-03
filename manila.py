import os
import torch
import numpy as np
import random
import cv2

from torchvision import transforms
from models.maniqa import MANIQA
from config import Config
from utils.inference_process import ToTensor, Normalize
from tqdm import tqdm


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Image(torch.utils.data.Dataset):
    def __init__(self, image_path, transform, num_crops=20):
        super(Image, self).__init__()

        self.img_name = os.path.basename(image_path)

        self.img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.img = np.array(self.img).astype('float32') / 255
        self.img = np.transpose(self.img, (2, 0, 1))

        self.transform = transform

        c, h, w = self.img.shape

        new_h = 224
        new_w = 224

        self.img_patches = []

        for i in range(num_crops):
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            patch = self.img[:, top: top + new_h, left: left + new_w]
            self.img_patches.append(patch)

        self.img_patches = np.array(self.img_patches)

    def get_patch(self, idx):
        patch = self.img_patches[idx]

        sample = {
            'd_img_org': patch,
            'score': 0,
            'd_name': self.img_name
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


# ============================================================
# 新增：寻找文件名中包含 keyword 的图片
# ============================================================
def find_image(folder, keyword):
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    for filename in os.listdir(folder):
        if keyword in filename and filename.lower().endswith(image_exts):
            return os.path.join(folder, filename)

    return None


# ============================================================
# 新增：给单张图片计算 MANIQA 分数
# ============================================================
def score_image(image_path, net, transform, num_crops, device='cuda'):

    Img = Image(
        image_path=image_path,
        transform=transform,
        num_crops=num_crops
    )

    avg_score = 0.0

    with torch.no_grad():
        net.eval()

        for i in range(num_crops):
            patch_sample = Img.get_patch(i)

            patch = patch_sample['d_img_org'].to(device)
            patch = patch.unsqueeze(0)

            score = net(patch)

            # 转成普通 float，避免一直保存 tensor
            avg_score += score.item()

    avg_score /= num_crops

    return avg_score


if __name__ == '__main__':

    cpu_num = 1

    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)

    torch.set_num_threads(cpu_num)

    setup_seed(20)

    # ========================================================
    # 路径
    # ========================================================
    txt_path = './prompts.txt'

    teacher_dir = './teacher'
    student_dir = './student'

    output_path = './maniqa_scores.txt'

    # ========================================================
    # config
    # ========================================================
    config = Config({

        "num_crops": 20,

        "patch_size": 8,
        "img_size": 224,
        "embed_dim": 768,
        "dim_mlp": 768,
        "num_heads": [4, 4],
        "window_size": 4,
        "depths": [2, 2],
        "num_outputs": 1,
        "num_tab": 2,
        "scale": 0.8,

        "ckpt_path": "./ckpt_koniq10k.pt",
    })

    # ========================================================
    # transform，只创建一次
    # ========================================================
    transform = transforms.Compose([
        Normalize(0.5, 0.5),
        ToTensor()
    ])

    # ========================================================
    # 模型只加载一次
    # ========================================================
    net = MANIQA(
        embed_dim=config.embed_dim,
        num_outputs=config.num_outputs,
        dim_mlp=config.dim_mlp,
        patch_size=config.patch_size,
        img_size=config.img_size,
        window_size=config.window_size,
        depths=config.depths,
        num_heads=config.num_heads,
        num_tab=config.num_tab,
        scale=config.scale
    )

    net.load_state_dict(
        torch.load(config.ckpt_path),
        strict=False
    )

    net = net.cuda()
    net.eval()

    # ========================================================
    # 遍历 txt
    # ========================================================
    with open(txt_path, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]

    with open(output_path, 'w', encoding='utf-8') as fout:

        # 输出表头
        fout.write("prompt\tteacher_score\tstudent_score\n")

        for prompt in tqdm(prompts):

            # 根据 prompt 找对应图片
            teacher_img = find_image(teacher_dir, prompt)
            student_img = find_image(student_dir, prompt)

            # 没找到就跳过
            if teacher_img is None:
                print(f"[Warning] teacher image not found: {prompt}")
                continue

            if student_img is None:
                print(f"[Warning] student image not found: {prompt}")
                continue

            # =================================================
            # 分别打分
            # =================================================
            teacher_score = score_image(
                teacher_img,
                net,
                transform,
                config.num_crops
            )

            student_score = score_image(
                student_img,
                net,
                transform,
                config.num_crops
            )

            # =================================================
            # 控制台输出
            # =================================================
            print(
                f"{prompt} | "
                f"teacher: {teacher_score:.6f} | "
                f"student: {student_score:.6f}"
            )

            # =================================================
            # 保存
            # =================================================
            fout.write(
                f"{prompt}\t"
                f"{teacher_score:.6f}\t"
                f"{student_score:.6f}\n"
            )
            fout.flush()

    print(f"\nDone! Results saved to: {output_path}")
