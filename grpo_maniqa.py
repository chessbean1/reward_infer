class MANIQAImage:
    """
    仿照 MANIQA 官方 predict_one_image.py 的 Image 类，
    但只保留 reward inference 真正需要的部分。
    """

    def __init__(self, image_path, num_crops=20, seed=20):

        self.img_name = os.path.basename(image_path)

        # 与官方代码一致：
        # cv2 -> RGB -> float32 -> /255 -> CHW
        self.img = cv2.imread(
            image_path,
            cv2.IMREAD_COLOR
        )

        if self.img is None:
            raise ValueError(
                f"Failed to read image: {image_path}"
            )

        self.img = cv2.cvtColor(
            self.img,
            cv2.COLOR_BGR2RGB
        )

        self.img = (
            np.array(self.img)
            .astype("float32")
            / 255.0
        )

        self.img = np.transpose(
            self.img,
            (2, 0, 1)
        )

        c, h, w = self.img.shape

        new_h = 224
        new_w = 224

        if h < new_h or w < new_w:
            raise ValueError(
                f"MANIQA requires image >= 224x224, "
                f"but got {w}x{h}: {image_path}"
            )

        # ----------------------------------------------------
        # 不使用全局 np.random.seed()
        #
        # 否则可能影响 DanceGRPO 自己的训练随机数。
        #
        # 使用独立 RNG：
        # 同一张图重复计算 reward 时 crop 一致。
        # ----------------------------------------------------
        rng = np.random.RandomState(seed)

        self.img_patches = []

        for _ in range(num_crops):

            # 官方是随机 crop
            top = rng.randint(
                0,
                h - new_h + 1
            )

            left = rng.randint(
                0,
                w - new_w + 1
            )

            patch = self.img[
                :,
                top:top + new_h,
                left:left + new_w
            ]

            self.img_patches.append(patch)

        self.img_patches = np.array(
            self.img_patches
        )


    def get_patch(self, idx):

        patch = self.img_patches[idx]

        # ----------------------------------------------------
        # 等价于 MANIQA 官方：
        #
        # Normalize(0.5, 0.5)
        # ToTensor()
        #
        # Normalize 官方实现就是：
        # (image - mean) / var
        # ----------------------------------------------------

        patch = (
            patch - 0.5
        ) / 0.5

        patch = torch.from_numpy(
            patch
        ).float()

        return patch


def calc_maniqa_score(
    image_path,
    reward_model,
    device,
    num_crops=20,
    crop_seed=20
):
    """
    仿照 MANIQA predict_one_image.py：
    对一张图随机裁 num_crops 个 224x224 patch，
    分别打分后取平均。

    返回 shape=[1] 的 Tensor，
    可以直接加入 DanceGRPO 的 all_rewards。
    """

    img = MANIQAImage(
        image_path=image_path,
        num_crops=num_crops,
        seed=crop_seed
    )

    avg_score = torch.zeros(
        1,
        device=device,
        dtype=torch.float32
    )

    reward_model.eval()

    with torch.no_grad():

        for i in range(num_crops):

            patch = img.get_patch(i)

            patch = patch.to(
                device=device,
                non_blocking=True
            )

            # [C, H, W]
            # ->
            # [1, C, H, W]
            patch = patch.unsqueeze(0)

            score = reward_model(
                patch
            )

            avg_score += score.float()

    avg_score = (
        avg_score
        / num_crops
    )

    return avg_score
