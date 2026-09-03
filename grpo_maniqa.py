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



#####sample from reference
        if args.use_maniqa:

            image_path = (
                f"./images/flux_{rank}_{index}.png"
            )

            maniqa_score = calc_maniqa_score(
                image_path=image_path,
                reward_model=reward_model,
                device=device,
                num_crops=args.maniqa_num_crops,
                crop_seed=args.maniqa_crop_seed,
            )

            all_rewards.append(
                maniqa_score
            )


###### initialize reward model
    preprocess_val = None
    processor = None
    reward_model = None


    # ========================================================
    # MANIQA
    # ========================================================

    if args.use_maniqa:

        # ----------------------------------------------
        # 把本地 MANIQA repo 加入 Python path
        #
        # 例如：
        # ./MANIQA/
        #     models/
        #         maniqa.py
        #         swin.py
        # ----------------------------------------------

        maniqa_root = os.path.abspath(
            args.maniqa_root
        )

        if not os.path.isdir(maniqa_root):
            raise FileNotFoundError(
                f"MANIQA root not found: "
                f"{maniqa_root}"
            )

        if maniqa_root not in sys.path:
            sys.path.insert(
                0,
                maniqa_root
            )

        # 与官方 predict_one_image.py 一样
        from models.maniqa import MANIQA


        # ----------------------------------------------
        # 创建模型
        #
        # 参数完全沿用官方
        # predict_one_image.py
        # ----------------------------------------------

        reward_model = MANIQA(
            embed_dim=768,
            num_outputs=1,
            dim_mlp=768,
            patch_size=8,
            img_size=224,
            window_size=4,
            depths=[2, 2],
            num_heads=[4, 4],
            num_tab=2,
            scale=0.8,
        )


        # ----------------------------------------------
        # 本地 checkpoint
        # ----------------------------------------------

        if not os.path.isfile(
            args.maniqa_ckpt_path
        ):
            raise FileNotFoundError(
                f"MANIQA checkpoint not found: "
                f"{args.maniqa_ckpt_path}"
            )


        # 官方：
        #
        # net.load_state_dict(
        #     torch.load(config.ckpt_path),
        #     strict=False
        # )
        #
        # 这里保持相同思路。
        # ----------------------------------------------

        checkpoint = torch.load(
            args.maniqa_ckpt_path,
            map_location="cpu"
        )

        reward_model.load_state_dict(
            checkpoint,
            strict=False
        )


        # ----------------------------------------------
        # MANIQA 只做 reward inference
        # ----------------------------------------------

        reward_model.requires_grad_(
            False
        )

        reward_model = reward_model.to(
            device
        )

        reward_model.eval()


        if rank == 0:
            print(
                "MANIQA reward model loaded: "
                f"{args.maniqa_ckpt_path}"
            )


    # ========================================================
    # 原来的 HPSv2
    # ========================================================

    if args.use_hpsv2:
