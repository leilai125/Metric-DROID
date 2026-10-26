"""Run Metric3D depth estimation over a directory of scenes."""

# standard library
from pathlib import Path
import argparse
import json
import os

# third party
import cv2
import numpy as np
from tqdm import tqdm

# metric 3d
from modules import Metric

DEPTH_SCALE = 100.0  # store depth in centimetres


def main(
    input_sequences,
    output_root,
    dataset_name,
    intrinsic_json,
    d_max=300.0,
    overwrite=True,
    save_colormap=False,
    checkpoint="./weights/metric_depth_vit_giant2_800k.pth",
    model_name="v2-g",
):
    # load per-scene intrinsics
    with open(intrinsic_json, "r") as file:
        intrinsics = json.load(file)

    # init metric 3d once for all scenes
    metric = Metric(checkpoint=checkpoint, model_name=model_name)

    # load scenes
    sequences_dir = Path(input_sequences).resolve()
    scenes = sorted(path for path in sequences_dir.iterdir() if path.is_dir())

    # create dataset output directory
    dataset_dir = Path(output_root).resolve() / dataset_name
    os.makedirs(dataset_dir, exist_ok=True)

    for scene in tqdm(scenes, desc="scenes"):
        # load this scene's intrinsic
        intr = np.asarray(intrinsics[scene.name], dtype=np.float32)[:4]

        # load JPG, JPEG, or PNG images
        image_dir = scene / "image_2"
        images = sorted(
            image
            for image in image_dir.iterdir()
            if image.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        # create scene output directory
        out_dir = dataset_dir / scene.name
        os.makedirs(out_dir, exist_ok=True)
        color_dir = out_dir / "colormap"
        if save_colormap:
            os.makedirs(color_dir, exist_ok=True)

        for image in tqdm(images, desc=f"scene {scene.name}", leave=False):
            depth_path = out_dir / f"{image.stem}.png"
            if overwrite or not depth_path.exists():
                depth = metric(rgb_image=image, intrinsic=intr, d_max=d_max)

                # save metric depth as uint16 centimetres
                depth_png = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                depth_png = np.clip(
                    np.rint(depth_png * DEPTH_SCALE), 0, np.iinfo(np.uint16).max
                ).astype(np.uint16)
                cv2.imwrite(str(depth_path), depth_png)

                # save colormap
                if save_colormap:
                    depth_color = metric.gray_to_colormap(depth)
                    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(color_dir / f"{image.stem}.png"), depth_color)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running Metric3D on multiple scenes")
    parser.add_argument("--sequences", type=str, required=True, help="directory containing scenes")
    parser.add_argument("--dataset", type=str, required=True, help="dataset output name")
    parser.add_argument("--intr", type=str, required=True, help="per-scene intrinsic JSON")
    parser.add_argument("--out", type=str, default="/data/lei/metric3dv2/gt_K", help="output root")
    parser.add_argument("--out-colormap", action="store_true", default=False)
    parser.add_argument("--dmax", type=float, default=300.0)
    parser.add_argument("--skip-existed", action="store_true", default=False)
    parser.add_argument("--checkpoint", type=str, default="./weights/metric_depth_vit_giant2_800k.pth")
    parser.add_argument("--model-name", type=str, default="v2-g", choices=["v2-L", "v2-S", "v2-g"])
    args = parser.parse_args()

    main(
        input_sequences=args.sequences,
        output_root=args.out,
        dataset_name=args.dataset,
        intrinsic_json=args.intr,
        d_max=args.dmax,
        overwrite=not args.skip_existed,
        save_colormap=args.out_colormap,
        checkpoint=args.checkpoint,
        model_name=args.model_name,
    )
