"""Run Metric-DROID and evaluate every scene in a dataset."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

from eval.trajectory_metrics import (
    evaluate_trajectory_files,
    format_trajectory_metrics,
    save_kitti_trajectory,
    save_trajectory_route_plot,
)
from modules import droid as Droid


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
METRIC_NAMES = ("ave_translation", "ave_rotation", "ate", "ave_scale_error")


def run_scene(image_dir, depth_dir, intrinsic, checkpoint, buffer, progress_desc):
    options = Droid.Options()
    options.intrinsic = intrinsic
    options.weights = checkpoint
    options.buffer = buffer
    options.disable_vis = True
    options.global_ba_frontend = 0
    options.final_backend = False
    options.progress_desc = progress_desc
    return Droid.run(image_dir, options, depth_dir=depth_dir)


def save_droid_trajectory(trajectory, path):
    trajectory = PoseTrajectory3D(
        positions_xyz=trajectory[:, 1:4],
        orientations_quat_wxyz=trajectory[:, [7, 4, 5, 6]],
        timestamps=trajectory[:, 0],
    )
    file_interface.write_kitti_poses_file(path, trajectory)


def average_metrics(scene_metrics, alignment):
    averages = {}
    for metric in METRIC_NAMES:
        values = [
            result[alignment][metric]
            for result in scene_metrics.values()
            if result[alignment][metric] is not None
        ]
        averages[metric] = float(np.mean(values)) if values else None
    return averages


def main():
    parser = argparse.ArgumentParser(
        description="Run Metric-DROID on all scenes and evaluate the trajectories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--depth-root", type=Path, required=True)
    parser.add_argument("--intrinsics-json", type=Path, required=True)
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--image-subdir", default="image_2")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "results" / "gt_K")
    parser.add_argument("--checkpoint", type=Path, default=PROJECT_ROOT / "weights" / "droid.pth")
    parser.add_argument("--buffer", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--plot-dpi", type=int, default=150)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    depth_root = args.depth_root.resolve()
    gt_dir = args.gt_dir.resolve()
    output_dir = (args.output_root / args.dataset_name).resolve()

    with args.intrinsics_json.open("r", encoding="utf-8") as file:
        intrinsics = json.load(file)

    scenes = sorted(
        scene
        for scene in intrinsics
        if (dataset_root / scene / args.image_subdir).is_dir()
        and (depth_root / scene).is_dir()
        and (gt_dir / f"{scene}.txt").is_file()
    )
    if not scenes:
        raise ValueError("No scenes have images, depth, intrinsics, and ground truth")

    before_dir = output_dir / "before_alignment"
    after_dir = output_dir / "after_alignment"
    before_plot_dir = output_dir / "visualizations" / "before_alignment"
    after_plot_dir = output_dir / "visualizations" / "after_alignment"
    for directory in (before_dir, after_dir, before_plot_dir, after_plot_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset_name}")
    print(f"Scenes: {len(scenes)}")
    print(f"Output: {output_dir}")
    print(f"Keyframe buffer: {args.buffer}")
    print("Frontend global BA: disabled")
    print("Final backend optimization: disabled")

    scene_metrics = {}
    for scene in scenes:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        image_dir = dataset_root / scene / args.image_subdir
        depth_dir = depth_root / scene
        gt_path = gt_dir / f"{scene}.txt"
        raw_path = before_dir / f"{scene}.txt"
        aligned_path = after_dir / f"{scene}.txt"
        image_count = sum(
            path.suffix.lower() in IMAGE_SUFFIXES
            for path in image_dir.iterdir()
        )

        print(f"\n[{scene}] {image_count} images")
        if raw_path.exists() and not args.overwrite:
            print(f"[{scene}] Reusing {raw_path}")
        else:
            trajectory = run_scene(
                image_dir=image_dir,
                depth_dir=depth_dir,
                intrinsic=np.asarray(intrinsics[scene], dtype=np.float32),
                checkpoint=args.checkpoint,
                buffer=args.buffer,
                progress_desc=f"{args.dataset_name} {scene}",
            )
            save_droid_trajectory(trajectory, raw_path)
            print(f"[{scene}] Saved {raw_path}")

        before_metrics, _, _ = evaluate_trajectory_files(
            gt_path, raw_path, alignment=None
        )
        after_metrics, aligned_poses, sim3 = evaluate_trajectory_files(
            gt_path, raw_path, alignment="7dof"
        )

        save_trajectory_route_plot(
            before_dir,
            gt_dir,
            sequence=scene,
            method_name="Metric-DROID (before alignment)",
            color="tab:orange",
            save_path=before_plot_dir / f"{scene}.png",
            dpi=args.plot_dpi,
        )

        if aligned_poses is None:
            sim3_result = None
        else:
            save_kitti_trajectory(aligned_poses, str(aligned_path))
            save_trajectory_route_plot(
                after_dir,
                gt_dir,
                sequence=scene,
                method_name="Metric-DROID (Sim(3) aligned)",
                color="tab:blue",
                save_path=after_plot_dir / f"{scene}.png",
                dpi=args.plot_dpi,
            )
            sim3_result = {
                "scale": float(sim3["scale"]),
                "rotation": sim3["rotation"].tolist(),
                "translation": sim3["translation"].tolist(),
            }

        scene_metrics[scene] = {
            "num_poses": image_count,
            "before_alignment": before_metrics,
            "after_alignment": after_metrics,
            "sim3": sim3_result,
        }
        print(format_trajectory_metrics(f"{scene} before alignment", before_metrics))
        print(format_trajectory_metrics(f"{scene} after Sim(3)", after_metrics))
        torch.cuda.empty_cache()

    averages = {
        "before_alignment": average_metrics(scene_metrics, "before_alignment"),
        "after_alignment": average_metrics(scene_metrics, "after_alignment"),
    }
    print("\nAggregate metrics")
    print(format_trajectory_metrics("Average before alignment", averages["before_alignment"]))
    print(format_trajectory_metrics("Average after Sim(3)", averages["after_alignment"]))

    report = {
        "dataset": args.dataset_name,
        "alignment": "Sim(3) / 7DoF",
        "settings": {
            "buffer": args.buffer,
            "global_ba_frontend": 0,
            "final_backend": False,
        },
        "scenes": scene_metrics,
        "average": averages,
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
