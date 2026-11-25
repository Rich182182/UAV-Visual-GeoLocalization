import argparse
import json
import os

import cv2
import gdown
from lightglue import LightGlue, SuperPoint
from lightglue.utils import rbd
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from tqdm import tqdm

from model import UAVEmbedderMultiTask
from train import download_dataset_if_needed, load_coords_df

WEIGHTS_GDRIVE_ID = "1vAtU3O-q0S9IlVUN9LvGkXCjX92nkhsc"
Image.MAX_IMAGE_PIXELS = None


def download_weights_if_needed(weights_path: str) -> str:
    if os.path.exists(weights_path):
        return weights_path
    os.makedirs(os.path.dirname(weights_path), exist_ok=True)
    print("Downloading pretrained weights from Google Drive...")
    gdown.download(id=WEIGHTS_GDRIVE_ID, output=weights_path, quiet=False, fuzzy=True)
    return weights_path


def generate_tiles_with_metadata(
    image_path: str, output_dir: str, tile_size: int = 2048, stride: int = 512
) -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    img_name = os.path.splitext(os.path.basename(image_path))[0]

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        width, height = img.size
        x_coords = sorted(set(list(range(0, width - tile_size, stride)) + [width - tile_size]))
        y_coords = sorted(set(list(range(0, height - tile_size, stride)) + [height - tile_size]))

        metadata = []
        for y in tqdm(y_coords, desc="Tiling satellite map"):
            for x in x_coords:
                tile_filename = f"{img_name}_{x}_{y}.jpg"
                tile_path = os.path.join(output_dir, tile_filename)
                if not os.path.exists(tile_path):
                    img.crop((x, y, x + tile_size, y + tile_size)).save(
                        tile_path, format="JPEG", quality=90
                    )
                metadata.append(
                    {
                        "tile_name": tile_filename,
                        "source_map": img_name,
                        "pixel_x": x,
                        "pixel_y": y,
                        "tile_size": tile_size,
                    }
                )
    return pd.DataFrame(metadata)


def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_m = 6371000.0
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return float(earth_radius_m * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a)))


def load_image_for_lightglue(img_path: str, device: torch.device) -> torch.Tensor:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (640, 640))
    tensor = torch.tensor(img, dtype=torch.float32, device=device) / 255.0
    return tensor.unsqueeze(0).unsqueeze(0)


def pixel_to_gps(pixel_x, pixel_y, tile_name, df_tiles, map_bounds, w_map, h_map):
    tile_info = df_tiles[df_tiles["tile_name"] == tile_name].iloc[0]
    global_x = tile_info["pixel_x"] + pixel_x
    global_y = tile_info["pixel_y"] + pixel_y

    lt_lat, lt_lon = map_bounds["LT_lat_map"], map_bounds["LT_lon_map"]
    rb_lat, rb_lon = map_bounds["RB_lat_map"], map_bounds["RB_lon_map"]

    lon = lt_lon + (global_x / w_map) * (rb_lon - lt_lon)
    lat = lt_lat - (global_y / h_map) * (lt_lat - rb_lat)
    return lat, lon


def find_gt_tile(gt_lat, gt_lon, map_bounds, w_map, h_map, df_tiles):
    pixel_x = (gt_lon - map_bounds["LT_lon_map"]) / (map_bounds["RB_lon_map"] - map_bounds["LT_lon_map"]) * w_map
    pixel_y = (map_bounds["LT_lat_map"] - gt_lat) / (map_bounds["LT_lat_map"] - map_bounds["RB_lat_map"]) * h_map

    for _, row in df_tiles.iterrows():
        tx, ty, ts = row["pixel_x"], row["pixel_y"], row["tile_size"]
        if tx <= pixel_x <= tx + ts and ty <= pixel_y <= ty + ts:
            return row["tile_name"]
    return None


def save_match_visualization(item: dict, title: str, save_path: str):
    img0 = cv2.cvtColor(cv2.imread(item["drone_path"]), cv2.COLOR_BGR2RGB)
    img0 = cv2.resize(img0, (640, 640))
    rot_mat = cv2.getRotationMatrix2D((320.0, 320.0), item["best_angle"], 1.0)
    img0 = cv2.warpAffine(img0, rot_mat, (640, 640))

    img1 = cv2.cvtColor(cv2.imread(item["pred_tile_path"]), cv2.COLOR_BGR2RGB)
    img1 = cv2.resize(img1, (640, 640))

    if item["gt_tile_path"] and os.path.exists(item["gt_tile_path"]):
        img2 = cv2.cvtColor(cv2.imread(item["gt_tile_path"]), cv2.COLOR_BGR2RGB)
        img2 = cv2.resize(img2, (640, 640))
    else:
        img2 = np.zeros((640, 640, 3), dtype=np.uint8)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.96)

    ax1.imshow(img0)
    ax1.set_title(f"UAV Query (Rotated {item['best_angle']}°)", fontsize=12)
    ax1.axis("off")

    if item["is_top1"]:
        status, color = "HIT: Top-1", "green"
    elif item["is_top5"]:
        status, color = "HIT: Top-5", "orange"
    else:
        status, color = "MISS: Outside Top-5", "red"

    ax2.imshow(img1)
    ax2.set_title(f"Retrieved Tile ({status})", fontsize=12, color=color)
    ax2.axis("off")

    ax3.imshow(img2)
    ax3.set_title("Ground Truth Tile", fontsize=12)
    ax3.axis("off")

    if item["m_kps0"] is not None and item["m_kps1"] is not None:
        kps0, kps1 = item["m_kps0"], item["m_kps1"]
        indices = np.arange(len(kps0))
        if len(kps0) > 40:
            rng = np.random.default_rng(42)
            indices = rng.choice(indices, 40, replace=False)

        for idx in indices:
            pt1, pt2 = kps0[idx], kps1[idx]
            ax1.scatter(pt1[0], pt1[1], c="red", s=18, edgecolors="white", linewidths=0.5)
            ax2.scatter(pt2[0], pt2[1], c="red", s=18, edgecolors="white", linewidths=0.5)
            con = ConnectionPatch(
                xyA=pt2, xyB=pt1, coordsA="data", coordsB="data",
                axesA=ax2, axesB=ax1, color="lime", lw=1.2, alpha=0.45,
            )
            ax2.add_artist(con)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    base_dir = download_dataset_if_needed(args.data_dir)
    weights_path = download_weights_if_needed(args.weights_path)

    scene_dir = os.path.join(base_dir, args.test_scene)
    drone_dir = os.path.join(scene_dir, "drone")
    sat_name = f"satellite{args.test_scene}.tif"
    sat_path = os.path.join(scene_dir, sat_name)
    tiles_dir = os.path.join(scene_dir, "satellite_tiles")
    meta_csv_path = os.path.join(scene_dir, "tiles_metadata.csv")

    df_coords = load_coords_df(base_dir)
    df_uav = pd.read_csv(os.path.join(scene_dir, f"{args.test_scene}.csv"))
    df_uav.columns = df_uav.columns.str.strip()
    map_bounds = df_coords[df_coords["mapname"] == sat_name].iloc[0]

    with Image.open(sat_path) as sat_img:
        w_map, h_map = sat_img.size

    if os.path.exists(meta_csv_path):
        df_tiles = pd.read_csv(meta_csv_path)
    else:
        df_tiles = generate_tiles_with_metadata(sat_path, tiles_dir, tile_size=2048, stride=512)
        df_tiles.to_csv(meta_csv_path, index=False)

    model = UAVEmbedderMultiTask(embedding_dim=256).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    extractor = SuperPoint(max_num_keypoints=4096).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    eval_transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    tile_names = df_tiles["tile_name"].tolist()
    db_embs_path = os.path.join(scene_dir, "tile_embeddings.pt")

    if os.path.exists(db_embs_path):
        saved = torch.load(db_embs_path, map_location=device)
        tile_names = saved["names"]
        db_embs = saved["embeddings"].to(device)
    else:
        embs = []
        with torch.no_grad():
            for name in tqdm(tile_names, desc="Building satellite embedding index"):
                img = Image.open(os.path.join(tiles_dir, name)).convert("RGB")
                tensor = eval_transform(img).unsqueeze(0).to(device)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    emb, _ = model(tensor)
                embs.append(emb.cpu())
        db_embs = torch.cat(embs, dim=0)
        torch.save({"names": tile_names, "embeddings": db_embs}, db_embs_path)
        db_embs = db_embs.to(device)

    results = []
    top1_correct, top5_correct = 0, 0

    pbar = tqdm(df_uav.iterrows(), total=len(df_uav), desc="Coarse-to-Fine Localization")
    for _, row in pbar:
        drone_filename = row["filename"]
        drone_path = os.path.join(drone_dir, drone_filename)
        gt_lat, gt_lon = row["lat"], row["lon"]

        pixel_x = (gt_lon - map_bounds["LT_lon_map"]) / (map_bounds["RB_lon_map"] - map_bounds["LT_lon_map"]) * w_map
        pixel_y = (map_bounds["LT_lat_map"] - gt_lat) / (map_bounds["LT_lat_map"] - map_bounds["RB_lat_map"]) * h_map

        gt_tile_name = find_gt_tile(gt_lat, gt_lon, map_bounds, w_map, h_map, df_tiles)
        gt_tile_path = os.path.join(tiles_dir, gt_tile_name) if gt_tile_name else None

        img = Image.open(drone_path).convert("RGB")
        tensor = eval_transform(img).unsqueeze(0).to(device)

        with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
            query_emb, _ = model(tensor)
            similarities = F.cosine_similarity(query_emb, db_embs)

        top5_idx = torch.topk(similarities, k=5).indices.cpu().numpy()
        top5_tiles = [tile_names[i] for i in top5_idx]

        is_top1, is_top5 = False, False
        for rank, t_idx in enumerate(top5_idx):
            tile_info = df_tiles[df_tiles["tile_name"] == tile_names[t_idx]].iloc[0]
            tx, ty, ts = tile_info["pixel_x"], tile_info["pixel_y"], tile_info["tile_size"]
            if tx <= pixel_x <= tx + ts and ty <= pixel_y <= ty + ts:
                if rank == 0:
                    is_top1 = True
                is_top5 = True

        top1_correct += int(is_top1)
        top5_correct += int(is_top5)

        img0_base = load_image_for_lightglue(drone_path, device)
        best_score = -1
        best_tile = top5_tiles[0]
        best_px, best_py = 1024.0, 1024.0
        best_angle = 0
        best_kps0, best_kps1 = None, None

        for candidate_tile in top5_tiles:
            img1 = load_image_for_lightglue(os.path.join(tiles_dir, candidate_tile), device)
            with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
                feats1 = extractor.extract(img1)

            for angle in (0, 45, 90, 135, 180, 225, 270, 315):
                img0_rot = TF.rotate(img0_base, angle)
                with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    feats0 = extractor.extract(img0_rot)
                    matches01 = matcher({"image0": feats0, "image1": feats1})

                feats0_r, feats1_r, matches01_r = [rbd(x) for x in (feats0, feats1, matches01)]
                matches = matches01_r["matches"]
                score = len(matches)

                if score > best_score:
                    best_score = score
                    best_tile = candidate_tile
                    best_angle = angle

                    if score >= 4:
                        m_kps0 = feats0_r["keypoints"][matches[..., 0]].cpu().numpy()
                        m_kps1 = feats1_r["keypoints"][matches[..., 1]].cpu().numpy()
                        best_kps0, best_kps1 = m_kps0, m_kps1

                        h_mat, _ = cv2.findHomography(m_kps0, m_kps1, cv2.USAC_MAGSAC, 3.0)
                        if h_mat is not None:
                            center_drone = np.array([[[320.0, 320.0]]])
                            center_sat = cv2.perspectiveTransform(center_drone, h_mat)
                            best_px = float(center_sat[0][0][0] * (2048 / 640))
                            best_py = float(center_sat[0][0][1] * (2048 / 640))
                        else:
                            best_px, best_py = 1024.0, 1024.0
                    else:
                        best_kps0, best_kps1 = None, None
                        best_px, best_py = 1024.0, 1024.0

        pred_lat, pred_lon = pixel_to_gps(best_px, best_py, best_tile, df_tiles, map_bounds, w_map, h_map)
        mhe = calculate_haversine_distance(gt_lat, gt_lon, pred_lat, pred_lon)

        results.append(
            {
                "filename": drone_filename,
                "drone_path": drone_path,
                "pred_tile_path": os.path.join(tiles_dir, best_tile),
                "gt_tile_path": gt_tile_path,
                "is_top1": is_top1,
                "is_top5": is_top5,
                "mhe": mhe,
                "best_angle": best_angle,
                "m_kps0": best_kps0,
                "m_kps1": best_kps1,
                "matches_count": best_score,
            }
        )
        pbar.set_postfix({"MHE": f"{mhe:.2f}m"})

    errors = [r["mhe"] for r in results]
    summary = {
        "total_samples": len(df_uav),
        "mean_mhe": float(np.mean(errors)),
        "median_mhe": float(np.median(errors)),
        "p90_mhe": float(np.percentile(errors, 90)),
        "top1_accuracy": top1_correct / len(df_uav),
        "top5_accuracy": top5_correct / len(df_uav),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "summary_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 50)
    print(f"Top-1 Retrieval Accuracy: {summary['top1_accuracy'] * 100:.2f}%")
    print(f"Top-5 Retrieval Accuracy: {summary['top5_accuracy'] * 100:.2f}%")
    print(f"Mean Haversine Error:     {summary['mean_mhe']:.2f} m")
    print(f"Median Haversine Error:   {summary['median_mhe']:.2f} m")
    print(f"P90 Haversine Error:      {summary['p90_mhe']:.2f} m")
    print("=" * 50)

    if args.save_plots:
        os.makedirs(args.assets_dir, exist_ok=True)
        results.sort(key=lambda x: x["mhe"])
        save_match_visualization(
            results[0],
            f"Best Match | MHE: {results[0]['mhe']:.2f} m | Inliers: {results[0]['matches_count']}",
            os.path.join(args.assets_dir, "best_match.png"),
        )
        save_match_visualization(
            results[-1],
            f"Failure Case | MHE: {results[-1]['mhe']:.2f} m | Inliers: {results[-1]['matches_count']}",
            os.path.join(args.assets_dir, "failure_case.png"),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate UAV Coarse-to-Fine Geo-Localization")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--weights-path", type=str, default="weights/uav_multitask_best.pth")
    parser.add_argument("--test-scene", type=str, default="06")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--assets-dir", type=str, default="assets")
    parser.add_argument("--save-plots", action="store_true")
    evaluate(parser.parse_args())