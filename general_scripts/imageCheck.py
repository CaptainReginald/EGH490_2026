import sys
from pathlib import Path

def remove_unmatched_images(img_dir, txt_dir, img_ext=".jpg", txt_ext=".txt", rel_prefix="data/images/train/", output_list="remaining_images.txt"):
    img_dir = Path(img_dir)
    txt_dir = Path(txt_dir)
    txt_basenames = {f.stem for f in txt_dir.glob(f"*{txt_ext}")}

    kept_images = []
    removed = 0
    for img_file in img_dir.glob(f"*{img_ext}"):
        if img_file.stem not in txt_basenames:
            img_file.unlink()
            print(f"Removed: {img_file.name}")
            removed += 1
        else:
            kept_images.append(f"{rel_prefix}{img_file.name}")

    # Write kept images to file
    with open(output_list, "w") as f:
        for line in kept_images:
            f.write(line + "\n")

    print(f"Done. Removed {removed} images. {len(kept_images)} images remain. List written to {output_list}.")

if __name__ == "__main__":
    remove_unmatched_images(
        "/home/alexanderjones/hpc-home/Data/cslics/2025_nov_spawn/project_334101_dataset_2025_11_12_05_39_15_ultralytics yolo detection 1.0/images/train",
        "/home/alexanderjones/hpc-home/Data/cslics/2025_nov_spawn/project_334101_dataset_2025_11_12_05_39_15_ultralytics yolo detection 1.0/labels/train"
    )