"""
Military Aircraft Detection - Dataset Preparation
Author: Dhruv Gaur
GitHub: https://github.com/dhruvgaur10/Military-Aircraft-Detection

Converts the Kaggle Military Aircraft Detection Dataset into YOLO format.

Source: https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset

The raw dataset provides a single CSV (labels_with_split.csv) with one row per
bounding box:

    filename,width,height,class,xmin,ymin,xmax,ymax,split

and the images themselves under dataset/<filename>.jpg. The split column
(train/validation/test) is authoritative and used as-is - no re-splitting.

Usage:
    python scripts/prepare_dataset.py --raw D:/aircraft_raw
    python scripts/prepare_dataset.py --raw D:/aircraft_raw --min-samples 40
"""

import argparse
import csv
import os
import shutil
from collections import Counter, defaultdict

# Kaggle split name -> YOLO split directory name
SPLIT_MAP = {
    'train': 'train',
    'validation': 'valid',
    'valid': 'valid',
    'val': 'valid',
    'test': 'test',
}


def load_rows(csv_path):
    """Read the flat label CSV into a list of dicts, skipping malformed rows."""
    rows = []
    skipped = 0
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                row['width'] = float(row['width'])
                row['height'] = float(row['height'])
                row['xmin'] = float(row['xmin'])
                row['ymin'] = float(row['ymin'])
                row['xmax'] = float(row['xmax'])
                row['ymax'] = float(row['ymax'])
            except (KeyError, ValueError, TypeError):
                skipped += 1
                continue
            split = SPLIT_MAP.get(row.get('split', '').strip().lower())
            if split is None:
                skipped += 1
                continue
            row['split'] = split
            rows.append(row)

    print(f"  Rows read      : {len(rows)}")
    if skipped:
        print(f"  Rows skipped   : {skipped} (malformed or unrecognised split)")
    return rows


def to_yolo(row, class_to_id):
    """Convert one VOC-corner row to a YOLO label line. Returns None if degenerate."""
    w, h = row['width'], row['height']
    xmin = max(0.0, min(row['xmin'], w))
    xmax = max(0.0, min(row['xmax'], w))
    ymin = max(0.0, min(row['ymin'], h))
    ymax = max(0.0, min(row['ymax'], h))

    if xmax <= xmin or ymax <= ymin or w <= 0 or h <= 0:
        return None

    xc = ((xmin + xmax) / 2) / w
    yc = ((ymin + ymax) / 2) / h
    bw = (xmax - xmin) / w
    bh = (ymax - ymin) / h

    if not all(0 <= v <= 1 for v in (xc, yc, bw, bh)) or bw <= 0 or bh <= 0:
        return None

    cls_id = class_to_id[row['class']]
    return f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"


def find_image(images_dir, filename, ext_cache):
    """Resolve a bare filename to its actual file on disk (extension may vary)."""
    if filename in ext_cache:
        return ext_cache[filename]

    for ext in ('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'):
        candidate = os.path.join(images_dir, filename + ext)
        if os.path.exists(candidate):
            ext_cache[filename] = candidate
            return candidate

    ext_cache[filename] = None
    return None


def main():
    parser = argparse.ArgumentParser(description='Prepare the aircraft dataset for YOLO training')
    parser.add_argument('--raw', type=str, required=True,
                        help='Path to the extracted Kaggle dataset (contains labels_with_split.csv and dataset/)')
    parser.add_argument('--out', type=str, default=None,
                        help='Output directory (default: <project>/data)')
    parser.add_argument('--min-samples', type=int, default=30,
                        help='Drop classes with fewer box instances than this (default: 30)')
    parser.add_argument('--move', action='store_true',
                        help='Move images instead of copying (saves disk, consumes the raw copy)')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    out_root = args.out or os.path.join(project_root, 'data')

    csv_path = os.path.join(args.raw, 'labels_with_split.csv')
    images_dir = os.path.join(args.raw, 'dataset')

    print("=" * 58)
    print("  DATASET PREPARATION - CSV to YOLO")
    print("=" * 58)
    print(f"  Raw source : {args.raw}")
    print(f"  Labels CSV : {csv_path}")
    print(f"  Images dir : {images_dir}")
    print(f"  Output     : {out_root}")
    print("=" * 58)

    if not os.path.exists(csv_path):
        raise SystemExit(f"labels_with_split.csv not found at {csv_path}")
    if not os.path.isdir(images_dir):
        raise SystemExit(f"Images directory not found at {images_dir}")

    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit("No usable rows found in the labels CSV.")

    # Pass 1: class frequency, to decide what survives --min-samples
    counts = Counter(r['class'] for r in rows)
    keep_classes = sorted(c for c, n in counts.items() if n >= args.min_samples)
    dropped = sorted(c for c, n in counts.items() if n < args.min_samples)

    print(f"\n  Classes discovered : {len(counts)}")
    print(f"  Total box rows     : {len(rows)}")
    if dropped:
        print(f"  Dropped (<{args.min_samples} boxes): {len(dropped)} -> {', '.join(dropped)}")
    print(f"  Classes kept       : {len(keep_classes)}")

    if not keep_classes:
        raise SystemExit("No classes survived the min-samples filter.")

    class_to_id = {c: i for i, c in enumerate(keep_classes)}

    # Pass 2: group rows by (filename, split), converting boxes as we go
    print("\n  Converting annotations...")
    by_image = defaultdict(lambda: {'split': None, 'lines': []})
    kept_boxes = skipped_boxes = 0

    for row in rows:
        if row['class'] not in class_to_id:
            skipped_boxes += 1
            continue
        line = to_yolo(row, class_to_id)
        if line is None:
            skipped_boxes += 1
            continue
        entry = by_image[row['filename']]
        entry['split'] = row['split']
        entry['lines'].append(line)
        kept_boxes += 1

    print(f"  Boxes converted    : {kept_boxes}")
    print(f"  Boxes skipped      : {skipped_boxes} (filtered class or invalid geometry)")
    print(f"  Images with labels : {len(by_image)}")

    # Pass 3: resolve image files and write out the YOLO tree
    print("\n  Resolving and copying images...")
    ext_cache = {}
    split_counts = Counter()
    missing_images = 0

    for name in ('train', 'valid', 'test'):
        split_dir = os.path.join(out_root, name)
        if os.path.isdir(split_dir):
            shutil.rmtree(split_dir)
        os.makedirs(os.path.join(split_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(split_dir, 'labels'), exist_ok=True)

    for filename, entry in by_image.items():
        split = entry['split']
        src_img = find_image(images_dir, filename, ext_cache)
        if src_img is None:
            missing_images += 1
            continue

        ext = os.path.splitext(src_img)[1]
        dst_img = os.path.join(out_root, split, 'images', filename + ext)
        dst_lbl = os.path.join(out_root, split, 'labels', filename + '.txt')

        if args.move:
            shutil.move(src_img, dst_img)
        else:
            shutil.copy2(src_img, dst_img)

        with open(dst_lbl, 'w') as f:
            f.write('\n'.join(entry['lines']) + '\n')

        split_counts[split] += 1

    if missing_images:
        print(f"  [!] Images referenced in CSV but not found on disk: {missing_images}")

    for name in ('train', 'valid', 'test'):
        print(f"  {name:<6}: {split_counts.get(name, 0)} images")

    yaml_path = os.path.join(out_root, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write("# Military Aircraft Detection Dataset\n")
        f.write("# Generated by scripts/prepare_dataset.py\n")
        f.write("# Source: https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset\n\n")
        f.write("train: train/images\n")
        f.write("val: valid/images\n")
        f.write("test: test/images\n\n")
        f.write(f"nc: {len(keep_classes)}\n")
        f.write(f"names: {keep_classes}\n")

    print("\n" + "=" * 58)
    print(f"  Classes  : {len(keep_classes)}")
    print(f"  data.yaml: {yaml_path}")
    print("  PREPARATION COMPLETE")
    print("=" * 58)


if __name__ == "__main__":
    main()
