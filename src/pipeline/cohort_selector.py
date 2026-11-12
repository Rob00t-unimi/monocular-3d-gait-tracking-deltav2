# -*- coding: utf-8 -*-
"""
Selection Script for 40 Benchmark Videos for 3D Gait Tracking Inference.

Partitions the Gait360 dataset into 8 balanced perspective categories (5 videos each):
1. 90° Sagittale (Destra -> Sinistra)                       [5 video]
2. 90° Sagittale (Sinistra -> Destra)                       [5 video]
3. 0-deg Pure Frontal (Approaching)                            [5 video]
4. 180-deg Pure Posterior (Receding)                          [5 video]
5. 45° Obliquo (Destra -> Sinistra): Front-Obliquo          [5 video]
6. 45° Obliquo (Destra -> Sinistra): Rear-Obliquo           [5 video]
7. 45° Obliquo (Sinistra -> Destra): Front-Obliquo          [5 video]
8. 45° Obliquo (Sinistra -> Destra): Rear-Obliquo           [5 video]

Strict Constraints:
- 40 unique participants across the 40 videos (0 duplicates).
- 100% of videos have IMU sensor on the right leg (T1, T2, T1ALT, T2ALT).
- Highest GT signal quality prioritized.

Outputs:
- data/selected_inference_videos_40.csv
- data/selected_inference_videos_40.txt
"""

import os
import re
import json
import csv
from pathlib import Path
from datetime import datetime
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_60_CSV = PROJECT_ROOT / "data" / "selected_inference_videos_60.csv"
OUTPUT_40_CSV = PROJECT_ROOT / "data" / "selected_inference_videos_40.csv"
OUTPUT_40_TXT = PROJECT_ROOT / "data" / "selected_inference_videos_40.txt"

CATEGORIES_8 = [
    "1. 90° Sagittale (Destra -> Sinistra)",
    "2. 90° Sagittale (Sinistra -> Destra)",
    "3. 0-deg Pure Frontal (Approaching)",
    "4. 180-deg Pure Posterior (Receding)",
    "5. 45-deg Oblique (Right -> Left) - Front-Oblique (Approaching)",
    "6. 45-deg Oblique (Right -> Left) - Rear-Oblique (Receding)",
    "7. 45-deg Oblique (Left -> Right) - Front-Oblique (Approaching)",
    "8. 45-deg Oblique (Left -> Right) - Rear-Oblique (Receding)"
]

def map_to_8class(row: pd.Series) -> str:
    cat = str(row['category_name'])
    sub = str(row.get('sub_perspective', ''))
    angle = str(row.get('angle', ''))
    
    if '90' in cat and ('Destra -> Sinistra' in cat or 'Right -> Left' in cat):
        return CATEGORIES_8[0]
    if '90' in cat and ('Sinistra -> Destra' in cat or 'Left -> Right' in cat):
        return CATEGORIES_8[1]
    if 'Front' in cat or angle == 'Front':
        return CATEGORIES_8[2]
    if 'Post' in cat or 'Rear' in cat or angle == 'Rear':
        return CATEGORIES_8[3]
    if '45' in cat and ('Destra -> Sinistra' in cat or 'Right -> Left' in cat):
        return CATEGORIES_8[4] if 'Front' in sub else CATEGORIES_8[5]
    if '45' in cat and ('Sinistra -> Destra' in cat or 'Left -> Right' in cat):
        return CATEGORIES_8[6] if 'Front' in sub else CATEGORIES_8[7]
    raise ValueError(f"Unknown category for row: {row['video_name']}")

def select_40_benchmark() -> pd.DataFrame:
    if not INPUT_60_CSV.exists():
        raise FileNotFoundError(f"Input file {INPUT_60_CSV} not found.")

    df_60 = pd.read_csv(INPUT_60_CSV)

    # 1. Take all 20 videos from the 45 degree categories (already balanced 5-5-5-5)
    v45 = df_60[df_60['category_name'].str.contains('45')].copy()
    used_persons = set(v45['person'].tolist())

    if len(used_persons) != 20:
        raise ValueError(f"Expected 20 unique persons in 45 deg videos, found {len(used_persons)}")

    # 2. Select 5 unique persons for each of the remaining 4 categories
    selected_other = []
    category_order = [
        ('90° Sagittale (Destra -> Sinistra)', '90_R2L'),
        ('90° Sagittale (Sinistra -> Destra)', '90_L2R'),
        ('Front', 'Front'),
        ('Rear', 'Rear')
    ]

    for cat_pattern, angle_pattern in category_order:
        mask = df_60['category_name'].str.contains(cat_pattern, regex=False) | (df_60['angle'] == angle_pattern)
        if 'Post' in cat_pattern:
            mask = mask | df_60['category_name'].str.contains('Post', regex=False)
        sub = df_60[mask].sort_values(
            by=['gt_score', 'gt_rom_deg'], ascending=[False, True]
        )
        picked = []
        for _, row in sub.iterrows():
            p = row['person']
            if p not in used_persons:
                picked.append(row)
                used_persons.add(p)
                if len(picked) == 5:
                    break
        if len(picked) != 5:
            raise ValueError(f"Could not select 5 unique persons for category {cat}")
        selected_other.extend(picked)

    # 3. Concatenate and assign 8-class category
    df_other = pd.DataFrame(selected_other)
    all_40 = pd.concat([df_other, v45], ignore_index=True)
    all_40['category_8class'] = all_40.apply(map_to_8class, axis=1)

    # Verify constraints
    assert len(all_40) == 40, f"Expected 40 videos, got {len(all_40)}"
    assert len(set(all_40['person'].tolist())) == 40, "Persons are not unique!"
    assert all(all_40['sensor_position'] == 'leg'), "Found non-leg sensor position!"
    
    cat_counts = all_40['category_8class'].value_counts()
    for cat in CATEGORIES_8:
        assert cat_counts.get(cat, 0) == 5, f"Category {cat} does not have exactly 5 videos!"

    # Sort deterministically by 8-class category and video name
    all_40 = all_40.sort_values(by=['category_8class', 'video_name']).reset_index(drop=True)
    return all_40

def export_manifests(df_40: pd.DataFrame):
    # 1. Export CSV
    cols = [
        'video_name', 'person', 'category_8class', 'category_name', 'sub_perspective',
        'experiment', 'test', 'sensor_position', 'angle', 'direction',
        'kps_origin', 'gt_status', 'gt_score', 'gt_rom_deg', 'gt_step_freq_hz',
        'ref_frame_idx', 'frames_num', 'condition', 'video_path', 'sensor_csv'
    ]
    df_40[cols].to_csv(OUTPUT_40_CSV, index=False, encoding='utf-8')
    print(f"Exported CSV: {OUTPUT_40_CSV}")

    # 2. Export TXT
    with open(OUTPUT_40_TXT, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("  3D GAIT TRACKING: 40 BENCHMARK INFERENCE VIDEOS (8 CATEGORIES x 5 VIDEOS)\n")
        f.write(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("  Total Videos: 40 / 40\n")
        f.write("  Unique Subjects: 40 / 40 (0 duplicates - balanced cohort)\n")
        f.write("  IMU Sensor: Right Thigh (T1, T2, T1ALT, T2ALT)\n")
        f.write("=" * 80 + "\n\n")

        for cat_name in CATEGORIES_8:
            sub = df_40[df_40['category_8class'] == cat_name]
            f.write(f"### {cat_name} ({len(sub)} / 5)\n")
            f.write("-" * 80 + "\n")
            for i, (_, r) in enumerate(sub.iterrows(), 1):
                f.write(
                    f"{i:2d}. {r['video_name']:<22} | "
                    f"Subject: {r['person']:<4} | "
                    f"Test: {r['test']:<5} | "
                    f"Keypoints: {r['kps_origin']:<22} | "
                    f"GT: {r['gt_status']:<8} (ROM {r['gt_rom_deg']}°, {r['gt_step_freq_hz']}Hz) | "
                    f"{r['condition']}\n"
                )
            f.write("\n")
    print(f"Exported TXT: {OUTPUT_40_TXT}")

def main(force: bool = False):
    if OUTPUT_40_CSV.exists() and OUTPUT_40_TXT.exists() and not force:
        df_40 = pd.read_csv(OUTPUT_40_CSV)
        print(f"Benchmark cohort manifest already exists at {OUTPUT_40_CSV.name}. Loading from cache (use --force to regenerate)...")
        print(f"- Total videos: {len(df_40)}")
        print(f"- Unique participants: {len(set(df_40['person'].tolist()))}")
        print(f"- Categories: {len(df_40['category_8class'].unique())} (5 videos each)")
        return df_40

    print("Selecting 40 benchmark videos across 8 balanced categories...")
    df_40 = select_40_benchmark()
    for col in ['video_path', 'sensor_csv']:
        if col in df_40.columns:
            df_40[col] = df_40[col].apply(lambda s: s.replace('\\', '/')[s.replace('\\', '/').find('data/Gait_Dataset/'):] if isinstance(s, str) and 'data/Gait_Dataset/' in s.replace('\\', '/') else s)
    export_manifests(df_40)
    print("\nBenchmark selection completed successfully!")
    print(f"- Total videos: {len(df_40)}")
    print(f"- Unique participants: {len(set(df_40['person'].tolist()))}")
    print(f"- Categories: {len(CATEGORIES_8)} (5 videos each)")
    return df_40

if __name__ == "__main__":
    main()
