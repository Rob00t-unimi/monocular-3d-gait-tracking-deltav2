"""Dataset scanning, multi-modal video/IMU alignment, and candidate quality caching pipeline."""

from pathlib import Path
import logging
from src.data_io.dataset_scanner import DatasetScanner

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def run_dataset_alignment(dataset_dir: str = "data/Gait_Dataset", cache_path: str = "data/cached_dataset_map.json", force: bool = False) -> list:
    """Scan the multi-view gait dataset and serialize the aligned video/OpenPose/IMU trial map."""
    ds_full = str((PROJECT_ROOT / dataset_dir).resolve())
    cache_full = str((PROJECT_ROOT / cache_path).resolve())
    base_dir = str(Path(cache_full).parent)
    scanner = DatasetScanner(base_dir=base_dir)
    subjects = scanner.scan(dataset_dir=ds_full, cache_path=cache_full, force=force)
    logging.info("Scanned %d subjects into %s", len(subjects), cache_full)
    return subjects

# --- Candidate Video Quality Evaluation & Cache Builder ---
# -*- coding: utf-8 -*-
"""
Precomputes candidate video metadata and GT quality metrics for the 60-video selection GUI.
Saves results into data/candidate_videos_cache.json.
"""

import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def get_x_coord(p_json: str):
    try:
        with open(p_json, 'r', encoding='utf-8') as f:
            d = json.load(f)
        if d.get('people'):
            kps = d['people'][0]['pose_keypoints_2d']
            x = kps[8*3] # MidHip x
            c = kps[8*3 + 2]
            if c > 0.1:
                return float(x)
    except Exception:
        pass
    return None

def detect_direction(kps_dict: dict):
    if not kps_dict:
        return None
    keys = sorted([int(k) for k in kps_dict.keys()])
    if len(keys) < 10:
        return None
    
    x_start = None
    for k in keys[:max(3, len(keys)//4)]:
        p_file = PROJECT_ROOT / kps_dict[str(k)]
        x_start = get_x_coord(str(p_file))
        if x_start is not None:
            break
            
    x_end = None
    for k in reversed(keys[-max(3, len(keys)//4):]):
        p_file = PROJECT_ROOT / kps_dict[str(k)]
        x_end = get_x_coord(str(p_file))
        if x_end is not None:
            break
            
    if x_start is not None and x_end is not None:
        return 'R2L' if x_start > x_end else 'L2R'
    return None

def evaluate_gt_quality(sensor_csv_rel: str, gt_eval_cache: dict):
    if not sensor_csv_rel:
        return None
        
    csv_name = Path(sensor_csv_rel).name
    proc_path = PROJECT_ROOT / "data" / "gt_cache" / f"{csv_name}_processed.csv"
    if not proc_path.exists():
        proc_path = PROJECT_ROOT / sensor_csv_rel
        if not proc_path.exists():
            return None
            
    if str(proc_path) in gt_eval_cache:
        return gt_eval_cache[str(proc_path)]
        
    try:
        df = pd.read_csv(proc_path)
        if 'Pitch' not in df or len(df) < 30:
            gt_eval_cache[str(proc_path)] = None
            return None
            
        dt = float(np.median(np.diff(df['Timestamp'].values)))
        pitch = df['Pitch'].values
        p_clean = pitch - np.mean(pitch)
        rom = float(np.max(p_clean) - np.min(p_clean))
        p_var = float(np.var(p_clean))
        
        # Biomechanical / sanity checks
        if p_var < 1e-4 or rom > 180.0 or np.isnan(rom) or np.isnan(p_var):
            res = {'status': 'Invalid', 'rom': rom, 'score': 0, 'freq': 0.0, 'purity': 0.0}
            gt_eval_cache[str(proc_path)] = res
            return res
            
        freqs = np.fft.rfftfreq(len(p_clean), d=dt)
        fft_vals = np.abs(np.fft.rfft(p_clean))
        valid_idx = (freqs >= 0.5) & (freqs <= 3.5)
        
        if not np.any(valid_idx):
            res = {'status': 'Imprecise', 'rom': round(rom, 1), 'score': 1, 'freq': 0.0, 'purity': 0.0}
            gt_eval_cache[str(proc_path)] = res
            return res
            
        dom_idx = np.argmax(fft_vals[valid_idx])
        dom_f = float(freqs[valid_idx][dom_idx])
        band_power = float(np.sum(fft_vals[valid_idx]**2)) + 1e-9
        peak_power = float(fft_vals[valid_idx][dom_idx]**2)
        ratio = peak_power / band_power
        
        # Classification
        if ratio >= 0.35 and 20.0 <= rom <= 75.0:
            status = 'Optimal'
            score = 3
        elif ratio >= 0.18 and 12.0 <= rom <= 90.0:
            status = 'Good'
            score = 2
        else:
            status = 'Imprecise'
            score = 1
            
        res = {
            'status': status,
            'rom': round(rom, 1),
            'score': score,
            'freq': round(dom_f, 2),
            'purity': round(ratio, 2)
        }
        gt_eval_cache[str(proc_path)] = res
        return res
    except Exception as e:
        gt_eval_cache[str(proc_path)] = None
        return None

def build_cache(output_path: Path = None, force: bool = False):
    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "candidate_videos_cache.json"

    if output_path.exists() and not force:
        print(f"Candidate videos cache already exists at {output_path.name}. Loading from cache (use --force to recompute)...")
        with open(output_path, 'r', encoding='utf-8') as f:
            cached = json.load(f)
        print(f"Total valid candidate videos: {cached.get('total_candidates', 0)}")
        print(f"Unique participants available: {cached.get('unique_persons_count', 0)}")
        cat_map = cached.get('category_map', {})
        for cat_key, cands in cached.get('candidates_by_cat', {}).items():
            n_auto = sum(1 for c in cands if c.get('kps_origin') in ('Automatic', 'Automatico'))
            n_ott = sum(1 for c in cands if c.get('gt_status') in ('Optimal', 'Ottimo'))
            print(f"  [{cat_map.get(cat_key, cat_key)}]: {len(cands)} videos ({n_auto} Automatic, {n_ott} GT Optimal)")
        return cached

    start_time = time.time()
    print("Building candidate videos cache...")
    
    map_path = PROJECT_ROOT / "data" / "cached_dataset_map.json"
    state_path = PROJECT_ROOT / "data" / "videos_state.json"
    
    if not map_path.exists() or not state_path.exists():
        raise FileNotFoundError("Missing cached_dataset_map.json or videos_state.json")
        
    with open(map_path, 'r', encoding='utf-8') as f:
        dataset_map = json.load(f)
    with open(state_path, 'r', encoding='utf-8') as f:
        videos_state = json.load(f)
        
    # Map videos from dataset map
    video_lookup = {}
    for p_id, exps in dataset_map.items():
        for e_id, tests in exps.items():
            for t_id, angles in tests.items():
                is_alt = t_id.endswith("ALT")
                t_num = int(t_id[1:-3]) if is_alt else int(t_id[1:])
                sensor_pos = "leg" if (is_alt or t_num in [1, 2]) else ("arm" if t_num in [3, 4] else "head")
                
                # Strict guarantee: only videos where the sensor is on the right leg
                if sensor_pos != "leg":
                    continue

                sensor_csv = angles.get('sensor_data')
                for a_id, val in angles.items():
                    if isinstance(val, dict) and 'video' in val:
                        v_name = Path(val['video']).name
                        video_lookup[v_name] = {
                            'person': p_id,
                            'person_num': int(p_id.replace('P', '')),
                            'experiment': e_id,
                            'test': t_id,
                            'sensor_position': sensor_pos,
                            'angle': a_id,
                            'video_path': str(val['video']).replace('\\', '/'),
                            'sensor_csv': str(sensor_csv).replace('\\', '/') if sensor_csv else None,
                            'sensor_csv_rel': sensor_csv,
                            'keypoints': val.get('keypoints', {})
                        }
                        
    gt_eval_cache = {}
    category_map = {
        '90_R2L': '90° Sagittal (Right -> Left)',
        '90_L2R': '90° Sagittal (Left -> Right)',
        '45_R2L': '45° Oblique (Right -> Left)',
        '45_L2R': '45° Oblique (Left -> Right)',
        'Front': 'Frontal (Approaching)',
        'Rear': 'Posterior (Receding)'
    }
    
    candidates_by_cat = {cat_id: [] for cat_id in category_map}
    total_valid = 0
    unique_persons = set()
    
    for v_name, st in videos_state.items():
        if v_name not in video_lookup:
            continue
            
        # 1. Filter out discarded
        if st.get('specific', {}).get('discarded', False) or st.get('fullbody', {}).get('discarded', False):
            continue
            
        ref_frame_idx = st.get('ref_frame_idx')
        kps_specific = st.get('specific', {}).get('keypoints', [])
        if ref_frame_idx is None or len(kps_specific) < 3:
            continue
            
        meta = video_lookup[v_name]
        v_path = Path(meta['video_path'])
        if not v_path.is_absolute():
            v_path = PROJECT_ROOT / v_path
        if not v_path.exists():
            continue
            
        if not meta['sensor_csv']:
            continue
        s_path = Path(meta['sensor_csv'])
        if not s_path.is_absolute():
            s_path = PROJECT_ROOT / s_path
        if not s_path.exists():
            continue
            
        # 2. Evaluate GT quality
        gt_info = evaluate_gt_quality(meta['sensor_csv_rel'], gt_eval_cache)
        if gt_info is None or gt_info['status'] == 'Invalid':
            continue
            
        # 3. Determine category
        angle = meta['angle']
        cat_key = None
        dir_label = "N/A"
        
        if angle == 'Front':
            cat_key = 'Front'
            dir_label = "Approaching"
        elif angle == 'Rear':
            cat_key = 'Rear'
            dir_label = "Receding"
        elif angle in ['90', '45']:
            d_code = detect_direction(meta['keypoints'])
            if d_code:
                cat_key = f"{angle}_{d_code}"
                dir_label = "Right -> Left" if d_code == 'R2L' else "Left -> Right"
            else:
                continue
        else:
            continue
            
        # 4. Check Keypoints Origin (Automatic vs Repositioned by Operator)
        # In ThighPointSelector, P2 and P3 are cast to int in automatic computation
        p1, p2, p3 = kps_specific[0], kps_specific[1], kps_specific[2]
        p2_is_int = float(p2[0]).is_integer() and float(p2[1]).is_integer()
        p3_is_int = float(p3[0]).is_integer() and float(p3[1]).is_integer()
        
        if p2_is_int and p3_is_int:
            kps_origin = "Automatic"
            kps_rank = 0
        else:
            kps_origin = "Repositioned (Operator)"
            kps_rank = 1
            
        is_alt = "_ALT" in v_name
        frames_num = len(meta['keypoints'])
        
        cand_entry = {
            'video_name': v_name,
            'person': meta['person'],
            'person_num': meta['person_num'],
            'experiment': meta['experiment'],
            'test': meta['test'],
            'sensor_position': meta['sensor_position'],
            'angle': angle,
            'direction': dir_label,
            'category_key': cat_key,
            'category_name': category_map[cat_key],
            'kps_origin': kps_origin,
            'kps_rank': kps_rank,
            'gt_status': gt_info['status'],
            'gt_score': gt_info['score'],
            'gt_rom': gt_info['rom'],
            'gt_freq': gt_info['freq'],
            'gt_purity': gt_info['purity'],
            'ref_frame_idx': int(ref_frame_idx),
            'frames_num': int(frames_num),
            'is_alt': bool(is_alt),
            'condition': "Alt (Loose Clothing)" if is_alt else "Standard",
            'video_path': meta['video_path'],
            'sensor_csv': meta['sensor_csv']
        }
        
        candidates_by_cat[cat_key].append(cand_entry)
        total_valid += 1
        unique_persons.add(meta['person'])
        
    # Sort candidates in each category according to priority:
    # 1. kps_rank ascending (Automatic=0 first)
    # 2. gt_score descending (Optimal=3, Good=2, Imprecise=1)
    # 3. is_alt ascending (Standard=False first)
    # 4. person_num ascending
    for cat_key in candidates_by_cat:
        candidates_by_cat[cat_key].sort(
            key=lambda x: (x['kps_rank'], -x['gt_score'], 1 if x['is_alt'] else 0, x['person_num'])
        )
        
    out_dict = {
        'total_candidates': total_valid,
        'unique_persons_count': len(unique_persons),
        'unique_persons_list': sorted(list(unique_persons), key=lambda p: int(p.replace('P', ''))),
        'category_map': category_map,
        'candidates_by_cat': candidates_by_cat
    }
    
    if output_path is None:
        output_path = PROJECT_ROOT / "data" / "candidate_videos_cache.json"
        
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(out_dict, f, indent=2)
        
    elapsed = time.time() - start_time
    print(f"Cache successfully generated in {elapsed:.2f}s!")
    print(f"Total valid candidate videos: {total_valid}")
    print(f"Unique participants available: {len(unique_persons)}")
    for cat_key, cands in candidates_by_cat.items():
        n_auto = sum(1 for c in cands if c['kps_origin'] in ('Automatic', 'Automatico'))
        n_ott = sum(1 for c in cands if c['gt_status'] == 'Optimal')
        print(f"  [{category_map[cat_key]}]: {len(cands)} videos ({n_auto} Automatic, {n_ott} GT Optimal)")
        
    return out_dict

def build_candidate_cache_main(force: bool = False):
    return build_cache(force=force)

def run_dataset_pipeline(
    mode: str = "all",
    dataset_dir: str = "data/Gait_Dataset",
    cache_path: str = "data/cached_dataset_map.json",
    force: bool = False,
) -> None:
    """Execute dataset scanning and/or candidate cache generation."""
    if mode in ("scan", "all"):
        run_dataset_alignment(dataset_dir=dataset_dir, cache_path=cache_path, force=force)
    if mode in ("candidates", "all"):
        build_candidate_cache_main(force=force)
