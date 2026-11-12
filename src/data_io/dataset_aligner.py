"""Dataset integrity verifier and structure aligner based on dataset_manifest.json.

Cross-platform implementation with fast partial-chunk SHA-256 hashing,
collision-safe temporary relocation, and dry-run preview capabilities.
"""

import os
import hashlib
import json
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DatasetAligner:
    """Verifies file hashes and reorganizes local datasets against an authoritative manifest."""

    def __init__(self, chunk_size: int = 1024 * 1024):
        """Initialize the aligner.

        Args:
            chunk_size: Size of data chunks (in bytes) read for partial SHA-256 hashing.
        """
        self.chunk_size = chunk_size

    def compute_file_hash(self, filepath: str) -> str:
        """Compute deterministic fast SHA-256 hash incorporating file size and head/mid/tail chunks.

        Args:
            filepath: Absolute or relative file path to hash.

        Returns:
            Hexadecimal SHA-256 digest string, or empty string on read error.
        """
        hasher = hashlib.sha256()
        try:
            size = os.path.getsize(filepath)
            hasher.update(str(size).encode("utf-8"))

            with open(filepath, "rb") as f:
                chunk = f.read(self.chunk_size)
                hasher.update(chunk)

                if size > self.chunk_size * 3:
                    f.seek(size // 2)
                    hasher.update(f.read(self.chunk_size))
                    f.seek(-min(self.chunk_size, size), os.SEEK_END)
                    hasher.update(f.read(self.chunk_size))

            return hasher.hexdigest()
        except Exception as e:
            logger.error("Error hashing %s: %s", filepath, e)
            return ""

    def scan_directory(
        self, base_dir: Path, expected_count: Optional[int] = None
    ) -> Dict[str, str]:
        """Scan a base directory and compute relative paths mapped to fast hashes.

        Args:
            base_dir: Root directory of the dataset.
            expected_count: Optional expected file count for progress bar.

        Returns:
            Mapping from relative posix path to SHA-256 hash.
        """
        path_to_hash: Dict[str, str] = {}
        if not base_dir.exists():
            logger.warning("Directory %s does not exist.", base_dir)
            return path_to_hash

        logger.info("Scanning and hashing files in %s...", base_dir)
        pbar = tqdm(desc="Scanning files", unit="file", total=expected_count)

        current_val = 0
        for filepath in base_dir.rglob("*"):
            if filepath.is_file():
                rel_path = filepath.relative_to(base_dir).as_posix()
                f_hash = self.compute_file_hash(str(filepath))
                if f_hash:
                    path_to_hash[rel_path] = f_hash

                if expected_count and current_val < expected_count:
                    pbar.update(1)
                    current_val += 1
                elif not expected_count:
                    pbar.update(1)

        pbar.close()
        return path_to_hash

    def align(
        self,
        manifest_path: str,
        data_dir: str,
        dry_run: bool = False,
        delete_extras: bool = False,
    ) -> Dict[str, Any]:
        """Align local dataset directory to match the target manifest structure.

        Args:
            manifest_path: Path to dataset_manifest.json.
            data_dir: Root directory of the dataset to verify and align.
            dry_run: If True, previews proposed file operations without writing to disk.
            delete_extras: If True, deletes files not present in the manifest.

        Returns:
            Summary dictionary with counts and itemized action lists.
        """
        manifest_file = Path(manifest_path).resolve()
        base_dir = Path(data_dir).resolve()

        if not manifest_file.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_file}")

        logger.info("Loading manifest: %s", manifest_file)
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest: Dict[str, str] = json.load(f)

        logger.info("Scanning directory: %s", base_dir)
        current_state = self.scan_directory(base_dir, expected_count=len(manifest))

        hash_to_current_paths: Dict[str, List[str]] = {}
        current_state_lower: Dict[str, str] = {}
        stem_to_current_paths_lower: Dict[str, str] = {}

        for path, f_hash in current_state.items():
            hash_to_current_paths.setdefault(f_hash, []).append(path)
            path_lower = path.lower()
            current_state_lower[path_lower] = path

            stem_lower = Path(path).with_suffix("").as_posix().lower()
            stem_to_current_paths_lower[stem_lower] = path

        to_delete: List[str] = []
        to_move: List[Tuple[str, str]] = []
        missing: List[str] = []
        format_mismatches: List[Tuple[str, str]] = []
        correct_count = 0

        processed_current_paths = set()
        handled_manifest_keys = set()
        manifest_keys = sorted(manifest.keys())

        # PASS 1: Lock in exact matches FIRST (same path and matching hash)
        for rel_path in tqdm(manifest_keys, desc="Locking exact matches", unit="file"):
            target_hash = manifest[rel_path]
            rel_path_lower = rel_path.lower()

            if rel_path_lower in current_state_lower:
                actual_on_disk = current_state_lower[rel_path_lower]
                if current_state[actual_on_disk] == target_hash:
                    correct_count += 1
                    processed_current_paths.add(actual_on_disk)
                    handled_manifest_keys.add(rel_path)

        # PASS 2: Handle moves, format mismatches, and missing files
        for rel_path in tqdm(manifest_keys, desc="Resolving discrepancies", unit="file"):
            if rel_path in handled_manifest_keys:
                continue

            target_hash = manifest[rel_path]
            target_stem_lower = Path(rel_path).with_suffix("").as_posix().lower()

            # Format mismatch check (e.g. extension case or slight container variant)
            if target_stem_lower in stem_to_current_paths_lower:
                found_path = stem_to_current_paths_lower[target_stem_lower]
                if found_path not in processed_current_paths:
                    format_mismatches.append((rel_path, found_path))
                    processed_current_paths.add(found_path)
                    continue

            # Move/relocate check based on identical content hash
            if target_hash in hash_to_current_paths:
                potential_srcs = [
                    p
                    for p in hash_to_current_paths[target_hash]
                    if p not in processed_current_paths
                ]
                if potential_srcs:
                    src = potential_srcs[0]
                    to_move.append((src, rel_path))
                    processed_current_paths.add(src)
                    continue

            missing.append(rel_path)

        # PASS 3: Extras not accounted for in manifest
        for path in tqdm(current_state.keys(), desc="Checking extra files", unit="file"):
            if path not in processed_current_paths:
                to_delete.append(path)

        summary = {
            "total_manifest": len(manifest),
            "correct": correct_count,
            "to_move_count": len(to_move),
            "to_move": to_move,
            "to_delete_count": len(to_delete),
            "to_delete": to_delete,
            "missing_count": len(missing),
            "missing": missing,
            "format_mismatches_count": len(format_mismatches),
            "format_mismatches": format_mismatches,
            "dry_run": dry_run,
        }

        logger.info(
            "Summary: Correct: %d, To Move: %d, To Delete: %d, Missing: %d, Format Mismatches: %d",
            correct_count,
            len(to_move),
            len(to_delete),
            len(missing),
            len(format_mismatches),
        )

        for expected, found in format_mismatches:
            logger.warning(
                "[FORMAT MISMATCH] Expected '%s' but found '%s'. Skipping to protect original file.",
                expected,
                found,
            )

        if dry_run:
            logger.info("Dry run enabled. No disk changes were applied.")
            for src, dest in to_move:
                logger.info("[DRY-RUN MOVE] %s -> %s", src, dest)
            if delete_extras:
                for path in to_delete:
                    logger.info("[DRY-RUN DELETE] %s", path)
            for path in missing:
                logger.warning("[DRY-RUN MISSING] %s", path)
            return summary

        # Execution of moves via safe temporary pool to prevent collisions
        if to_move:
            temp_dir = base_dir / ".align_temp"
            temp_dir.mkdir(exist_ok=True)

            temp_moves: List[Tuple[Path, Path]] = []
            for i, (src, dest) in enumerate(
                tqdm(to_move, desc="Relocating to safe temp pool", unit="file")
            ):
                src_full = base_dir / src
                temp_full = temp_dir / f"file_{i}.tmp"
                shutil.move(str(src_full), str(temp_full))
                temp_moves.append((temp_full, base_dir / dest))

            for temp_full, dest_full in tqdm(
                temp_moves, desc="Restoring canonical hierarchy", unit="file"
            ):
                dest_full.parent.mkdir(parents=True, exist_ok=True)
                if dest_full.exists():
                    if dest_full.is_dir():
                        shutil.rmtree(str(dest_full))
                    else:
                        dest_full.unlink()
                shutil.move(str(temp_full), str(dest_full))

            if temp_dir.exists():
                shutil.rmtree(str(temp_dir))

        # Optional deletion of untracked extras
        if delete_extras and to_delete:
            for path in tqdm(to_delete, desc="Deleting extra files", unit="file"):
                full_path = base_dir / path
                if full_path.exists():
                    if full_path.is_dir():
                        shutil.rmtree(str(full_path))
                    else:
                        full_path.unlink()

        logger.info("Dataset alignment completed successfully.")
        return summary
