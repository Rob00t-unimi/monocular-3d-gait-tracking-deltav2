"""
Orchestrator for the Ground Truth (IMU) processing pipeline.
Coordinates loading, frequency estimation, filtering, and validation.
"""

import logging
import os
import json
from pathlib import Path
from typing import List, Set, Tuple, Optional
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src.core.interfaces import ISignalLoader, ISignalProcessor, ISignalValidator, IFrequencyEstimator
from src.data_io.csv_loader import CsvSignalLoader
from src.preprocessing.signals.frequency_analysis import GaitFrequencyEstimator
from src.preprocessing.signals.filtering import ButterworthSignalProcessor
from src.preprocessing.signals.validation import PhysiologicalSignalValidator

logger = logging.getLogger(__name__)

class SignalOrchestrator:
    """
    Main entry point for processing all IMU signals in the dataset.
    
    Attributes:
        cache_dir: Directory where processed CSVs and metadata are stored.
    """

    def __init__(self, cache_dir: str = "data/gt_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Dependency Injection (Default implementations)
        self.loader: ISignalLoader = CsvSignalLoader()
        self.estimator: IFrequencyEstimator = GaitFrequencyEstimator(self.loader)
        self.processor: ISignalProcessor = ButterworthSignalProcessor()
        self.validator: ISignalValidator = PhysiologicalSignalValidator()

        self.invalid_cache_path = self.cache_dir / "invalid_csvs.json"
        self.cutoff_cache_path = self.cache_dir / "cutoff_freqs.json"

    def run_pipeline(self, scanner, batch_size: int = 50, force_recalc: bool = False) -> Tuple[Set[str], Tuple[float, float]]:
        """
        Runs the full GT processing suite.
        
        Args:
            scanner: DatasetScanner instance.
            batch_size: Number of samples for frequency profiling.
            force_recalc: If True, ignores all caches and re-processes everything.
            
        Returns:
            Tuple of (set of invalid CSV paths, (lowcut, highcut) frequencies).
        """
        logger.info(f"Starting Ground Truth pipeline (Force={force_recalc})...")
        unique_csvs = scanner.get_csvs()
        
        # 1. Frequency Analysis
        lowcut, highcut = self._get_frequency_bounds(unique_csvs, batch_size, force_recalc)
        
        # 2. Batch Processing
        invalid_csvs = self._process_all_files(unique_csvs, lowcut, highcut, force_recalc)
        
        # 3. Persistence
        self._save_invalid_cache(invalid_csvs)
        
        return invalid_csvs, (lowcut, highcut)

    def _get_frequency_bounds(self, csv_paths: List[str], batch_size: int, force_recalc: bool) -> Tuple[float, float]:
        """Handles caching logic for frequency estimation."""
        if not force_recalc and self.cutoff_cache_path.exists():
            try:
                with open(self.cutoff_cache_path, 'r') as f:
                    data = json.load(f)
                    return data["lowcut"], data["highcut"]
            except Exception:
                logger.warning("Corrupted frequency cache, recalculating...")

        sample_paths = [Path(p) for p in csv_paths[:batch_size]]
        low, high = self.estimator.estimate(sample_paths)
        
        with open(self.cutoff_cache_path, 'w') as f:
            json.dump({"lowcut": low, "highcut": high}, f)
            
        return low, high

    def _process_all_files(self, csv_paths: List[str], lowcut: float, highcut: float, force_recalc: bool) -> Set[str]:
        """Iterates over all CSVs and applies the processing chain."""
        # Load existing invalid cache if available to avoid full re-run (only if not forcing)
        if not force_recalc and self.invalid_cache_path.exists():
            try:
                with open(self.invalid_cache_path, 'r') as f:
                    logger.info("IMU Signals already processed. Loading from cache...")
                    return set(json.load(f))
            except Exception:
                pass

        invalid = set()
        
        with logging_redirect_tqdm():
            for path_str in tqdm(csv_paths, desc="Filtering IMU", unit="csv"):
                if not self._process_single_file(path_str, lowcut, highcut):
                    invalid.add(path_str)
                    
        return invalid

    def _process_single_file(self, path_str: str, lowcut: float, highcut: float) -> bool:
        """Loads, filters, validates and saves a single signal file."""
        input_path = Path(path_str)
        output_path = self.cache_dir / (input_path.name + "_processed.csv")
        
        df = self.loader.load(input_path)
        if df is None: return False
        
        df_filtered = self.processor.process(df, lowcut, highcut)
        
        if not self.validator.is_valid(df_filtered):
            return False
            
        df_filtered.to_csv(output_path, index=False)
        return True

    def _save_invalid_cache(self, invalid_set: Set[str]):
        """Persists the list of invalid files to avoid re-processing."""
        with open(self.invalid_cache_path, 'w') as f:
            json.dump(list(invalid_set), f)
