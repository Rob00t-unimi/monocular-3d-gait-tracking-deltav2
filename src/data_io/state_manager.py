"""
Module for persisting processing progress to disk.
Implements IStateRepository protocol.
"""

import json
import logging
from pathlib import Path
from src.core.interfaces import IStateRepository
from src.core.models import ProcessingState

logger = logging.getLogger(__name__)

class JsonStateRepository(IStateRepository):
    """
    Saves and loads processing state as a JSON file.
    
    Attributes:
        filepath: Path to the JSON state file.
        _cache: In-memory copy of the entire states dictionary.
    """

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._cache = self._load_full_cache()

    def _load_full_cache(self) -> dict:
        """Loads the whole JSON file into memory."""
        if not self.filepath.exists():
            return {}
        try:
            with open(self.filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load state file {self.filepath}: {e}")
            return {}

    def load(self, name: str) -> ProcessingState:
        """
        Retrieves the state for a specific video.
        
        Args:
            name: Video name identifier.
            
        Returns:
            A ProcessingState object (fresh or populated).
        """
        data = self._cache.get(name, {})
        state = ProcessingState()
        
        if data:
            state.ref_frame_idx = data.get("ref_frame_idx")
            
            # Map fullbody
            fb = data.get("fullbody", {})
            state.fullbody_discarded = fb.get("discarded", False)
            state.fullbody_keypoints = fb.get("keypoints", {})
            
            # Map specific
            sp = data.get("specific", {})
            state.specific_discarded = sp.get("discarded", False)
            state.specific_keypoints = sp.get("keypoints", [])
            
        return state

    def save(self, name: str, state: ProcessingState) -> None:
        """
        Saves the state of a video and flushes cache to disk.
        
        Args:
            name: Video name identifier.
            state: The current state object.
        """
        self._cache[name] = state.to_dict()
        try:
            with open(self.filepath, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save state to {self.filepath}: {e}")

# Default shared instance for the project
videos_state_manager = JsonStateRepository(Path("data/videos_state.json"))
