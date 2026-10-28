import json
import re
from pathlib import Path
import logging
from tqdm import tqdm

from src.core.models import VideoInfo

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DatasetScanner:
    """Scans the dataset directory and yields VideoInfo entities."""

    def __init__(self, base_dir: str):
        self.expected_files = 629539
        # Ensure we resolve relative to the ACTUAL script execution location if path is relative
        self.base_dir = Path(base_dir).resolve()
        
        # Safety check: never allow writing to system root
        if str(self.base_dir) == '/':
             raise ValueError("DatasetScanner base_dir cannot be the system root (/).")

        # Ensure the base directory (usually 'data/') exists
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_dir = self.base_dir / "Gait_Dataset"
        self.cached_dataset_map_path = self.base_dir / "cached_dataset_map.json"
        self.cached_dataset_map = {}
        self._load()

    def _load(self):
        """Loads the Json dataset structure."""
        self.cached_dataset_map = {}
        if self.cached_dataset_map_path.exists():
            logger.info(f"Loading cached map from {self.cached_dataset_map_path}")
            with open(self.cached_dataset_map_path, 'r', encoding='utf-8') as f:
                self.cached_dataset_map = json.load(f)
        else:
            logger.info("No cache found. Starting full dataset extraction...")
            self._extract()
            self._save()

    def _save(self):
        """Saves the Json dataset structure."""
        with open(self.cached_dataset_map_path, 'w', encoding='utf-8') as f:
            json.dump(self.cached_dataset_map, f, indent=4)
        logger.info(f"Map successfully saved to {self.cached_dataset_map_path}")

    def _extract(self):
        """
        Scans the dataset directory and builds a nested dictionary (files_map)
        organizing files by Participant, Experiment, Test, and Camera Angle.
        Parses filenames using regex to identify videos, skeletons, and keypoint JSONs.
        """
        pbar = tqdm(total=self.expected_files, desc="Scanning Dataset", unit="file")
        self.cached_dataset_map = {}

        for file_path in self.dataset_dir.rglob('*'):
            if not file_path.is_file() or file_path.name.startswith('.'): continue

            pbar.update(1)

            filename = file_path.name
            if not filename.startswith('P'): continue       # Filter irrelevant files (e.g., Anthropometry.xlsx)

            # 1. Extract Participant, Experiment, Test and Relative Path
            match = re.match(r'P(\d+)_E(\d+)T(\d+)', filename)      # patterns like P1_E1T1
            if not match: continue

            is_alt = "_ALT" in filename
            p_key, e_key, t_num = f"P{match.group(1)}", f"E{match.group(2)}", match.group(3)
            t_key = f"T{t_num}ALT" if is_alt else f"T{t_num}"

            workspace_root = self.base_dir.parent
            rel_path = str(file_path.resolve().relative_to(workspace_root.resolve()))

            # Initialize the nested structure if it doesn't exist
            self.cached_dataset_map.setdefault(p_key, {}).setdefault(e_key, {}).setdefault(t_key, {})
            test_dict = self.cached_dataset_map[p_key][e_key][t_key]

            # 2. Handle the CSV file (sensor_data)
            if filename.endswith('.csv'):
                test_dict['sensor_data'] = rel_path
                continue

            # 3. Extract the angle (e.g., 45, 90, Front, Rear)
            angle_match = re.search(r'_(?:ALT_)?(\d+|front|Front|Rear|rear)(?:_|\.)', filename)
            if not angle_match: continue
            angle_key = angle_match.group(1)

            test_dict.setdefault(angle_key, {})
            angle_dict = test_dict[angle_key]

            # 4. Assign media files and JSONs using relative paths
            if filename.endswith('.mp4'): angle_dict['video'] = rel_path
            elif filename.endswith('_skeleton.avi'): angle_dict['skeleton_video'] = rel_path
            elif filename.endswith('_keypoints.json'):
                # Extract the frame (which is a 12-digit sequence)
                frame_match = re.search(r'_(\d{12})_keypoints\.json', filename)
                if frame_match:
                    frame_idx = int(frame_match.group(1))
                    angle_dict.setdefault('keypoints', {})
                    angle_dict['keypoints'][frame_idx] = rel_path
        pbar.close()
        logger.info("Dataset extraction completed.")

    def __iter__(self):
        """ Yields: VideoInfo: The next Video object in the dataset. """
        # Iterate over all participants
        for p_key, person in self.cached_dataset_map.items():
            p_num = int(p_key[1:])
            # Iterate over all experiments for the participant
            for e_key, experiment in person.items():
                e_num = int(e_key[1:])
                # Iterate over all tests
                for t_key, test_dict in experiment.items():
                    is_alt = t_key.endswith("ALT")
                    t_num = int(t_key[1:-3]) if is_alt else int(t_key[1:])
                    sensor_data = test_dict.get("sensor_data")
                    # Iterate over the keys in the test dictionary.
                    for a_key, video_data in test_dict.items():
                        if a_key == "sensor_data": continue

                        path = Path(video_data["video"])
                        sensor_position = self._get_sensor_pos(is_alt, t_num)

                        yield VideoInfo(
                            name = path.name,
                            video_path = path,
                            skeleton_path = Path(video_data.get("skeleton_video", path)),
                            keypoints_map = video_data.get("keypoints", {}),
                            sensor_data_path = sensor_data,
                            person = p_num,
                            experiment = e_num,
                            test = t_num,
                            angle = a_key,
                            is_alt = is_alt,
                            sensor_position = sensor_position,
                            interests_keypoints = self._get_interests_kpoints(sensor_position)
                        )

    def _get_sensor_pos(self, is_alt, test_number):
        if is_alt or test_number in [1, 2]: return "leg"
        if test_number in [3, 4]: return "arm"
        if test_number in [5, 6]: return "head"
        raise ValueError(f"Invalid test number: {test_number}")

    def _get_interests_kpoints(self, sensor_position):
        if sensor_position == "leg": return ["RHip", "RKnee", "RAnkle", "RHeel", "RBigToe", "RSmallToe"]
        if sensor_position == "arm": return ["RShoulder", "RElbow"]
        if sensor_position == "head": return ["Nose", "Neck", "REye", "REar"]
        raise ValueError(f"Invalid sensor position: {sensor_position}")

    def get_csvs(self):
        unique_csvs = set()
        for v in self:
            if v.sensor_data_path and v.sensor_data_path is not None:
                unique_csvs.add(v.sensor_data_path)
        return list(unique_csvs)

    def get_video_info(self, person_number, experiment_number, test_number, video_degree, alternative_clothing=False):
        """
        Retrieves a specific VideoInfo object based on the provided identifiers.
        Args:
            person_number (int): The participant ID (e.g., 1 for P1).
            experiment_number (int): The experiment ID (e.g., 1 for E1).
            test_number (int): The test ID (e.g., 1 for T1).
            video_degree (str/int): The camera angle (e.g., '45', '90', 'Front', 'Rear').
            alternative_clothing (bool): Whether to look for the ALT version of the test.
        Returns:
            VideoInfo: An instance of the VideoInfo class.
        """
        try:
            p_key = f"P{person_number}"
            e_key = f"E{experiment_number}"
            t_key = f"T{test_number}ALT" if alternative_clothing else f"T{test_number}"
            deg_key = str(video_degree)

            test_dict = self.cached_dataset_map[p_key][e_key][t_key]
            video_data = test_dict[deg_key]
            sensor_data = test_dict.get("sensor_data")
            v_path = Path(video_data["video"])

            sensor_pos = self._get_sensor_pos(alternative_clothing, test_number)

            return VideoInfo(
                name = v_path.name,
                video_path = v_path,
                skeleton_path = Path(video_data.get("skeleton_video", v_path)),
                keypoints_map = video_data.get("keypoints", {}),
                sensor_data_path = sensor_data,
                person = person_number,
                experiment = experiment_number,
                test = test_number,
                angle = deg_key,
                is_alt = alternative_clothing,
                sensor_position = sensor_pos,
                interests_keypoints = self._get_interests_kpoints(sensor_pos)
            )
        except KeyError as e:
            logger.error(f"Missing data in map for P{person_number} E{experiment_number} T{test_number}: {e}")
            raise
