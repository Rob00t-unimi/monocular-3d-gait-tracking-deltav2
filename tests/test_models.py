import unittest
from src.core.models import ProcessingState

class TestModels(unittest.TestCase):
    def test_processing_state_update_fullbody(self):
        state = ProcessingState()
        state.update_fullbody(10, {"Nose": [0, 0]})
        self.assertEqual(state.ref_frame_idx, 10)
        self.assertEqual(state.fullbody_keypoints, {"Nose": [0, 0]})

    def test_processing_state_update_specific(self):
        state = ProcessingState()
        state.update_specific(20, [[1,2], [3,4], [5,6]])
        self.assertEqual(state.ref_frame_idx, 20)
        self.assertEqual(state.specific_keypoints, [[1,2], [3,4], [5,6]])

    def test_processing_state_to_dict(self):
        state = ProcessingState(
            ref_frame_idx=10,
            fullbody_keypoints={"Nose": [0, 0]},
            specific_keypoints=[[1,2], [3,4], [5,6]]
        )
        d = state.to_dict()
        self.assertEqual(d["ref_frame_idx"], 10)
        self.assertEqual(d["fullbody"]["keypoints"], {"Nose": [0, 0]})
        self.assertEqual(d["specific"]["keypoints"], [[1,2], [3,4], [5,6]])

    def test_processing_state_to_dict_discarded(self):
        state = ProcessingState(fullbody_discarded=True, specific_discarded=True)
        d = state.to_dict()
        self.assertTrue(d["fullbody"]["discarded"])
        self.assertTrue(d["specific"]["discarded"])

if __name__ == '__main__':
    unittest.main()
