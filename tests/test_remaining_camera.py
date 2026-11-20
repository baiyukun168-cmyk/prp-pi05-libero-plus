import json
from pathlib import Path
import unittest

from src.evaluation.camera_task_selection import select_camera


class RemainingCameraTest(unittest.TestCase):
    def test_selects_exact_remaining_populations(self):
        metadata=json.loads(Path("results/paired/task_classification.json").read_text(encoding="utf-8"))
        for suite,count in (("libero_spatial",376),("libero_object",396),("libero_goal",408)):
            selected=select_camera(metadata,suite)
            self.assertEqual(len(selected),count)
            self.assertEqual(len({row["task_id"] for row in selected}),count)
            for row in selected:
                original=metadata[suite][row["task_id"]]
                self.assertEqual(original["category"],"Camera Viewpoints")
                self.assertEqual(original["name"],row["official_name"])

    def test_rejects_metadata_with_wrong_id_order(self):
        metadata={"libero_goal":[{"id":2,"category":"Camera Viewpoints","name":"task","difficulty_level":1}]}
        with self.assertRaises(ValueError):
            select_camera(metadata,"libero_goal")


if __name__=="__main__":
    unittest.main()
