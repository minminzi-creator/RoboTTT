# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Read-only container / table indexing audit for all three training cameras."""

import argparse
import json
import subprocess
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

from rlinf.models.embodiment.gr00t.gr00t_n1d7.sequence_utils import (
    dataset_settings,
    load_sft_config,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = dataset_settings(load_sft_config(args.config))
    root = Path(config["data"]["path"])
    info = json.loads((root / "meta/info.json").read_text())
    modality = json.loads((root / "meta/modality.json").read_text())
    episodes = [
        json.loads(line)
        for line in (root / "meta/episodes.jsonl").read_text().splitlines()
    ]
    cameras = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]
    assert all(camera in modality["video"] for camera in cameras)
    assert info["fps"] == 20 and len(episodes) == 104
    records = []
    max_timestamp_error = 0.0
    for episode in episodes:
        index, length = episode["episode_index"], episode["length"]
        keys = {"episode_chunk": index // info["chunks_size"], "episode_index": index}
        table = pd.read_parquet(
            root / info["data_path"].format(**keys),
            columns=["frame_index", "timestamp"],
        )
        assert len(table) == length
        assert np.array_equal(table["frame_index"].to_numpy(), np.arange(length))
        timestamps = table["timestamp"].to_numpy().astype(np.float64)
        assert np.isfinite(timestamps).all()
        error = float(np.max(np.abs(timestamps - np.arange(length) / 20)))
        assert error <= 1e-5, (index, error)
        max_timestamp_error = max(max_timestamp_error, error)
        for camera in cameras:
            original = modality["video"][camera]["original_key"]
            path = root / info["video_path"].format(video_key=original, **keys)
            data = json.loads(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=nb_frames,avg_frame_rate,r_frame_rate,start_time,duration,width,height,time_base",
                        "-of",
                        "json",
                        str(path),
                    ],
                    text=True,
                    timeout=30,
                )
            )
            assert len(data["streams"]) == 1
            stream = data["streams"][0]
            assert int(stream["nb_frames"]) == length, (index, camera, stream)
            assert (
                Fraction(stream["avg_frame_rate"])
                == Fraction(stream["r_frame_rate"])
                == 20
            )
            assert float(stream["start_time"]) == 0.0
            assert abs(float(stream["duration"]) - length / 20) <= 1e-6
            assert (stream["width"], stream["height"]) == (256, 256)
            records.append(
                {
                    "episode": index,
                    "camera": camera,
                    "file": str(path.relative_to(root)),
                    "frames": length,
                    "stream": stream,
                    "table_timestamp_max_error_seconds": error,
                }
            )
    result = {
        "status": "passed",
        "episodes": len(episodes),
        "videos": len(records),
        "fps": info["fps"],
        "camera_order": cameras,
        "camera_keys_match_directly_without_positional_fallback": True,
        "frame_indices_zero_based_and_contiguous": True,
        "all_camera_frame_counts_equal_table_rows": True,
        "max_table_timestamp_error_seconds": max_timestamp_error,
        "scope": "Container and table index alignment; not an independent sensor synchronization measurement",
        "records": records,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({key: value for key, value in result.items() if key != "records"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
