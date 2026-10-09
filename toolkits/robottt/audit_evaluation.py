# Copyright 2026 The RLinf Authors. SPDX-License-Identifier: Apache-2.0
"""Verify native rollout memory records and fully decode the framework's videos."""

import argparse
import json
import subprocess
from pathlib import Path


def audit(run: Path, videos: Path, use_ttt: bool, episodes: int, steps: int) -> dict:
    assert steps % 8 == 0
    calls_per_episode = steps // 8
    resets = {}
    records = []
    for rank in range(8):
        path = run / f"rollout_state_rank{rank}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == episodes * calls_per_episode, (rank, len(rows))
        resets[rank] = []
        for index, row in enumerate(rows):
            offset = index % calls_per_episode
            reset = offset == 0
            assert row["mode"] == "eval" and row["stage_id"] == 0
            assert row["dit_forwards"] == (5 if use_ttt else 4)
            assert row["commits"] == int(use_ttt)
            assert row["reset_mask"] == [reset]
            assert row["native_action_shape"] == [1, 8, 12]
            assert row["native_actions_finite"]
            expected_incoming = None
            if use_ttt and index:
                expected_incoming = [offset if offset else calls_per_episode]
            assert row["incoming_position"] == expected_incoming
            assert row["committed_position"] == ([offset + 1] if use_ttt else None)
            if reset:
                assert len(row["reset_observation_sha256"]) == 64
                resets[rank].append(row["reset_observation_sha256"])
        records.append({"rank": rank, "calls": len(rows), "resets": episodes})
    files = sorted(videos.rglob("*.mp4"))
    assert len(files) == 8 * episodes, len(files)
    decoded = []
    for path in files:
        probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_read_frames,width,height",
                    "-of",
                    "json",
                    str(path),
                ],
                text=True,
                timeout=60,
            )
        )["streams"][0]
        frames = int(probe["nb_read_frames"])
        # RecordVideo includes the initial reset observation when available.
        assert frames in (steps, steps + 1), (path, frames)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"],
            check=True,
            capture_output=True,
            timeout=60,
        )
        decoded.append(
            {
                "file": str(path),
                "frames": frames,
                "width": probe["width"],
                "height": probe["height"],
            }
        )
    return {
        "passed": True,
        "run": str(run),
        "use_ttt": use_ttt,
        "ranks": 8,
        "episodes": 8 * episodes,
        "environment_steps_per_episode": steps,
        "reset_observation_sha256": resets,
        "workers": records,
        "videos": decoded,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--videos", type=Path, required=True)
    parser.add_argument("--use-ttt", action="store_true")
    parser.add_argument("--episodes", type=int, default=2, help="Episodes per rank")
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--paired-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run, args.videos, args.use_ttt, args.episodes, args.steps)
    if args.paired_report:
        other = json.loads(args.paired_report.read_text())
        # JSON converts integer rank keys to strings.
        current = {str(k): v for k, v in result["reset_observation_sha256"].items()}
        result["paired_reset_observations_equal"] = (
            current == other["reset_observation_sha256"]
        )
        assert result["paired_reset_observations_equal"], (
            "Paired reset observations differ"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in ("videos", "reset_observation_sha256", "workers")
            }
        )
    )


if __name__ == "__main__":
    main()
