# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from rlinf.models.embodiment.gr00t.simulation_io import (
    convert_to_robocasa365_action_n1d7,
)


def test_robocasa_conversion_binarizes_both_discrete_controls():
    action_chunk = {
        "base_motion": np.array(
            [[[-0.2, 0.1, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.8, -0.4, 0.1, 0.0]]],
            dtype=np.float32,
        ),
        "control_mode": np.array([[[-0.2], [0.0], [0.8]]], dtype=np.float32),
        "end_effector_position": np.array(
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]],
            dtype=np.float32,
        ),
        "end_effector_rotation": np.zeros((1, 3, 3), dtype=np.float32),
        "gripper_close": np.array([[[-0.2], [0.0], [0.8]]], dtype=np.float32),
    }

    converted = convert_to_robocasa365_action_n1d7(action_chunk, chunk_size=3)

    assert converted.shape == (1, 3, 12)
    np.testing.assert_array_equal(converted[0, :, 6], [-1.0, 1.0, 1.0])
    np.testing.assert_array_equal(converted[0, :, 11], [-1.0, 1.0, 1.0])
    np.testing.assert_allclose(converted[0, 0, :6], [1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(converted[0, 0, 7:11], [-0.2, 0.1, 0.0, 0.0])
