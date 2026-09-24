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

from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from rlinf.models.embodiment.gr00t import gr00t_n1d7


def _runtime_model(
    *,
    action_dim=12,
    valid_action_dim=12,
    image_nums=3,
    head_action_dim=12,
    head_valid_action_dim=12,
):
    return SimpleNamespace(
        action_dim=action_dim,
        valid_action_dim=valid_action_dim,
        image_nums=image_nums,
        action_head=SimpleNamespace(
            env_action_dim=head_action_dim,
            valid_action_dim=head_valid_action_dim,
        ),
    )


def _shape_cfg():
    return OmegaConf.create(
        {
            "action_dim": 12,
            "valid_action_dim": 12,
            "image_nums": 3,
        }
    )


def test_runtime_shape_audit_accepts_aligned_model_and_action_head():
    assert hasattr(gr00t_n1d7, "_audit_runtime_model_shapes")
    observed = gr00t_n1d7._audit_runtime_model_shapes(
        _runtime_model(), _shape_cfg()
    )

    assert observed == {
        "action_dim": 12,
        "valid_action_dim": 12,
        "image_nums": 3,
    }


@pytest.mark.parametrize(
    ("overrides", "mismatch_name"),
    [
        ({"action_dim": 7}, "model.action_dim"),
        ({"valid_action_dim": 7}, "model.valid_action_dim"),
        ({"image_nums": 2}, "model.image_nums"),
        ({"head_action_dim": 7}, "action_head.env_action_dim"),
        ({"head_valid_action_dim": 7}, "action_head.valid_action_dim"),
    ],
)
def test_runtime_shape_audit_rejects_every_model_or_head_mismatch(
    overrides, mismatch_name
):
    with pytest.raises(ValueError, match=mismatch_name):
        gr00t_n1d7._audit_runtime_model_shapes(
            _runtime_model(**overrides), _shape_cfg()
        )
