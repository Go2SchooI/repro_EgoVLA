# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0

import os

import torch
from transformers import PretrainedConfig, SiglipImageProcessor
from transformers.utils import is_flash_attn_2_available

from llava.model.multimodal_encoder.vision_encoder import VisionTower, VisionTowerS2

from .siglip import SiglipVisionModel


def _get_siglip_attn_implementation() -> str:
    requested_impl = os.environ.get("EGO_VLA_SIGLIP_ATTN_IMPLEMENTATION", "flash_attention_2")
    if requested_impl != "flash_attention_2":
        return requested_impl

    if is_flash_attn_2_available():
        return requested_impl

    fallback_impl = os.environ.get("EGO_VLA_SIGLIP_ATTN_FALLBACK", "sdpa")
    print(
        f"FlashAttention2 is unavailable for SigLIP vision tower. Falling back to {fallback_impl}.",
        flush=True,
    )
    return fallback_impl


class SiglipVisionTower(VisionTower):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig) -> None:
        super().__init__(model_name_or_path, config)
        # TODO(ligengl): why pass config here leading to errors?
        self.vision_tower = SiglipVisionModel.from_pretrained(
            model_name_or_path,
            attn_implementation=_get_siglip_attn_implementation(),
            torch_dtype=eval(config.model_dtype),
        )
        self.image_processor = SiglipImageProcessor.from_pretrained(model_name_or_path)
        self.is_loaded = True


class SiglipVisionTowerS2(VisionTowerS2):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig) -> None:
        super().__init__(model_name_or_path, config)
        self.vision_tower = SiglipVisionModel.from_pretrained(
            model_name_or_path,
            attn_implementation=_get_siglip_attn_implementation(),
            torch_dtype=eval(config.model_dtype),
        )
        self.image_processor = SiglipImageProcessor.from_pretrained(model_name_or_path)
        # Make sure it crops/resizes the image to the largest scale in self.scales to maintain high-res information
        self.image_processor.size["height"] = self.image_processor.size["width"] = self.scales[-1]
        self.is_loaded = True
