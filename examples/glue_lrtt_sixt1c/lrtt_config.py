# -*- coding: utf-8 -*-
"""LRTT configuration helper for GLUE tasks."""

import sys
import os

# Add LRTT src to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LRTT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, os.path.join(LRTT_ROOT, "src"))

from aihwkit.simulator.configs.lrtt_config import (
    lrtt_sixt1c_ab_ideal_config,
    lrtt_idealized_config,
    PythonLRTTRPUConfig,
)


def get_glue_preset_config(
    rank: int = 8,
    transfer_every: int = 1000,
    lora_alpha: float = 32.0,
    use_ideal: bool = False,
) -> PythonLRTTRPUConfig:
    """Get LRTT configuration preset for GLUE tasks.

    Args:
        rank: LRTT rank (similar to LoRA rank)
        transfer_every: Transfer frequency in steps
        lora_alpha: LoRA scaling factor
        use_ideal: If True, use fully idealized config

    Returns:
        Configured PythonLRTTRPUConfig
    """
    if use_ideal:
        return lrtt_idealized_config(
            rank=rank,
            transfer_every=transfer_every,
            lora_alpha=lora_alpha,
        )
    else:
        return lrtt_sixt1c_ab_ideal_config(
            rank=rank,
            transfer_every=transfer_every,
            lora_alpha=lora_alpha,
        )
