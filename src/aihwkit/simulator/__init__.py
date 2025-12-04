# -*- coding: utf-8 -*-

# (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
#
# Licensed under the MIT license. See LICENSE file in the project root for details.

"""RPU simulator bindings."""

# This import is required in order to load the `torch` shared libraries, which
# the simulator shared library is linked against.
import torch

# Import rpu_base from system aihwkit
import importlib.util
import sys
import os

# Find system aihwkit's rpu_base
_system_aihwkit_path = "/usr/local/lib/python3.10/dist-packages/aihwkit/simulator"
_rpu_base_so = os.path.join(_system_aihwkit_path, "rpu_base.cpython-310-x86_64-linux-gnu.so")

if os.path.exists(_rpu_base_so):
    spec = importlib.util.spec_from_file_location("rpu_base", _rpu_base_so)
    rpu_base = importlib.util.module_from_spec(spec)
    sys.modules["aihwkit.simulator.rpu_base"] = rpu_base
    spec.loader.exec_module(rpu_base)
else:
    raise ImportError(f"Cannot find rpu_base at {_rpu_base_so}")
