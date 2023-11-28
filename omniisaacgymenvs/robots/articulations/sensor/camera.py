# Copyright (c) 2021-2023, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# from omni.isaac.kit import SimulationApp 
# simulation_app = SimulationApp({"headless": True})


import copy
import math
from typing import Callable, List, Optional, Sequence, Tuple

import carb
import numpy as np
import omni
import omni.replicator.core as rep
from omni.isaac.core.prims.base_sensor import BaseSensor
from omni.isaac.core.utils.carb import get_carb_setting
from omni.isaac.core.utils.prims import (
    define_prim,
    get_all_matching_child_prims,
    get_prim_at_path,
    get_prim_path,
    get_prim_type_name,
    is_prim_path_valid,
)

from omni.isaac.core.utils.render_product import get_resolution, set_camera_prim_path, set_resolution
from omni.isaac.core_nodes.bindings import _omni_isaac_core_nodes
from pxr import Sdf, Usd, UsdGeom, Vt

if __name__=="__main__":
    print(1)