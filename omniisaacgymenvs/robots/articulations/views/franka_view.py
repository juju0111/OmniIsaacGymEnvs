from typing import Optional

from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.prims.xform_prim_view import XFormPrimView


class FrankaView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "FrankaView",
        custom_path_name:str = None, 
    ) -> None:
        """[summary]"""

        super().__init__(prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False)

        if custom_path_name==None:
            default_name = "/World/envs/.*/franka"
        else:
            default_name = "/World/envs/.*/franka/" + custom_path_name
            
        self._hands = RigidPrimView(
            prim_paths_expr= default_name + "/panda_link7", name="hands_view", reset_xform_properties=False
        )
        self._lfingers = RigidPrimView(
            prim_paths_expr= default_name + "/panda_leftfinger", name="lfingers_view", reset_xform_properties=False
        )
        self._rfingers = RigidPrimView(
            prim_paths_expr= default_name + "/panda_rightfinger",
            name="rfingers_view",
            reset_xform_properties=False,
        )

        self._camera = XFormPrimView(
            prim_paths_expr= default_name + "/panda_hand/Depth",
            name="camera_view",
            reset_xform_properties=False,
        )

    def initialize(self, physics_sim_view):
        super().initialize(physics_sim_view)

        self._gripper_indices = [self.get_dof_index("panda_finger_joint1"), self.get_dof_index("panda_finger_joint2")]

    @property
    def gripper_indices(self):
        return self._gripper_indices
