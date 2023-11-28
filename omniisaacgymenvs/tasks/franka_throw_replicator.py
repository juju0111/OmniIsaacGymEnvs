# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

# RAI Lab RL push task 
import math
import glob

import random 
import numpy as np
import torch
from omni.isaac.cloner import Cloner
from omni.isaac.core.objects import DynamicCuboid
from omni.isaac.core.objects import DynamicCustomObject
from omni.isaac.core.prims import RigidPrim, RigidPrimView, XFormPrim
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.utils.torch.transformations import *
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.franka import Franka
from omniisaacgymenvs.robots.articulations.views.franka_view import FrankaView
from pxr import Usd, UsdGeom, PhysxSchema
import omni.usd
from gym import spaces

from omni.isaac.core.utils.bounds import compute_aabb, compute_obb, create_bbox_cache, get_obb_corners
from omni.isaac.core.utils.numpy.rotations import quats_to_rot_matrices, euler_angles_to_quats, quats_to_euler_angles
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.core.utils.prims import find_matching_prim_paths

from PIL import Image

# from omni.isaac.core.prims.base_sensor import BaseSensor

class FrankaRAIThrow_RepTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)

        self.distX_offset = 0.04
        self.dt = 1 / 60.0

        self._num_observations = 23
        self._num_actions = 9

        # use multi-dimensional observation for pointclod from camera (2.5 D)
        # High Dim 
        self.observation_space = spaces.Box(
            np.ones((2048, 3), dtype=np.float32) * -np.Inf, 
            np.ones((2048, 3), dtype=np.float32) * np.Inf)
        # self.observation_space = spaces.Box(
        #     np.ones((100, 3), dtype=np.float32) * -np.Inf, 
        #     np.ones((100, 3), dtype=np.float32) * np.Inf)

        # import DR extensions
        enable_extension("omni.replicator.isaac")
        enable_extension("omni.isaac.sensor")
        import omni.replicator.core as rep
        import omni.replicator.isaac as dr
        from omni.isaac.sensor import Camera_light

        self.camera = Camera_light
        self.t=  0

        self.nucleus_server = get_assets_root_path()
        self.asset_folder = self.nucleus_server + "/Isaac/Samples/Examples/FrankaNutBolt/"
        
        self._table_position = np.array([0.5, 0.0, 0.005])  # Gf.Vec3f(0.5, 0.0, 0.0)
        self._table_scale = 0.0125
        self._franka_position = np.array([0.15, 0.0, 1.03330328])  # Gf.Vec3f(0.269, 0.1778, 0.0)

        RLTask.__init__(self, name, env)

        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self._device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self._device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self._device).repeat((self.num_envs, 1))
        self.goal_in_buf = torch.zeros_like(self.reset_buf)

    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        self._max_episode_length = self._task_cfg["env"]["episodeLength"]

        self.action_scale = self._task_cfg["env"]["actionScale"]
        self.start_position_noise = self._task_cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self._task_cfg["env"]["startRotationNoise"]
        self.num_props = self._task_cfg["env"]["numProps"]

        self.dof_vel_scale = self._task_cfg["env"]["dofVelocityScale"]
        self.dof_torque_xyz_scale = self._task_cfg["env"]["dofTorqueScale_xyz"]
        self.dof_torque_abg_scale = self._task_cfg["env"]["dofTorqueScale_abg"]            
        self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self._task_cfg["env"]["rotRewardScale"]
        self.around_handle_reward_scale = self._task_cfg["env"]["aroundHandleRewardScale"]
        self.open_reward_scale = self._task_cfg["env"]["openRewardScale"]
        self.finger_dist_reward_scale = self._task_cfg["env"]["fingerDistRewardScale"]
        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"]
        self.finger_close_reward_scale = self._task_cfg["env"]["fingerCloseRewardScale"]

        self.camera_width = self._task_cfg["env"]["cameraWidth"]
        self.camera_height = self._task_cfg["env"]["cameraHeight"]

    def cleanup(self) -> None:
        # initialize remaining buffers
        RLTask.cleanup(self)

        # 우리는 point cloud 쓸꺼니까 각 env부터 받아올 obs_buf 아래와 같이 초기화 함. 
        # override observation buffer for camera data
        self.obs_buf = torch.zeros(
            (self.num_envs, 20000, 3), device=self.device, dtype=torch.float)

    def set_up_scene(self, scene) -> None:
        ## Tabel, Robot 추가
        self.get_table()
        self.get_franka()

        # 환경 Clone 실행 
        super().set_up_scene(scene, filter_collisions=False)


        self._frankas = FrankaView(prim_paths_expr="/World/envs/.*/franka/panda", name="franka_view", custom_path_name="panda")
        self._tables = RigidPrimView(
            prim_paths_expr="/World/envs/.*/table", name="table_view", reset_xform_properties=False
        )
    
        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._camera)
        scene.add(self._frankas._lfingers)
        scene.add(self._frankas._rfingers)
        scene.add(self._tables)

        cam_paths = find_matching_prim_paths("/World/envs/.*/franka/panda/panda_hand/Depth")

        ################
        # initialize pytorch writer for vectorized collection
        # 카메라가 torch vectorized collection으로 받아와야함. 
        # set up cameras
        self.rep.orchestrator._orchestrator._is_started = True

        self.render_products = []
        env_pos = self._env_pos.cpu()
        self.Depth = [] 
        for i in range(self._num_envs):
            self.Depth.append(self.camera(
                prim_path=cam_paths[i],
                resolution=(self.camera_width, self.camera_height)
            ))

            self.Depth[i].create_render_product()
            self.Depth[i].set_camera_param()
            
            self.render_products.append(self.Depth[i]._render_product)

        self.pytorch_listener = self.PytorchListener()
        self.pytorch_writer = self.rep.WriterRegistry.get("PytorchWriter")
        print("Writer : ",self.pytorch_writer)
        self.pytorch_writer.initialize(listener=self.pytorch_listener, device="cuda", depth=True ) # , is_annot=True)
        self.pytorch_writer.attach(self.render_products)
        
        # Object, Goal bin 선언 
        self.get_object()
        self.get_goal_bin()

        self._objects = RigidPrimView(
            prim_paths_expr="/World/envs/.*/object", name="object_view", reset_xform_properties=False
        )
        self._bins = RigidPrimView(
            prim_paths_expr="/World/envs/.*/bin", name="bin_view", reset_xform_properties=False
        )
        
        scene.add(self._objects)
        scene.add(self._bins)

        self.init_data()
        return

    def initialize_views(self, scene):
        super().initialize_views(scene)(self, scene)
        
        if scene.object_exists("franka_view"):
            scene.remove_object("franka_view", registry_only=True)
        if scene.object_exists("hands_view"):
            scene.remove_object("hands_view", registry_only=True)
        if scene.object_exists("lfingers_view"):
            scene.remove_object("lfingers_view", registry_only=True)
        if scene.object_exists("rfingers_view"):
            scene.remove_object("rfingers_view", registry_only=True)

        self._frankas = FrankaView(prim_paths_expr="/World/envs/.*/franka", name="franka_view")

        self._tables = RigidPrimView(
            prim_paths_expr="/World/envs/.*/table", name="table_view", reset_xform_properties=False
        )
        self._objects = RigidPrimView(
            prim_paths_expr="/World/envs/.*/object", name="object_view", reset_xform_properties=False
        )
        self._bins = RigidPrimView(
            prim_paths_expr="/World/envs/.*/bin", name="bin_view", reset_xform_properties=False
        )
        
        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._camera)
        scene.add(self._frankas._lfingers)
        scene.add(self._frankas._rfingers)
        scene.add(self._tables)
        scene.add(self._objects)
        scene.add(self._bins)

        self.init_data()

    def get_franka(self):
        franka = Franka(
            prim_path=self.default_zero_env_path + "/franka", 
            name="franka",
            usd_path= "/home/juju/Downloads/panda_with_cam_d435.usd",
            custom_prim_name="panda", 
            translation=self._franka_position,
            orientation=torch.tensor([1.0, 0.0, 0.0, 0.0])
        )
        self._sim_config.apply_articulation_settings(
            "franka", get_prim_at_path(franka.prim_path + "/panda"), self._sim_config.parse_actor_config("franka")
        )

    def get_table(self):
        table_path = self.asset_folder + "SubUSDs/Shop_Table/Shop_Table.usd"

        table = DynamicCustomObject(
            prim_path= self.default_zero_env_path + "/table",
            name = "table",
            usd_path=table_path,
            scale=np.array([self._table_scale]),
            translation=self._table_position,
            mass=5000,
            color=np.array([0.1,0.1,0.1])
        )

        self._sim_config.apply_articulation_settings(
            "table", get_prim_at_path(table.prim_path), self._sim_config.parse_actor_config("table")
        )

        bb_cache = create_bbox_cache()
        combined_range_arr = compute_aabb(bb_cache, table.prim_path)
        print("AABB : ", combined_range_arr)
        self._table_height = combined_range_arr[-1]
        self._table_bbox = combined_range_arr
        _table_x = self._table_bbox[3] - self._table_bbox[0]
        _table_y = self._table_bbox[4] - self._table_bbox[1]
        _table_z = self._table_bbox[5] - self._table_bbox[2]
        self._table_xyz = [_table_x, _table_y, _table_z]
        self._franka_position[2] = self._table_height + 0.005

        # apply rigid body API and schema # 중력 적용 x 
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(table.prim_path)
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.GetDisableGravityAttr().Set(True)
        rb.GetRetainAccelerationsAttr().Set(True)
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        attr = prim.GetAttribute("physxRigidBody:disableGravity")
        print("Table gravity info : ",attr)

    def get_object(self):
        self.rep.orchestrator._orchestrator._is_started = True

        # from ShapeNet core
        self.throw_asset_1 = self._find_usd_assets(root = "/home/juju/juhan_ws/shapenet_nomat/for_throw", categories=["good_throw"], max_asset_size=None, split=1)
        
        """
        # spheres = self.rep.create.sphere(count=10, scale=self.rep.distribution.uniform(1., 3.))
        # print("Spheres :", spheres)
        # with self.rep.trigger.on_custom_event(event_name="Randomize!"):
        #     with spheres:
        #         mod = self.rep.modify.pose(position=self.rep.distribution.uniform((-50., -50., -50.), (50., 50., 50.)))
        # # Send event
        # self.rep.utils.send_og_event("Randomize!")

        # replicator 말고 기존 불러오는 방식으로 선언 시 했을 때 시행착오 
        # 각 환경마다 table위에 다른 object를 올리고 싶었으나, 실패. -> 가장 처음 올라온 object로 무지성 clone 되버림. 
        """
        print("env pose : ", self._env_pos)
        # sampled_obj = random.sample(self.throw_asset_1["good_throw"],1)[0]
        sampled_obj = self.throw_asset_1["good_throw"][-5]
        print("Sampled_obj usd path : ", sampled_obj)
        object_ = DynamicCustomObject(
            prim_path= self.default_zero_env_path + "/object",
            name = "object",
            usd_path = sampled_obj,
            translation= self._franka_position + np.array([0.4, 0., 0.05]),
            # translation= np.array([0, -1, 1.5]),
            orientation= euler_angles_to_quats(euler_angles=np.array([90,0,0]),degrees=True),
            scale = np.array([0.1]),
            mass=0.3,
            color=np.array([0., 1.0, 0.])
        )

        # object의 prim_path 있으면 그 object의 world 기준 min_x,y,z max_x,y,z 측정 가능 (trimesh의 mesh의 bounds 기능과 동일)
        # 처음에 막 선언 후 수정가능 
        bb_cache = create_bbox_cache()
        combined_range_arr = compute_aabb(bb_cache, object_.prim_path)
        can_height = combined_range_arr[5] - combined_range_arr[2] # height 
        obj_pos, _ = object_.get_world_pose()
        obj_pos[2] = self._table_height + can_height/2 + 0.005
        object_.set_world_pose(position=obj_pos)
        
        self._sim_config.apply_articulation_settings(
            "object", get_prim_at_path(object_.prim_path), self._sim_config.parse_actor_config("object")
        )

        """
        ###########
        # 아래쪽은 replicator 사용 시 시행착오 
        ### replicator에 object 등록시켜두면 segmentation도 되고 얻을 수 있는 정보 아주 많으나,
        ### Dynamic simulation이 안돼 -> 미쳐버림 . 기존 simulation랑 replicator simulation이랑 time_step_per_second가 다르게 설정 된 거 같음. 
        #### 추후 개발 더 되고  사용 가능할 것으로 추정... 

        # obj_reps = []
        # for i in range(self._num_envs):
        #     obj_rep = self.rep.create.from_usd(sampled_obj, semantics=[("class", "object")], count=1, obj_category="object")
        #     obj_reps.append(obj_rep)

        # obj_prim_path = "/Replicator/object/.*"
        # obj_prim_paths = find_matching_prim_paths(obj_prim_path)
        # print("object_prim_paths : ",obj_prim_paths)

        # # Init setting!
        # for i, obj_rep in enumerate(obj_reps):
        #     with obj_rep:
        #         self.rep.physics.mass(mass = self.rep.distribution.uniform((0.075),(0.6)),
        #                               density = 100,
        #                               center_of_mass = None)
        #         self.rep.physics.physics_material(
        #             static_friction = self.rep.distribution.uniform((0.05),(0.2)),
        #             dynamic_friction = self.rep.distribution.uniform((0.85),(0.98)),
        #             restitution = self.rep.distribution.uniform((0.05),(0.15))
        #         )
        #         self.rep.modify._scale(scale = 0.1)
        #         self.rep.modify._rotation(self.rep.distribution.uniform((90, 0, 0), (90, 0, 0)), "XYZ")
        #         self.rep.modify._position(self.rep.distribution.uniform((-1, -1, 0),(1,1,1)))
                
        #     self._sim_config.apply_articulation_settings(
        #         f"object_rep{i}", get_prim_at_path(obj_prim_paths[i]), self._sim_config.parse_actor_config("object")
        #     )
            
        # combined_range_arr = compute_aabb(bb_cache, obj_prim_paths[0])
        # print("Combined_range_arr", combined_range_arr)
        # self._sim_config.apply_articulation_settings(
        #     "object_rep", get_prim_at_path(object_.prim_path), self._sim_config.parse_actor_config("object")
        # )
        """
    def get_goal_bin(self):
        self.basket_asset = self._find_usd_assets(root = "/home/juju/juhan_ws/shapenet_nomat/for_throw", categories=["good_trash_bin"],max_asset_size=None, split=1)
        
        sampled_obj = self.basket_asset["good_trash_bin"][-2]
        print("Sampled_bin usd path : ", sampled_obj)

        self.goal_start_translation = np.array([1, 0, 1])
        goal_ = DynamicCustomObject(
            prim_path=self.default_zero_env_path + "/bin",
            usd_path = sampled_obj,
            translation=self.goal_start_translation,
            orientation= euler_angles_to_quats(euler_angles=np.array([90,0,0]),degrees=True),
            scale= np.array([1.]),
            color=np.array([0.2, 0.2, 0.5])
        )

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(goal_.prim_path)
        # bin 위치 수정하기 
        bb_cache = create_bbox_cache()
        goal_object_world_bbox = compute_aabb(bb_cache, goal_.prim_path)
        goal_object_x = goal_object_world_bbox[3] - goal_object_world_bbox[0] 
        goal_object_y = goal_object_world_bbox[4] - goal_object_world_bbox[1] 
        goal_object_z = goal_object_world_bbox[5] - goal_object_world_bbox[2] 
        
        self.bin_xyz = [goal_object_x, goal_object_y, goal_object_z]

        goal_pos, _ = goal_.get_world_pose()
        goal_pos[0] = goal_pos[0] + goal_object_x/2 + 0.05 # (0.05 : offset)  
        goal_pos[1] = goal_pos[1] + (self._table_xyz[1])/2 + goal_object_y/2 + 0.05 
        goal_pos[2] = self._table_height
        goal_.set_world_pose(position=goal_pos)
        
        # apply rigid body API and schema # 중력 적용 x 
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.GetDisableGravityAttr().Set(True)
        rb.GetRetainAccelerationsAttr().Set(True)
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        attr = prim.GetAttribute("physxRigidBody:disableGravity")
        # print("object gravity info : ",attr)

        self._sim_config.apply_articulation_settings(
            "bin", get_prim_at_path(goal_.prim_path), self._sim_config.parse_actor_config("goal_object")
        )
        """
        #### replicator 
        # bin_rep = self.rep.create.from_usd(sampled_obj, semantics=[("class", "bin")], count=self._num_envs, obj_category="bin")
        # with bin_rep:
        #     self.rep.modify.pose(scale=1)
        
        # bin_prim_path = "/Replicator/bin/.*"
        # bin_prim_paths = find_matching_prim_paths(bin_prim_path)
        # print("bin_prim_paths : ",bin_prim_paths)
        """

    def _find_usd_assets(self, root, categories, max_asset_size, split, target:str = "*/*.usd" ,train=True):
        """Look for USD files under root/category for each category specified.
        For each category, generate a list of all USD files found and select
        assets up to split * len(num_assets) if `train=True`, otherwise select the
        remainder.
        """
        references = {}
        for category in categories:
            all_assets = glob.glob(os.path.join(root, category,target), recursive=True)
            # Filter out large files (which can prevent OOM errors during training)
            if max_asset_size is None:
                assets_filtered = all_assets
            else:
                assets_filtered = []
                for a in all_assets:
                    if os.stat(a).st_size > max_asset_size * 1e6:
                        print(f"{a} skipped as it exceeded the max size {max_asset_size} MB.")
                    else:
                        assets_filtered.append(a)

            num_assets = len(assets_filtered)
            if num_assets == 0:
                raise ValueError(f"No USDs found for category {category} under max size {max_asset_size} MB.")


            references[category] = assets_filtered
        return references


    def get_random_physics_material(self):
        from omni.isaac.core.utils.string import find_unique_string_name
        from omni.isaac.core.utils.prims import is_prim_path_valid
        from omni.isaac.core.materials.physics_material import PhysicsMaterial

        static_friction = np.random.uniform(0.1, 0.3)
        dynamic_friction = np.random.uniform(0.9, 1.0)
        restitution = np.random.uniform(0.02, 0.05)
        
        physics_material_path = find_unique_string_name(
                            initial_name="/World/Physics_Materials/physics_material",
                            is_unique_fn=lambda x: not is_prim_path_valid(x),
                        )
        
        physics_material = PhysicsMaterial(
                            prim_path=physics_material_path,
                            dynamic_friction=dynamic_friction,
                            static_friction=static_friction,
                            restitution=restitution,
                        )
        return physics_material


    def init_data(self) -> None:
        def get_env_local_pose(env_pos, xformable, device):
            """Compute pose in env-local coordinates"""
            world_transform = xformable.ComputeLocalToWorldTransform(0)
            world_pos = world_transform.ExtractTranslation()
            world_quat = world_transform.ExtractRotationQuat()

            px = world_pos[0] - env_pos[0]
            py = world_pos[1] - env_pos[1]
            pz = world_pos[2] - env_pos[2]
            qx = world_quat.imaginary[0]
            qy = world_quat.imaginary[1]
            qz = world_quat.imaginary[2]
            qw = world_quat.real

            return torch.tensor([px, py, pz, qw, qx, qy, qz], device=device, dtype=torch.float)
        
        stage = get_current_stage()
        hand_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda/panda_link7")),
            self._device,
        )
        lfinger_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda/panda_leftfinger")),
            self._device,
        )
        rfinger_pose = get_env_local_pose(
            self._env_pos[0],
            UsdGeom.Xformable(stage.GetPrimAtPath("/World/envs/env_0/franka/panda/panda_rightfinger")),
            self._device,
        )

        finger_pose = torch.zeros(7, device=self._device)
        finger_pose[0:3] = (lfinger_pose[0:3] + rfinger_pose[0:3]) / 2.0
        finger_pose[3:7] = lfinger_pose[3:7]
        hand_pose_inv_rot, hand_pose_inv_pos = tf_inverse(hand_pose[3:7], hand_pose[0:3])

        grasp_pose_axis = 1
        franka_local_grasp_pose_rot, franka_local_pose_pos = tf_combine(
            hand_pose_inv_rot, hand_pose_inv_pos, finger_pose[3:7], finger_pose[0:3]
        )

        franka_local_pose_pos += torch.tensor([0, 0.04, 0], device=self._device)
        self.franka_local_grasp_pos = franka_local_pose_pos.repeat((self._num_envs, 1))
        self.franka_local_grasp_rot = franka_local_grasp_pose_rot.repeat((self._num_envs, 1))

        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )

        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )

        # Franka 초기 위치 변경 가능 
        self.franka_default_dof_pos = torch.tensor(
            [ 0.0, -0.785, 0.0, -2.356, 0.0, 1.571 + 0.3, 0.785, 0.035, 0.035], device=self._device
        )
        
        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)


    def get_observations(self) -> dict:
        franka_pos, _ = self._frankas.get_world_poses(clone=False)
        hand_pos, hand_rot = self._frankas._hands.get_world_poses(clone=False)
        franka_dof_pos = self._frankas.get_joint_positions(clone=False)
        franka_dof_vel = self._frankas.get_joint_velocities(clone=False) # Shape : [num_envs, num_activator] revolute joint는 각속도, prismetic joint는 선속도
        franka_dof_torque = self._frankas.get_measured_joint_forces(joint_indices=torch.tensor([1,2,3,4,5,6,7,8])) # Shape : [num_envs, total_link_num, 6 dof force (x,y,z,roll,pitch,yaw)]
        """
        # get_measured_joint_efforts : torch.Size([env_num, 9]) # 각 actuator에 적용된 종합 힘. -> 7 dof robot + 2 actuator gripper
        # get_measured_joint_forces shape :  torch.Size([env_num, 11, 6]) # 연결된 각각의 joint에 적용된 6 dof 힘 -> base 고정 링크 + 로봇팔 8 개 링크 + 그리퍼 링크 2 -> 총 11 개 링크  
        # 예시   # Effort :  tensor([[ 0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00,
                #   0.0000e+00],
                # [ 5.2287e+02, -8.0750e+01,  3.4861e+02, -8.6969e+01,  8.5961e+01,
                #   1.0256e+02],
                # [-8.0872e+01, -3.7305e+02, -4.7550e+02,  8.5107e+01, -9.6281e+01,
                #   9.1903e+01],
                # [ 3.7456e+02,  2.2677e+02, -3.8889e+02,  9.6205e+01, -1.2140e+02,
                #   4.9885e+01],
                # [-1.9532e+02,  4.1473e+02,  3.2616e+02,  8.9077e+01, -4.0063e+01,
                #   8.5110e+01],
                # [ 3.9790e+02, -3.5780e+02, -7.4849e+01, -2.5952e+01, -1.9314e+01,
                #  -1.1372e+01],
                # [ 3.4150e+02,  3.7579e+02, -9.1971e+01,  2.3253e+01, -2.7014e+01,
                #  -1.3488e+01],
                # [-6.2229e+00, -3.8651e+00, -1.0437e+01, -1.6591e-02,  1.2840e+00,
                #  -4.0790e-01],
                # [ 6.8070e+00,  3.1449e+00, -3.8709e+00, -1.2294e-01,  2.0252e-01,
                #   2.4074e-02],
                # [ 2.6875e-02, -2.2372e-01, -1.3630e-01,  5.4805e-03,  2.3676e-03,
                #  -2.6375e-03],
                # [-1.1019e-01,  2.3918e-01, -1.1349e-01, -5.8273e-03, -1.3643e-03,
                #   2.8578e-03]], device='cuda:0')
        """
        
        self.franka_dof_pos = franka_dof_pos
        self.franka_lfinger_pos, self.franka_lfinger_rot = self._frankas._lfingers.get_world_poses(clone=False)
        self.franka_rfinger_pos, self.franka_rfinger_rot = self._frankas._lfingers.get_world_poses(clone=False)

        dof_pos_scaled = (
            2.0 
            * (franka_dof_pos - self.franka_dof_lower_limits) 
            / (self.franka_dof_upper_limits - self.franka_dof_lower_limits) 
            - 1.0
        ) # 각 joint 별로 -1~1 사이 값 갖게 됨 

        # TODO torque normalize 더 좋은 방법 생각해보기.. x,y,z,a,b,g의 scale이 좀 다 달라서 어케 줘야할지 애매 
        franka_dof_torque[:,:4,:3] /= self.dof_torque_xyz_scale
        franka_dof_torque[:,:4,3:] /= self.dof_torque_abg_scale
        # print("normalized joint torques : ", franka_dof_torque[0])

        # 내 case의 경우
        # Basket 위치
        self.bin_pos, self.bin_rot = self._bins.get_world_poses(clone=False)
        self.obj_pos, self.obj_rot = self._objects.get_world_poses(clone=False)
    
        bin_related_pos = self.bin_pos - franka_pos

        # vision 정보 
        """
        # Depth 받고
        # 이걸 intrinsic param으로 cam 기준 pc를 얻어야함. (World 기준 pc를 얻을 필요 없음) 
        # world로 바꾸는 건 standalone_examples/api/omni.isaac.franka/follow_target_with_rmpflow_copy.py 참고
        # images = self.pytorch_listener.get_rgb_data() # Shape : [num_envs, 3, width, height]
        # depths = self.pytorch_listener.get_depth_data() # Shape : [num_envs, width, height]
        # segmentation_results, is_info_list = self.pytorch_listener.get_semantic_segmentation()
        # cam_point_cloud, reset_env_ids = self.pytorch_listener.get_point_cloud_data(K = torch.tensor(self.Depth[0].camera_matrix), npoints=20000) 

        # point cloud shape : [num_envs, npoints, 3]
        """
        images = self.pytorch_listener.get_rgb_data() # Shape : [num_envs, 3, width, height]
        depths = self.pytorch_listener.get_depth_data() # Shape : [num_envs, width, height]
        cam_point_cloud, reset_env_ids = self.pytorch_listener.get_point_cloud_data(K = torch.tensor(self.Depth[0].camera_matrix), npoints=20000) 
        if reset_env_ids != []:
            self.reset_buf[reset_env_ids] = 1 

        # TODO
        self.obs_buf = torch.cat(
            (
                dof_pos_scaled, # 
                franka_dof_vel * self.dof_vel_scale,
                # bin_related_pos,
            ),
            dim=-1,
        )
        print(self.obs_buf.shape)
        self.franka_torques = franka_dof_torque
        self.cam_point_cloud = cam_point_cloud

        self.obs_buf = torch.rand(
            (self.num_envs, 2048, 3), device=self.device, dtype=torch.float)

        observation = {self._frankas.name: {"obs_buf": self.obs_buf, "pc_buf" : self.cam_point_cloud}}
        # print("pos scaled : ", dof_pos_scaled.shape)
        # print("franka vel : ", (franka_dof_vel * self.dof_vel_scale).shape)
        # print("bin_related_pos scaled : ", bin_related_pos.shape)
        # print("franka_torques : ", franka_dof_torque.shape)
        # print("pc shape : ", cam_point_cloud.shape)

        # # Observation data save
        # if self.t%300 == 0:
        #     np.savez("rgb_depth_pc", rgb = images.permute(0,2,3,1).detach().cpu().numpy(), depth= depths.detach().cpu().numpy(), pc= cam_point_cloud.detach().cpu().numpy()) # pc = pc_full ,world_pc= world_pc , seg = instance_data.detach().cpu().numpy()) # cam_pos= cam_pos, cam_rot=cam_rot)
        #     self.t = 0 
        # self.t += 1

        return observation
    
    def get_cam_ros_pose(self):
        pass 
        # TODO

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return 
        
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            # print("reset_env_ids :", reset_env_ids)
            self.reset_idx(reset_env_ids)

        self.actions = actions.clone().to(self._device)
        # 어떻게 행동 할지 간단한 수식 구현 필요..! 어떤 행동을 하게 만들지에 따라 달라질 듯
        # Franka Cabinet 여는 경우, 현재 joint position에 action으로 나온 delta position을 더해서 set_joint_position_targets
        targets = self.franka_dof_targets + self.actions * self.franka_dof_speed_scales * self.action_scale * self.dt 
        self.franka_dof_targets[:] = tensor_clamp(targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        env_ids_int32 = torch.arange(self._frankas.count, dtype=torch.int32, device=self._device)

        self._frankas.set_joint_position_targets(self.franka_dof_targets, indices=env_ids_int32)


    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids)+1, 6), device=self.device)

        # Robot이든 object든 초기화 시, 0 위치로 가는게 아니라, 특정 random pose로 초기화 가능 이건 추후에 보기로 
        # reset franka
        pos = tensor_clamp(
            self.franka_default_dof_pos.unsqueeze(0)
            + 0.25 * (torch.rand((len(env_ids), self.num_franka_dofs), device=self._device) - 0.5),
            self.franka_dof_lower_limits,
            self.franka_dof_upper_limits,
        )
        dof_pos = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
        dof_vel = torch.zeros((num_indices, self._frankas.num_dof), device=self._device)
        dof_pos[:, :] = pos
        self.franka_dof_targets[env_ids, :] = pos
        self.franka_dof_pos[env_ids, :] = pos

        self._frankas.set_joint_position_targets(self.franka_dof_targets[env_ids], indices=indices)
        self._frankas.set_joint_positions(dof_pos, indices=indices)
        self._frankas.set_joint_velocities(dof_vel, indices=indices)
        
        # reset table 
        self._tables.set_world_poses(positions=self.table_init_pos[env_ids], orientations=self.table_init_rot[env_ids], indices=indices)
        table_velocities = torch.zeros_like(self.table_init_velocities, dtype=torch.float, device=self.device)
        self._tables.set_velocities(table_velocities[env_ids], indices)
    
        # reset object
        # basket 위치 임의로 할당 + object 위치 tabel 위에 임의로 할당. 
        reset_type_tensor = torch.rand(num_indices)
        obj_type_1 = torch.where(reset_type_tensor < 0.5)[0]  # obj_type_1의 인덱스
        obj_type_2 = torch.where(reset_type_tensor >= 0.5)[0]  # obj_type_2의 인덱스
        
        if len(obj_type_1):
            obj_ids = env_ids[obj_type_1]
            new_object_pos = (self.object_init_pos[obj_ids] + self._env_pos[obj_ids]) 
            new_object_pos[:,0] += 0.1 * rand_floats[obj_type_1, 0]
            new_object_pos[:,1] += 0.3 * rand_floats[obj_type_1, 1]
            new_object_rot = randomize_rotation(
                rand_floats[obj_type_1, 3], rand_floats[obj_type_1, 4], self.x_unit_tensor[indices[obj_type_1]], self.y_unit_tensor[indices[obj_type_1]]
            )
            self._objects.set_world_poses(new_object_pos, new_object_rot, indices[obj_type_1])
        
        if len(obj_type_2):
            obj_ids = env_ids[obj_type_2]
            new_object_pos = (self.object_init_pos[obj_ids] + self._env_pos[obj_ids]) 
            new_object_pos[:,0] += 0.1 * rand_floats[obj_type_2, 0]
            new_object_pos[:,1] += 0.3 * rand_floats[obj_type_2, 1]
            self._objects.set_world_poses(new_object_pos, indices= indices[obj_type_2])

        object_velocities = torch.zeros_like(self.object_init_velocities, dtype=torch.float, device=self.device)
        self._objects.set_velocities(object_velocities[env_ids], indices)

        # reset bin
        reset_type_tensor = torch.rand(num_indices)
        bin_type_1 = torch.where(reset_type_tensor < 0.33)[0]  # bin_type_1의 인덱스
        bin_type_2 = torch.where(reset_type_tensor > 0.67)[0]  # bin_type_2의 인덱스
        # bin_type_1 또는 bin_type_2에 속하지 않는 나머지 인덱스
        all_indices = torch.arange(len(reset_type_tensor))
        bin_type_3 = all_indices[(reset_type_tensor >= 0.33) & (reset_type_tensor <= 0.67)]

        if len(bin_type_1):
            new_bin_pos = (self.bin_init_pos[env_ids[bin_type_1]] + self._env_pos[env_ids[bin_type_1]])
            new_bin_pos[:,1] -= torch.rand(len(bin_type_1)).cuda() * (self._table_xyz[1] + self.bin_xyz[1]) 
            new_bin_pos[:,2] -= (2*torch.rand(len(bin_type_1)).cuda() -1) * 0.2 
            # new_bin_rot = TODO
            self._bins.set_world_poses(new_bin_pos, indices=indices[bin_type_1])

        if len(bin_type_2):
            new_bin_pos = (self.bin_init_pos[env_ids[bin_type_2]] + self._env_pos[env_ids[bin_type_2]])
            new_bin_pos[:,0] -= torch.rand(len(bin_type_2)).cuda() * (self._table_xyz[0] + self.bin_xyz[0]) 
            new_bin_pos[:,2] -= (2*torch.rand(len(bin_type_2)).cuda() -1) * 0.2
        #     new_bin_rot = randomize_rotation(
        #         rand_floats[env_ids[bin_type_2],5] / 10, rand_floats[env_ids[bin_type_2],5] * 0.001, self.x_unit_tensor[env_ids[bin_type_2]], self.y_unit_tensor[env_ids[bin_type_2]]
        #     )
            self._bins.set_world_poses(new_bin_pos, indices=indices[bin_type_2])

        if len(bin_type_3):
            new_bin_pos = (self.bin_init_pos[env_ids[bin_type_3]] + self._env_pos[env_ids[bin_type_3]])
            new_bin_pos[:,0] -= torch.rand(len(bin_type_3)).cuda() * (self._table_xyz[0] + self.bin_xyz[0])
            new_bin_pos[:,1] -= (self._table_xyz[1] + self.bin_xyz[1] + 0.1)  
            new_bin_pos[:,2] -= (2*torch.rand(len(bin_type_3)).cuda() -1) * 0.2
        #     new_bin_rot = randomize_rotation(
        #         rand_floats[env_ids[bin_type_3],5] / 10, rand_floats[env_ids[bin_type_3],5] * 0.001, self.x_unit_tensor[env_ids[bin_type_3]], self.y_unit_tensor[env_ids[bin_type_3]]
        #     )
            self._bins.set_world_poses(new_bin_pos, indices = indices[bin_type_3])

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.goal_in_buf[env_ids] = 0 

    def post_reset(self):
        # FRANKA 관련 초기화 
        self.num_franka_dofs = self._frankas.num_dof
        self.franka_dof_pos = torch.zeros((self.num_envs, self.num_franka_dofs), device=self._device)
        dof_limits = self._frankas.get_dof_limits()
        self.franka_dof_lower_limits = dof_limits[0, :, 0].to(device=self._device)
        self.franka_dof_upper_limits = dof_limits[0, :, 1].to(device=self._device)
        self.franka_dof_speed_scales = torch.ones_like(self.franka_dof_lower_limits)
        self.franka_dof_speed_scales[self._frankas.gripper_indices] = 0.1
        self.franka_dof_targets = torch.zeros(
            (self._num_envs, self.num_franka_dofs), dtype=torch.float, device=self._device
        )
    
        # table 관련 초기화 
        self.table_init_pos, self.table_init_rot = self._tables.get_world_poses()
        self.table_init_velocities = torch.zeros_like(
                        self._tables.get_velocities(), dtype=torch.float, device=self.device
                    )

        # object 관련 초기화
        self.object_init_pos, self.object_init_rot = self._objects.get_world_poses()
        self.object_init_pos -= self._env_pos
        self.object_init_velocities = torch.zeros_like(
            self._objects.get_velocities(), dtype=torch.float, device=self.device
        )

        # bin 관련 초기화 
        self.bin_init_pos, _ = self._bins.get_world_poses()
        self.bin_init_pos -= self._env_pos
        self.bin_init_rot = np.array([90,0,0])


        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        #  이건 작성하기 나름 - 고민 많이 필요 
        # Sparse reward setting 골 넣으면 1 아니면 0        
        self.rew_buf = torch.zeros_like(self.rew_buf)
        self.rew_buf = torch.where(self.goal_in_buf > 15, torch.ones_like(self.rew_buf), self.rew_buf)


    def is_done(self) -> None:
        # 언제 reset 할지 조건들 선언 

        # bin 안에 들어가면 성공. reset 
        self.goal_in_buf = torch.where(((self.obj_pos[:,0] <= self.bin_pos[:,0] + self.bin_xyz[0]/2) & (self.obj_pos[:,0] >= self.bin_pos[:,0] - self.bin_xyz[0]/2) & \
                            (self.obj_pos[:,1] <= self.bin_pos[:,1] + self.bin_xyz[1]/2) & (self.obj_pos[:,1] >= self.bin_pos[:,1] - self.bin_xyz[1]/2) & \
                            (self.obj_pos[:,2] <= self.bin_pos[:,2] ) & (self.obj_pos[:,2] >= self.bin_pos[:,2] - self.bin_xyz[2])),
                            self.goal_in_buf + 1,
                            self.goal_in_buf)
        self.reset_buf = torch.where(self.goal_in_buf > 20, torch.ones_like(self.reset_buf), self.reset_buf)
        # Can이 table에서 떨어지면 reset 
        self.reset_buf = torch.where(self.obj_pos[:,2] < 0.2, torch.ones_like(self.reset_buf), self.reset_buf)
        # Max iteration 넘어가면 reset 
        self.reset_buf = torch.where(
            self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf
        )



@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )
