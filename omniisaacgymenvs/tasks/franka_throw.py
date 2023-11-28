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

from omni.isaac.core.utils.numpy.rotations import quats_to_rot_matrices, euler_angles_to_quats
from omni.isaac.core.utils.extensions import enable_extension
from omni.isaac.core.utils.prims import find_matching_prim_paths

from PIL import Image

# from omni.isaac.core.prims.base_sensor import BaseSensor

class FrankaRAIThrowTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.update_config(sim_config)

        self.distX_offset = 0.04
        self.dt = 1 / 60.0

        self._num_observations = 23
        self._num_actions = 9

        # use multi-dimensional observation for pointclod from camera (2.5 D)
        # High Dim 
        # self.observation_space = spaces.Box(
        #     np.ones((20000, 3), dtype=np.float32) * -np.Inf, 
        #     np.ones((20000, 3), dtype=np.float32) * np.Inf)
        self.observation_space = spaces.Box(
            np.ones((100, 3), dtype=np.float32) * -np.Inf, 
            np.ones((100, 3), dtype=np.float32) * np.Inf)

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
        
        self._table_position = np.array([0.5, 0.0, 0.0])  # Gf.Vec3f(0.5, 0.0, 0.0)
        self._table_scale = 0.0125
        self._franka_position = np.array([0.85, 0.0, 1.03330328])  # Gf.Vec3f(0.269, 0.1778, 0.0)

        RLTask.__init__(self, name, env)
    
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
        # 여기에 우리가 할 task, scene 불러와야함   
        self.get_franka()

        ## Tabel 추가
        self.get_table()

        # 환경 Clone 실행 
        super().set_up_scene(scene, filter_collisions=False)


        self._frankas = FrankaView(prim_paths_expr="/World/envs/.*/franka/panda", name="franka_view", custom_path_name="panda")
        self._tables = RigidPrimView(
            prim_paths_expr="/World/envs/.*/table", name="table_view", reset_xform_properties=False
        )
    
        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._lfingers)
        scene.add(self._frankas._rfingers)
        scene.add(self._frankas._camera)
        scene.add(self._tables)

        # 환경마다 다른 objecet 추가 ## Reset 시, init 시, 동일하게 적용되어야 함!! 주의 *** 
        # if self.num_props > 0:
        #     self.get_props()

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
        # print("Camera render_products", self.render_products)
        
        self.pytorch_listener = self.PytorchListener()
        self.pytorch_writer = self.rep.WriterRegistry.get("PytorchWriter")
        print("Writer : ",self.pytorch_writer)
        self.pytorch_writer.initialize(listener=self.pytorch_listener, device="cuda", depth=True)
        self.pytorch_writer.attach(self.render_products)
        
        ##
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

        #### 
        # Replicator를 prop 안에 넣고, Object 추가하면 되겠는데
        # if self.num_props > 0:
        #     self._props = RigidPrimView(
        #         prim_paths_expr="/World/envs/.*/prop/.*", name="prop_view", reset_xform_properties=False
        #     )
        #     scene.add(self._props)

        # scene.enable_bounding_boxes_computations()
        # self._add_table()

        self.init_data()
        return

    def get_franka(self):
        franka = Franka(prim_path=self.default_zero_env_path + "/franka", name="franka", usd_path= "/home/juju/Downloads/panda_with_cam_d435.usd" , custom_prim_name="panda", translation=self._franka_position)
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
            mass=100
        )
        self._sim_config.apply_articulation_settings(
            "table", get_prim_at_path(table.prim_path), self._sim_config.parse_actor_config("table")
        )


    def get_object(self):
        self.rep.orchestrator._orchestrator._is_started = True

        # from ShapeNet core
        self.throw_asset_1 = self._find_usd_assets(root = "/home/juju/juhan_ws/shapenet_nomat/for_throw", categories=["good_throw"], max_asset_size=None, split=1)
        
        # spheres = self.rep.create.sphere(count=10, scale=self.rep.distribution.uniform(1., 3.))
        # print("Spheres :", spheres)
        # with self.rep.trigger.on_custom_event(event_name="Randomize!"):
        #     with spheres:
        #         mod = self.rep.modify.pose(position=self.rep.distribution.uniform((-50., -50., -50.), (50., 50., 50.)))
        # # Send event
        # self.rep.utils.send_og_event("Randomize!")

        # 각 환경마다 table위에 다른 object를 올리고 싶었으나, 실패. -> 가장 처음 올라온 object로 무지성 clone 되버림. 
        print("env pose : ", self._env_pos)
        # sampled_obj = random.sample(self.throw_asset_1["good_throw"],1)[0]
        sampled_obj = self.throw_asset_1["good_throw"][-3]
        object_ = DynamicCustomObject(
            prim_path= self.default_zero_env_path + "/object",
            name = "object",
            usd_path = sampled_obj,
            translation= self._franka_position + np.array([-0.4, 0., 0.2]),
            # translation= np.array([0, -1, 1.5]),
            scale = np.array([0.1]),
            # scale = np.array([1]),
            mass=0.1
        )
        self._sim_config.apply_articulation_settings(
            "object", get_prim_at_path(object_.prim_path), self._sim_config.parse_actor_config("object")
        )
    
    def get_goal_bin(self):
        self.basket_asset = self._find_usd_assets(root = "/home/juju/juhan_ws/shapenet_nomat/for_throw", categories=["good_trash_bin"],max_asset_size=None, split=1)
        
        sampled_obj = self.basket_asset["good_trash_bin"][-2]
        # add_reference_to_stage(sampled_obj, self.default_zero_env_path + "/bin")

        self.goal_start_translation = np.array([0, -1, 1])
        goal = DynamicCustomObject(
            prim_path=self.default_zero_env_path + "/bin",
            usd_path = sampled_obj,
            translation=self.goal_start_translation,
            orientation= euler_angles_to_quats(euler_angles=np.array([90,0,0]),degrees=True),
            scale= np.array([1.]),
        )

        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(goal.prim_path)

        # apply rigid body API and schema
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.GetDisableGravityAttr().Set(True)
        rb.GetRetainAccelerationsAttr().Set(True)
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim)



        attr = prim.GetAttribute("physxRigidBody:disableGravity")
        print("object gravity info : ",attr)
        self._sim_config.apply_articulation_settings(
            "bin", get_prim_at_path(goal.prim_path), self._sim_config.parse_actor_config("goal_object")
        )

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
        restitution = np.random.uniform(0.05, 0.4)
        
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
        # table도 추가하기 
        self._tables = RigidPrimView(
            prim_paths_expr="/World/envs/.*/table", name="table_view", reset_xform_properties=False
        )

        scene.add(self._frankas)
        scene.add(self._frankas._hands)
        scene.add(self._frankas._camera_hand)
        scene.add(self._frankas._lfingers)
        scene.add(self._frankas._rfingers)
        # 아래에 table, object 추가 
        scene.add(self._tables)

        self.init_data()

    def get_props(self):
        # torch로 쓰여져있음 ! 
        prop_cloner = Cloner()
        drawer_pos = torch.tensor([0.0515, 0.0, 0.7172])
        prop_color = torch.tensor([0.2, 0.4, 0.6])

        props_per_row = int(math.ceil(math.sqrt(self.num_props)))
        prop_size = 0.08
        prop_spacing = 0.09
        xmin = -0.5 * prop_spacing * (props_per_row - 1)
        zmin = -0.5 * prop_spacing * (props_per_row - 1)
        prop_count = 0

        prop_pos = []
        for j in range(props_per_row):
            prop_up = zmin + j * prop_spacing
            for k in range(props_per_row):
                if prop_count >= self.num_props:
                    break
                propx = xmin + k * prop_spacing
                prop_pos.append([propx, prop_up, 0.0])

                prop_count += 1

        prop = DynamicCuboid(
            prim_path=self.default_zero_env_path + "/prop/prop_0",
            name="prop",
            color=prop_color,
            size=prop_size,
            density=100.0,
        )
        self._sim_config.apply_articulation_settings(
            "prop", get_prim_at_path(prop.prim_path), self._sim_config.parse_actor_config("prop")
        )

        prop_paths = [f"{self.default_zero_env_path}/prop/prop_{j}" for j in range(self.num_props)]
        prop_cloner.clone(
            source_prim_path=self.default_zero_env_path + "/prop/prop_0",
            prim_paths=prop_paths,
            positions=np.array(prop_pos) + drawer_pos.numpy(),
            replicate_physics=False,
        )

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

        # object 관련  초기화할 부분 추가 
        self.gripper_forward_axis = torch.tensor([0, 0, 1], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )

        self.gripper_up_axis = torch.tensor([0, 1, 0], device=self._device, dtype=torch.float).repeat(
            (self._num_envs, 1)
        )

        self.franka_default_dof_pos = torch.tensor(
            [1.157, -1.066, -0.155, -2.239, -1.841, 1.003, 0.469, 0.035, 0.035], device=self._device
        )
        
        self.actions = torch.zeros((self._num_envs, self.num_actions), device=self._device)



    def get_observations(self) -> dict:
        hand_pos, hand_rot = self._frankas._hands.get_world_poses(clone=False)
        franka_dof_pos = self._frankas.get_joint_positions(clone=False)
        franka_dof_vel = self._frankas.get_joint_velocities(clone=False)

        # get_measured_joint_efforts : torch.Size([512, 9]) # 각 actuator에 적용된 종합 힘. -> 7 dof robot + 2 actuator gripper
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
        franka_dof_torque = self._frankas.get_measured_joint_forces()
        self.franka_dof_pos = franka_dof_pos

        self.franka_lfinger_pos, self.franka_lfinger_rot = self._frankas._lfingers.get_world_poses(clone=False)
        self.franka_rfinger_pos, self.franka_rfinger_rot = self._frankas._rfingers.get_world_poses(clone=False)

        dof_pos_scaled = (
            2.0 
            * (franka_dof_pos - self.franka_dof_lower_limits) 
            / (self.franka_dof_upper_limits - self.franka_dof_lower_limits) 
            - 1.0
        )

        # Target 선언 가능 

        # Observation 할 부분 선언 
            # 내 case의 경우
                # Basket 위치
                # 로봇 관련 state (vel, torque)
                # vision 정보 
        # self.obs_buf = torch.cat(
        #     (
        #         dof_pos_scaled,
        #         franka_dof_vel * self.dof_vel_scale,
        #         to_target,
        #         self.cabinet_dof_pos[:, 3].unsqueeze(-1),
        #         self.cabinet_dof_vel[:, 3].unsqueeze(-1),
        #     ),
        #     dim=-1,
        # )

        # Depth 받고 + Segmentation도 같이 받고 
        # 이걸 intrinsic param으로 cam 기준 pc를 얻어야함. (World 기준 pc를 얻을 필요 없음)
        # self.obs_buf = 
        
        images = self.pytorch_listener.get_rgb_data()
        depths = self.pytorch_listener.get_depth_data()
        # print("Image shape : ", images.shape)
        # print("depths shape : ", depths.shape)
        
        # For test until it could load robot in the scene
        self.obs_buf = torch.zeros(
            (self.num_envs, 20000, 3), device=self.device, dtype=torch.float)
        
        self.obs_buf = torch.rand(
            (self.num_envs, 100, 3), device=self.device, dtype=torch.float)
        
        observation = {self._frankas.name: {"obs_buf": self.obs_buf}}
        
        if self.t%300 == 0:
            np.savez("rgb_depth_pc", rgb = images.permute(0,2,3,1).detach().cpu().numpy(), depth= depths.detach().cpu().numpy()) # pc = pc_full ,world_pc= world_pc , seg = instance_data.detach().cpu().numpy()) # cam_pos= cam_pos, cam_rot=cam_rot)
            self.t = 0 
        self.t += 1
        return observation
    

    def pre_physics_step(self, actions) -> None:
        if not self._env._world.is_playing():
            return 
    
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.to(self._device)

        # 어떻게 행동 할지 간단한 수식 구현 필요..! 어떤 행동을 하게 만들지에 따라 달라질 듯
        # Franka Cabinet 여는 경우, 현재 joint position에 action으로 나온 delta position을 더해서 set_joint_position_targets
        # Cartpole의 경우 cart의 force를 바로 넣어버림 action은 0~1 사이로 나오고 max push effort를 곱해서 scaling
        targets = self.franka_dof_targets + self.franka_dof_speed_scales * self.dt * self.actions * self.action_scale
        self.franka_dof_targets[:] = tensor_clamp(targets, self.franka_dof_lower_limits, self.franka_dof_upper_limits)
        env_ids_int32 = torch.arange(self._frankas.count, dtype=torch.int32, device=self._device)
        self._frankas.set_joint_position_targets(self.franka_dof_targets, indices=env_ids_int32)


    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        num_indices = len(indices)

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

        # reset table

        # reset object
        # basket 위치 임의로 할당 + object 위치 tabel 위에 임의로 할당. 

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def post_reset(self):

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

        # object 관련 초기화
        # if self.num_props > 0:
        #     self.default_prop_pos, self.default_prop_rot = self._props.get_world_poses()
        #     self.prop_indices = torch.arange(self._num_envs * self.num_props, device=self._device).view(
        #         self._num_envs, self.num_props
        #     )


        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        #  이건 작성하기 나름 - 고민 많이 필요 
        # TODO 
        self.rew_buf = torch.zeros(self._frankas.count)

    def is_done(self) -> None:
        # 언제 reset 할지 조건들 선언 

        # reset if drawer is open or max length reached
        # self.reset_buf = torch.where(self.cabinet_dof_pos[:, 3] > 0.39, torch.ones_like(self.reset_buf), self.reset_buf)
        
        self.reset_buf = torch.where(
            self.progress_buf >= self._max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf
        )