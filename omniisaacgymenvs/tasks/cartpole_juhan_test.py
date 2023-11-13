# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math

import numpy as np
import torch
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.robots.articulations.cartpole import Cartpole


class CartpoleJuhanTestTask(RLTask):
    def __init__(self, name, sim_config, env, offset=None) -> None:
        """_summary_

        Args:
            name (str): Env name
            sim_config (_type_): contain task information, such as the number of env used for the task,
                                and physics params, including simulation dt, GPU buffer dim  etc.. 
            env (VecEnvBase): in our case, this will be a VenEnvRLGames object defined by the rlgames_train.py
            offset (_type_, optional): _description_. Defaults to None.
        
        Description:
            In this task, we will use 4 observations to represent the joint position & velocities for the cart and 
            pole joints, and 1 action to apply as the force to the cart. 

        """
        self.update_config(sim_config=sim_config)
        self._max_episode_length = 600 
        
        #these must be defined in the task class 
        self._num_observations = 4 
        self._num_actions = 1 
        
        # Call the parent class constructor to initialize key RL variables
        # RL task 셋팅을 위한 Pytorch RL-specific interface 제공 
        # 환경 복제 + RL 알고리즘을 위한 데이터 수집
        # ******** RL 환경 학습을 위해 camera data collection 위해서 replicator를 설정해야함. ********* 
        RLTask.__init__(self, name, env, offset=None)
        return 
    
    def update_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config 
        self._task_cfg = sim_config.task_config 
        
        # parse task config parameters 
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._cartpole_positions = torch.tensor([0., 0., 2.])
        
        # reset and actions related variables 
        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]

    # apply physics params to the robot & clone it multiple times to build out vectorized env !!
    # 반드시 있어야할 함수임 -> world 초기화 시, isaac sim framework에 의해 자동적으로 실행될 것임. 
    def set_up_scene(self, scene) -> None: 
        # first create a single env
        self.get_cartpole() 
        
        # call the parent class to clone the single env 
        # ground plane과 light prims 까지 표현함. + 서로 복제된 env끼리 충돌 필터링이 적용됨. 
        super().set_up_scene(scene)
        
        # *************** Construct an ArticulationView object to hold out collection of env************
        self._cartpoles = ArticulationView(
            prim_paths_expr="/World/envs/.*/Cartpole", name="cartpole_view", reset_xform_properties=False
        )
        # self._cartpoles
        # register the ArticulationView object to the world, so that it can be initialized
        scene.add(self._cartpoles)
    
    # 이걸 불러옴으로써 single env를 불러오는 것임. 
    def get_cartpole(self):
        # add a single robot to the stage
        # print("print Cartpole default zero env path : ", self.default_zero_env_path)
        
        cartpole = Cartpole(
            prim_path=self.default_zero_env_path + "/Cartpole", name="Cartpole", translation=self._cartpole_positions
        )
        # print("print Cartpole prim path : ", get_prim_at_path(cartpole.prim_path))
        # 바로 위 코드에서 cartpole robot instance를 받아오고 받아온 asset에 Simconfig의 파라미터를 적용시킴. 
        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings(
            "Cartpole", get_prim_at_path(cartpole.prim_path), self._sim_config.parse_actor_config("Cartpole")
        )
        
        
    def post_reset(self):
        # retrieve cart and pole joint indices
        self._cart_dof_idx = self._cartpoles.get_dof_index("cartJoint")
        self._pole_dof_idx = self._cartpoles.get_dof_index("poleJoint")
        
        # randomize all envs
        indices = torch.arange(self._cartpoles.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)
    
    # Cartpole task를 초기화하여 각 env별로 임의의 starting state를 갖게 하여 RL policy가 randomized state에서
    # 학습이 start되도록 하게한다. 여러 환경 중 env_ids를 가져다가 그 ids에 대해 초기화를 해버림!!! 
    def reset_idx(self, env_ids):  # sourcery skip: extract-duplicate-method
        num_resets = len(env_ids)
        
        # 초기화 할 때, 로봇의 모든 joint_dof_idx를 포함하긴 해야함 -> carpole은 cart와 pole로 이뤄저 있어서
        # dof_idx를 두 그룹으로 나누어 초기화 한 것임. 
        # 그리고 초기화는 set_joint_positions & set_joint_velocities로 초기화 될 수 있음 
        # 
        
        # randomize DOF positions
        dof_pos = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
        # cart pos init
        dof_pos[:, self._cart_dof_idx] = 1.0 * (1. - 2. * torch.rand(num_resets, device= self._device))
        # pole pos init 
        dof_pos[:, self._pole_dof_idx] = 0.125 * math.pi * (1. - 2. * torch.rand(num_resets, device=self._device))
        
        # random DOF velocities
        dof_vel = torch.zeros((num_resets, self._cartpoles.num_dof), device=self._device)
        dof_vel[:, self._cart_dof_idx] = 0.5 * (1. - 2. * torch.rand(num_resets, device=self._device))
        dof_vel[:, self._pole_dof_idx] = 0.25 * math.pi * (1. - 2. * torch.rand( num_resets, device=self._device))
        
        # apply randomized joint pos, vel to envs 
        indices = env_ids.to(dtype = torch.int32)
        self._cartpoles.set_joint_positions(dof_pos, indices=indices)
        self._cartpoles.set_joint_velocities(dof_pos, indices=indices)
        
        # reset the reset buffer & progress buffer after applying reset 
        # buffer가 무슨 역할을 하지 ? 
            # reset_buf : reset이 필요한지 아닌지 나타내는 boolean tensor
        self.reset_buf[env_ids] = 0 
            # progress_buf : reset되고 몇번의 step이 진행 되었는지 
        self.progress_buf[env_ids] = 0         

    # 시뮬레이션에서 step 하기 전에 next step 진행전에 env에 들어온 action을 적용해야한다. 
    # reset_buf에 표시된 것들은 reset할 수 있고 그게 아닌 경우는 RL policy로부터 받은 action대로 시뮬됨
    def pre_physics_step(self, actions) -> None:
        # make sure sim has not been stopped from the UI
        if not self._env._world.is_playing():
            return
        
        # extract env indices that need reset and reset them
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        
        # make sure actions buffer is on the same device as the sim
        actions = actions.to(self._device)
        
        # Action 또한 Robot의 dof에 맞게 적용 되어야한다. 
        # Cartpole의 경우 force는 cart에게만 나오기 때문에 pole에는 따로 적용하지 않고 cart에만 적용한다. 
        # compute forces from the actions 
        forces = torch.zeros((self._cartpoles.count, self._cartpoles.num_dof), dtype=torch.float32, device=self._device)
        forces[:, self._cart_dof_idx] = self._max_push_effort * actions[:, 0]
        
        # apply actions to all of the env
        indices = torch.arange(self._cartpoles.count, dtype=torch.int32, device=self._device)
        self._cartpoles.set_joint_efforts(forces, indices=indices)
        
    # 이 함수는 sim에서 states를 추출하는 함수이며 RL의 state로 쓰일 것임 
    # return 시, output은 Isaac Sim에서 썼던 BaseTask의 정의에 따라 Dictionary로 return함
    def get_observations(self) -> dict:
        # retrieve joint pos and vel
        dof_pos = self._cartpoles.get_joint_positions(clone=False)
        dof_vel = self._cartpoles.get_joint_velocities(clone=False)
        
        # extract joint states for the cart and pole joints
        cart_pos = dof_pos[:, self._cart_dof_idx]
        cart_vel = dof_vel[:, self._cart_dof_idx]
        pole_pos = dof_pos[:, self._pole_dof_idx]
        pole_vel = dof_vel[:, self._pole_dof_idx]

        # populate the observations buffer
        # RLTask class 내부적으로 self.obs_buf 라는 변수가 선언되어 있음. 이게 state 국룰 변수임. 
        self.obs_buf[:, 0] = cart_pos
        self.obs_buf[:, 1] = cart_vel
        self.obs_buf[:, 2] = pole_pos
        self.obs_buf[:, 3] = pole_vel
        # print("Observation : " ,self.obs_buf[:5])
        # construct the observations dict and return 
        observations = {self._cartpoles.name: {"obs_buf": self.obs_buf}}
        return observations 
    
    # Observation으로부터 reward를 측정할 수 있다!!! 
    # reward function이 calculate_metrics임. 
    # BaseTask의 interface를 따라서 calculate_metrics로 선언되었음!!
    
    # pole이 수직으로 새워져있으면 +reward, 
    # Cart나 Pole이 너무 빨리 움직이면 패널티 -> cart, pole의 absolute pos, vel로 측정함. 
    # 패널팉가 시무스하게 움직이도록 만듦. 
    # bad state에 도달하면 large penalty -2를 부여 (distance limit을 넘어가거나, pole이 90도 이상 꺾여버릴 경우)
    
    # 계산된 reward는 rew_buf에 저장 됨
    def calculate_metrics(self) -> dict:
        #use states from the observation buffer to compute reward
        cart_pos = self.obs_buf[:,0]
        cart_vel = self.obs_buf[:,1]
        pole_angle = self.obs_buf[:,2]
        pole_vel = self.obs_buf[:,3]
        
        # define the reward function based on pole angle and robot velocities 
        reward = 1. - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.3 * torch.abs(pole_vel)
        # penalize the policy if the cart moves too far on the rail
        reward = torch.where(torch.abs(cart_pos) > self._reset_dist, torch.ones_like(reward)* -2 + reward, reward)
        # penalize the policy if the pole moves beyond 90 degrees 
        reward = torch.where(torch.abs(pole_angle) > np.pi /2, torch.ones_like(reward) - 2. + reward, reward)
        
        # assign rewards to the reward buffer 
        self.rew_buf[:] = reward 
        # print("Reward : ", self.rew_buf[:5])
        # print("reset buf : ", self.reset_buf[:5])
        # print("progress buf : ", self.progress_buf[:5])
    def is_done(self) -> None:
        cart_pos = self.obs_buf[:,0]
        pole_pos = self.obs_buf[:,2]
        
        # check for which conditions are met and mark the env that satisfy the conditions
        resets = torch.where(torch.abs(cart_pos) > self._reset_dist, 1, 0)
        resets = torch.where(torch.abs(pole_pos) > math.pi / 2, 1, resets)
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)
    
        # assign the resets to the reset buffer 
        self.reset_buf[:] = resets         
        
    