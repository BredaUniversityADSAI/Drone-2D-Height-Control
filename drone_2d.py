#%%
import drone_2d_custom_gym_env
import gym
import numpy as np
import time

#%%

class drone_2d(gym.Env):
    def __init__(self,dt=0.025, starting_throttle = -1):
        super(drone_2d, self).__init__()
        self.env = gym.make('drone-2d-custom-v0', render_sim=True, render_path=True, render_shade=True,
                   shade_distance=70, n_steps=500, n_fall_steps=0, change_target=True,
                   initial_throw=False)
        # action space is 1 number from -1 to 1
        
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = self.env.observation_space
        self.dt = dt
        self.action_base = np.array([starting_throttle, starting_throttle])
        # PID control parameters
        self.kp = 6
        self.ki = 0.01
        self.kd = 3
        # PID control variables
        self.integral_error = 0
        self.previous_error = 0
        # PID control parameters for pos
        self.kp_p = 1.2
        self.ki_p = 0.1
        self.kd_p = 0.6
        # PID control variables for pos
        self.integral_error_p = 0
        self.previous_error_p = 0

    def step(self, action):
        pos = self.state[6]
        error_p = self.possetpoint - pos
        self.integral_error += error_p
        derivative_error_p = (error_p - self.previous_error_p)/self.dt
        self.previous_error_p = error_p
        rollsetpoint = -np.clip(error_p * self.kp_p + self.integral_error_p * self.ki_p + derivative_error_p * self.kd_p, -0.35, 0.35)

        angle = self.state[3]
        error = rollsetpoint - angle
        self.integral_error += error
        derivative_error = (error - self.previous_error)/self.dt
        self.previous_error = error
        action_mod_roll = (error * self.kp + self.integral_error * self.ki + derivative_error * self.kd)

        action_base = np.array([self.action_base[0] - action_mod_roll, self.action_base[1] + action_mod_roll])

        full_action = action_base + action

        self.state, self.reward, self.done, self.info = self.env.step(full_action)
        return self.state, self.reward, self.done, self.info

    def reset(self):
        self.state = self.env.reset()
        while (self.state[3] > 0.35 or self.state[3] < -0.35) or (self.state[7] < 0.0 or self.state[7] > 0.2):
            self.state = self.env.reset() 
        self.possetpoint = self.state[6]
        return self.state
    
    def render(self):
        self.env.render()



