"""
Habitat Environment for Household Robot Task Simulation

This module provides a custom OpenAI Gym environment for simulating household robot tasks
using the habitat framework. It supports various object interactions and task scenarios.
The code is based on https://github.com/facebookresearch/habitat-lab and https://github.com/apple/ml-llarp 

Dependencies:
- habitat-lab
- gym
- numpy
- PIL
"""
import gym
import os
import time
import json
import imageio
from PIL import Image 
import numpy as np
import habitat
import hydra
from habitat.datasets import make_dataset
from embodiedbench.envs.eb_habitat.config.default_structured_configs import (
    ThirdRGBSensorConfig,
)
from habitat.gym.gym_definitions import _add_sim_sensor_to_config
from omegaconf import OmegaConf

from habitat_sim.utils import viz_utils as vut
from embodiedbench.envs.eb_habitat.config import default_structured_configs
import embodiedbench.envs.eb_habitat.predicate_task
import embodiedbench.envs.eb_habitat.config
import embodiedbench.envs.eb_habitat.measures
from embodiedbench.envs.eb_habitat.utils import observations_to_image, merge_to_file, draw_text
from embodiedbench.main import logger

HABITAT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config/task/language_rearrangement.yaml')


ValidEvalSets = [
        'base', 'common_sense', 'complex_instruction', 
        'spatial_relationship', 'visual_appearance', 'long_horizon'
    ] 

def add_receptacle(string, skill):
    if 'table_0' in skill[1][0]:
        string += 'table ' + skill[1][0].split('table_0')[1]
    elif 'fridge' in skill[1][0]:
        string += 'refrigerator push point'
    elif 'refrigerator' in skill[1][0]:
        string += 'refrigerator' 
    elif 'drawer_right' in skill[1][0]:
        string += 'right drawer of the kitchen counter'
    elif 'drawer_left' in skill[1][0]:
        string += 'left drawer of the kitchen counter'
    elif 'chair_0' in skill[1][0]:
        string += 'chair ' + skill[1][0].split('chair_0')[1]
    elif 'tvstand' in skill[1][0]:
        string += 'TV stand'
    elif 'counter_left' in skill[1][0]:
        string += 'left counter in the kitchen'
    elif 'counter_right' in skill[1][0]:
        string += 'right counter in the kitchen'
    elif 'sink' in skill[1][0]:
        string += 'sink in the kitchen'
    elif 'sofa' in skill[1][0]:
        string += 'sofa' 
    elif 'cab' in skill[1][0]:
        string += 'cabinet ' + skill[1][0].split('_')[-1]
    else:
        raise NotImplementedError
    return string


def transform_action_to_natural_language(skill_set):
    language_skill_set = []
    for skill in skill_set:
        if 'nav' in skill[0]:
            string = 'navigate to the '
            string = add_receptacle(string, skill)
        elif 'pick' in skill[0]:
            string = 'pick up the ' + skill[0].split('_')[1]
        elif 'open' in skill[0]:
            string = 'open the '
            if 'fridge' in skill[0]:
                string += 'refrigerator'
            elif 'cab' in skill[0]:
                string += 'cabinet ' + skill[1][0].split('_')[-1]
            else:
                raise NotImplementedError
        elif 'close' in skill[0]:
            string = 'close the '
            if 'fridge' in skill[0]:
                string += 'refrigerator'
            elif 'cab' in skill[0]:
                string += 'cabinet ' + skill[1][0].split('_')[-1]
            else:
                raise NotImplementedError
        elif 'place' in skill[0]:
            string = 'place at the '
            string = add_receptacle(string, skill)
        else:
            raise NotImplementedError
        
        language_skill_set.append(string)
    return language_skill_set



class EBHabEnv(gym.Env):
    def __init__(self, eval_set='train', exp_name='', down_sample_ratio=1.0, start_epi_index=0, resolution=600, recording=False):
        """
        Initialize the HabitatRearrange environment.
        """
        # load config
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        self.config = habitat.get_config(HABITAT_CONFIG_PATH)
        _add_sim_sensor_to_config(self.config, ThirdRGBSensorConfig())
        # set the dataset
        assert eval_set in ValidEvalSets
        OmegaConf.set_readonly(self.config, False)
        self.config.habitat.dataset.data_path = os.path.join(os.path.dirname(__file__), 'datasets/{}.pickle'.format(eval_set))
        self.config.habitat.simulator.agents.main_agent.sim_sensors.head_rgb_sensor.height = resolution
        self.config.habitat.simulator.agents.main_agent.sim_sensors.head_rgb_sensor.width = resolution
        self.resolution = resolution

        # modify config path to ease data loading
        self.dataset = make_dataset(self.config.habitat.dataset.type, config=self.config.habitat.dataset)

        # initilaize env
        self.env = habitat.gym.make_gym_from_config(self.config, self.dataset)
        self.observation_space = self.env.observation_space
        # action of LanguageRearangeEnv is discrete value from 0 to 69
        self.action_space = self.env.action_space

        # Episode tracking
        self.down_sample_ratio = down_sample_ratio
        self.number_of_episodes = self.env.number_of_episodes * down_sample_ratio
        self._reset = False
        self._current_episode_num = 0 
        while start_epi_index >= 1 and self._current_episode_num < start_epi_index:
            self.env.reset(return_info=False)
            self._current_episode_num += 1

        self._current_step = 0
        self._max_episode_steps = 30
        self._cur_invalid_actions = 0
        self._max_invalid_actions = 10
        self._episode_start_time = 0
        # is holding an object
        self.is_holding = False
        self.episode_log = []

        # init instruction and skill sets
        self.episode_language_instruction = ''
        self.episode_data = None
        self.skill_set = self.env.env.env._env.task.actions['pddl_hl_action']._action_datas
        self.language_skill_set = transform_action_to_natural_language(self.skill_set)

        # env feedback and image save
        # feedback verbosity, 0: concise, 1: verbose
        self.feedback_verbosity = 1
        self.log_path = 'running/eb_habitat/{}'.format(exp_name)
        # video recorder
        self.recording = recording
        self.episode_video = []
        
    def current_episode(self, all_info: bool = False):
        return self.env.current_episode(all_info)


    def reset(self, **kwargs):
        """
        Reset the environment for a new episode. The env will iterate over all the task data from the dataset
        Returns: observation
        """
        assert self._current_episode_num <= self.number_of_episodes
        obs, info = self.env.reset(return_info=True, **kwargs)
        logger.info('Episode {}: {}'.format(str(self._current_episode_num), str(self.current_episode())))
        self.episode_language_instruction = info['lang_goal']
        self.episode_data = self.dataset.episodes[self._current_episode_num]
        self._current_step = 0
        self._cur_invalid_actions = 0
        self._current_episode_num += 1
        self.is_holding = False
        self._reset = True
        self.episode_log = []
        if self.recording:
            self.episode_video = []
        self._episode_start_time = time.time()
        return obs

    def get_env_feedback(self, info):
        """
        Generate feedback message for the current step.
        Args:
            info (dict): Action execution information
        Returns:
            str: Descriptive message about step outcome
        """
        if info['was_prev_action_invalid']:
            env_feedback = 'Last action is invalid.'
            if 'pick' in info['action'] and self.feedback_verbosity:
                if self.is_holding:
                    env_feedback += ' Robot cannot pick any object when holding something. Please place the object before picking something.'
                else:
                    env_feedback += ' Robot cannot pick any object that is not near the robot. Navigate to other place to find the object.'
            elif 'place' in info['action'] and self.feedback_verbosity:
                if self.is_holding:
                    env_feedback += ' Robot cannot place any object that is not near the robot. Navigate to other place to find the object.'
                else:
                    env_feedback += ' Robot cannot place any object when not holding something. Please pick the object before place it.'
            elif 'open' in info['action'] and self.feedback_verbosity:
                env_feedback += " Check whether the receptacle is already open or the robot is not near the receptacle."
            elif 'close' in info['action'] and self.feedback_verbosity:
                env_feedback += " Check whether the receptacle is already closed or the robot is not near the receptacle."
        else:
            env_feedback = 'Last action executed successfully'
            if 'pick' in info['action'] and self.feedback_verbosity:
                self.is_holding = True
                env_feedback += ' and you are holding {}.'.format(info['action'].split('(')[0].split('_')[1])
            elif 'place' in info['action'] and self.feedback_verbosity:
                self.is_holding = False
                env_feedback += ' and you are holding nothing.'
            elif 'open' in info['action'] and self.feedback_verbosity:
                if 'fridge' in info['action']:
                    env_feedback += ' and now refrigerator is open.'
                elif 'cab' in info['action']:
                    env_feedback += ' and now cabinet {} is open.'.format(info['action'].split('(')[1].strip(')').split('_')[1])
                else:
                    raise NotImplementedError
            elif 'close' in info['action'] and self.feedback_verbosity:
                if 'fridge' in info['action']:
                    env_feedback += ' and now refrigerator is closed.'
                elif 'cab' in info['action']:
                    env_feedback += ' and now cabinet {} is closed.'.format(info['action'].split('(')[1].strip(')').split('_')[1])
                else:
                    raise NotImplementedError
            else:
                env_feedback += '.'
        
        # we don't use this info
        # env_feedback += ' The current task progress is {}.'.format(info['task_progress'])
        return env_feedback

    def step(self, action, reasoning='', **kwargs):
        """
        Execute a single environment step.
        Args:
            action (int): Index of action in action space
        Returns:
            tuple: (observation, reward, done, environment feedback)
        """
        assert self._reset, 'Reset env before stepping'
        self._current_step += 1
        obs, reward, done, info = self.env.step(action, **kwargs)
        if self.recording:
            self.episode_video.append(self.env.render("rgb_array"))

        if info['was_prev_action_invalid']:
            self._cur_invalid_actions += 1

        # if exceed the max step
        if self._current_step >= self._max_episode_steps or self._cur_invalid_actions >= self._max_invalid_actions:
            done = True
        # env feedback
        env_feedback = self.get_env_feedback(info)
        info['env_feedback'] = env_feedback
        info['env_step'] = self._current_step
        info['episode_elapsed_seconds'] = time.time() - self._episode_start_time,
        info['action_id'] = action
        info['action_description'] = self.language_skill_set[action]
        info['reasoning'] = reasoning
        info['instruction'] = self.episode_language_instruction
        info['last_action_success'] = 1 - float(info['was_prev_action_invalid'])
        info['task_success'] = info['predicate_task_success']
        if info['task_success']:
            info['task_progress'] = 1.0
        self.episode_log.append(info)
        return obs, reward, done, info

    def seed(self, seed=None):
        self.env.seed(seed)

    def save_image(self, obs, key='head_rgb'):
        """Save current agent observation as a PNG image."""
        folder = self.log_path + '/images/episode_{}'.format(self._current_episode_num)
        if not os.path.exists(folder):
            os.makedirs(folder)
        img = Image.fromarray(observations_to_image(obs, key))
        # time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        image_path = os.path.join(folder, 'episode_{}_step_{}.png'.format(self._current_episode_num, self._current_step)) #, time_stamp))
        img.save(image_path)
        return image_path

    def save_episode_log(self):
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        # time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        filename = 'episode_{}_step_{}.json'.format(self._current_episode_num, self._current_step) #, time_stamp)
        if len(self.episode_log):
            with open(os.path.join(self.log_path, filename), 'w', encoding='utf-8') as f:
                for item in self.episode_log:
                    json.dump(item, f, ensure_ascii=False)
                    f.write('\n')  
        
        if len(self.episode_video):
            folder = self.log_path + '/video'
            if not os.path.exists(folder):
                os.makedirs(folder)
            video_writer = imageio.get_writer(os.path.join(folder, 'video_episode_{}_steps_{}.mp4'.format(self._current_episode_num, self._current_step)), fps=30)
            for data in self.episode_video:
                video_writer.append_data(data)
            video_writer.close()



    def render(self, mode: str = "rgb"):
        return self.env.render(mode)

    def close(self) -> None:
        """Terminate the environment."""
        self.env.close()


if __name__ == '__main__':
    """
    Example usage of the EBHabEnv environment.
    Demonstrates environment interaction with random actions.
    """
    env = EBHabEnv(eval_set='base')
    obs = env.reset()
    print([(i, name) for i, name in enumerate(env.language_skill_set)])
    for _ in range(30):
        env.save_image(obs)
        action = int(input('action id: ')) #env.action_space.sample()
        if action in env.language_skill_set:
            action = env.language_skill_set.index(action)
        else:
            action = int(action)
            if action < 0:
                break

        obs_new, reward, done, info = env.step(action)
        print(reward, done, info)
        env.save_image(obs_new)
        if done:
            break
    env.close()
