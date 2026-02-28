"""
AI2THOR Environment for Household Robot Task Simulation

This module provides a custom OpenAI Gym environment for simulating household robot tasks
using the AI2THOR framework. It supports various object interactions and task scenarios.

Dependencies:
- ai2thor
- gym
- numpy
- PIL
"""

import os
import time
import gym
import json
import numpy as np
from PIL import Image 

# Import custom modules
import embodiedbench.envs.eb_alfred.utils as utils
from embodiedbench.envs.eb_alfred.utils import alfred_objs, alfred_open_obj, alfred_pick_obj, alfred_slice_obj, alfred_open_obj, alfred_toggle_obj, alfred_recep
from embodiedbench.envs.eb_alfred.thor_connector import ThorConnector
from embodiedbench.envs.eb_alfred.data.preprocess import Dataset
from embodiedbench.envs.eb_alfred.gen import constants
from embodiedbench.main import logger

# global information
X_DISPLAY = '1'
ALFRED_SPLIT_PATH = os.path.join(os.path.dirname(__file__), 'data/splits/splits.json')
ALFRED_REWARD_PATH = os.path.join(os.path.dirname(__file__), 'models/config/rewards.json')
ALFRED_DATASET_PATH = os.path.join(os.path.dirname(__file__), 'data/json_2.1.0')
ValidEvalSets = [
    'base', 'common_sense', 'complex_instruction', 'spatial', 
    'visual_appearance', 'long_horizon'
    ]


def get_global_action_space():
    """
    Generate a comprehensive action space for the environment.
    
    Returns:
        list: A list of supported action strings for various object interactions
    """
    action_space = []

    # Low-level navigation actions (same as eb_navigation)
    action_space.extend([
        "Move forward by 0.25",
        "Move backward by 0.25",
        "Move rightward by 0.25",
        "Move leftward by 0.25",
        "Rotate to the right by 90 degrees.",
        "Rotate to the left by 90 degrees.",
        "Tilt the camera upward by 30 degrees.",
        "Tilt the camera downward by 30 degrees.",
    ])
    
    # Generate find actions for all objects
    findable_objs = alfred_objs
    action_space.extend([f"find a {obj}" for obj in findable_objs])
    
    # Generate pickup, putdown, and drop actions
    pickup_objs = alfred_pick_obj
    for obj in pickup_objs:
        action_space.extend([
            f"pick up the {obj}", 
        ])
    
    action_space.extend([
            f"put down the object in hand", 
            f"drop the object in hand"
        ])
    
    # Generate open/close actions
    open_objs = alfred_open_obj
    for obj in open_objs:
        action_space.extend([
            f"open the {obj}", 
            f"close the {obj}"
        ])
    
    # Generate toggle actions
    turn_on_objs = alfred_toggle_obj
    for obj in turn_on_objs:
        action_space.extend([
            f"turn on the {obj}", 
            f"turn off the {obj}"
        ])
    
    # Generate slice actions
    slice_objs = alfred_slice_obj
    action_space.extend([f"slice the {obj}" for obj in slice_objs])
    
    return action_space


class EBAlfEnv(gym.Env):
    """
    Custom OpenAI Gym environment for simulating household robot tasks.
    
    Attributes:
        env (ThorConnector): Interface for AI2THOR interactions
        action_space (gym.spaces.Discrete): Discrete action space 
        language_skill_set (list): Readable action descriptions
    """
    def __init__(self, eval_set='base', exp_name='', down_sample_ratio=1.0, selected_indexes=[], detection_box=False, resolution=500):
        """
        Initialize the AI2THOR environment.
        """
        super().__init__()
        self.data_path = ALFRED_SPLIT_PATH
        self.reward_config_path = ALFRED_REWARD_PATH
        self.resolution = resolution
        self.env = ThorConnector(x_display=X_DISPLAY, player_screen_height=resolution, player_screen_width=resolution)

        # load dataset
        assert eval_set in ValidEvalSets
        self.down_sample_ratio = down_sample_ratio
        self.dataset = self._load_dataset(eval_set)
        if len(selected_indexes):
            self.dataset = [self.dataset[i] for i in selected_indexes]
        
        # Episode tracking
        self.number_of_episodes = len(self.dataset)
        self._reset = False
        self._current_episode_num = 0
        self.selected_indexes = selected_indexes
        self._initial_episode_num = 0
        self._current_step = 0
        self._max_episode_steps = 30
        self._cur_invalid_actions = 0
        self._max_invalid_actions = 10
        self._episode_start_time = 0
        self.episode_log = []
        
        # Task-related attributes
        self.episode_language_instruction = ''
        self.episode_data = None
        # Initialize action space
        self.language_skill_set = None
        self.action_space = None

        # env feedback and image save
        # feedback verbosity, 0: concise, 1: verbose
        self.feedback_verbosity = 0
        self.log_path = 'running/eb_alfred/{}'.format(exp_name)

        self.detection = detection_box # add detection in image
        self.name_to_id_dict = None
        self.id_to_name_dict = None
        self.language_skill_set = get_global_action_space()
        self.action_space = gym.spaces.Discrete(len(self.language_skill_set))


    def generate_additional_action_space(self):
        """
        Generate additional actions for receptacles with multiple instances
        """
        # Generate pickup, putdown, and drop actions
        add_findable_objs = []
        add_pickable_objs = []

        recept_obj_dict = {}
        pickable_obj_dict = {}
        name_to_id_dict = {}
        for obj in self.env.last_event.metadata['objects']:
            if obj['receptacle']:
                if obj['objectType'] in recept_obj_dict:
                    recept_obj_dict[obj['objectType']].append(obj['objectId']) 
                else:
                    recept_obj_dict[obj['objectType']] = [obj['objectId']]
            elif obj['pickupable']:
                if obj['objectType'] in pickable_obj_dict:
                    pickable_obj_dict[obj['objectType']].append(obj['objectId'])
                else:
                    pickable_obj_dict[obj['objectType']] = [obj['objectId']]

    
        # store the mapping for object with multiple instances
        for key in recept_obj_dict:
            if len(recept_obj_dict[key]) >= 2:
                for i in range(len(recept_obj_dict[key])):
                    if i == 0:
                        name_to_id_dict[key] = recept_obj_dict[key][i]
                    else:
                        name_to_id_dict[key + '_{}'.format(i+1)] = recept_obj_dict[key][i]
                        add_findable_objs.append(key + '_{}'.format(i+1))
        
        for key in pickable_obj_dict:
            if len(pickable_obj_dict[key]) >= 2:
                for i in range(len(pickable_obj_dict[key])):
                    if i == 0:
                        name_to_id_dict[key] = pickable_obj_dict[key][i]
                    else:
                        name_to_id_dict[key + '_{}'.format(i+1)] = pickable_obj_dict[key][i]
                        add_pickable_objs.append(key + '_{}'.format(i+1))

        id_to_name_dict = {}
        for key in name_to_id_dict:
            id_to_name_dict[name_to_id_dict[key]] = key

        # Generate find actions for additional objects
        add_findable_objs = sorted(list(set(add_findable_objs)))
        add_pickable_objs = sorted(list(set(add_pickable_objs)))
        action_space = [f"find a {obj}" for obj in add_findable_objs]
        for obj in add_findable_objs:
            if obj.split('_')[0] in alfred_open_obj:
                action_space.extend([
                    f"open the {obj}", 
                    f"close the {obj}"
                ])
        for obj in add_pickable_objs:
            if obj.split('_')[0] in alfred_pick_obj:
                action_space.extend([
                    f"find a {obj}", 
                ])

        self.language_skill_set = get_global_action_space() + action_space
        self.action_space = gym.spaces.Discrete(len(self.language_skill_set))
        self.name_to_id_dict = name_to_id_dict
        self.id_to_name_dict = id_to_name_dict

    def _load_dataset(self, eval_set):
        with open(self.data_path) as f:
            dataset_split = json.load(f)
        dataset = dataset_split[eval_set]
        if 0 <= self.down_sample_ratio < 1:
            select_every = round(1 / self.down_sample_ratio)
            dataset = dataset[0:len(dataset):select_every]
        return dataset


    def current_episode(self):
        """Return current episode"""
        res = None
        try:
            res = utils.load_task_json(self.dataset[self._current_episode_num])
        except:
            print("episode failed to load trying next episode")
            self.current_episode_num += 1
            self.current_episode()
        return res
    
    def _reset_controller(self, task):
        """Restore scene from a task name and replace instruction"""
        traj_data = utils.load_task_json(task)
        traj_data['turk_annotations']['anns'][task['repeat_idx']]['task_desc'] = task["instruction"] 
        self.episode_data = traj_data
        args_dict = {'data': ALFRED_DATASET_PATH, 'pframe': 300, 'fast_epoch': False,
                    'use_templated_goals': False, 'dout': 'exp/model', 'pp_folder': 'pp',
                    'reward_config': self.reward_config_path, 'max_steps': 1000}
        model_args = utils.dotdict(args_dict)
        
        # Extract scene configuration
        scene_num = traj_data['scene']['scene_num']
        object_poses = traj_data['scene']['object_poses']
        dirty_and_empty = traj_data['scene']['dirty_and_empty']
        object_toggles = traj_data['scene']['object_toggles']

        scene_name = 'FloorPlan%d' % scene_num
        self.episode_language_instruction = task["instruction"]
        # Restore scene configuration
        logger.info(f"Restoring scene {scene_name}...")
        self.env.reset(scene_name)
        self.env.restore_scene(object_poses, object_toggles, dirty_and_empty)
        if traj_data['scene']['init_action']['action'] == 'TeleportFull':
            del traj_data['scene']['init_action']["rotateOnTeleport"]
            traj_data['scene']['init_action']["standing"] = True
        self.env.step(dict(traj_data['scene']['init_action']))
        self.env.set_task(traj_data, model_args, reward_type='dense', max_episode_length=self._max_episode_steps)
        #############################
        self.generate_additional_action_space()

    def reset(self):
        """
        Reset the environment for a new episode.
        
        Returns:
            observation
        """
        assert self._current_episode_num < self.number_of_episodes
        self._reset_controller(self.dataset[self._current_episode_num])
        self._current_step = 0
        self._cur_invalid_actions = 0
        self._current_episode_num += 1
        obs = {
            'head_rgb': self.env.last_event.frame,
        }
        self._reset = True
        self.episode_log = []
        self._episode_start_time = time.time()
        return obs


    def step(self, action, reasoning=''):
        """
        Execute a single environment step.
        Args:
            action (int): Index of action in action space
        Returns:
            tuple: (observation, reward, done, environment feedback)
        """
        assert self._reset, 'Reset env before stepping'
        info = {}
        self._current_step += 1
        if type(action) == int:
            lang_action  = self.language_skill_set[action]
        elif type(action) == str:
            lang_action  = action
        else:
            raise NotImplementedError

        if 'find' in lang_action or 'open' in lang_action or 'close' in lang_action:
            lang_action_split = lang_action.split(' ')
            if (self.name_to_id_dict is not None) and lang_action_split[-1] in self.name_to_id_dict: # multiple instances
                lang_action = ' '.join(lang_action_split[:-1] + [self.name_to_id_dict[lang_action_split[-1]]])

        event = self.env.llm_skill_interact(lang_action)  # 最底层的Thorenv执行动作
        if not event['success']:
            self._cur_invalid_actions += 1
        
        ## test calculate reward
        reward, done = self.env.get_transition_reward()
        subgoal_met = self.env.get_goal_conditions_met()
        info['task_success'] = float(self.env.get_goal_satisfied())
        info['task_progress'] = subgoal_met[0] / subgoal_met[1]

        obs = {
            'head_rgb': self.env.last_event.frame,
        }
        # if exceed the maximum episode steps or the goal is achieved
        if self._current_step >= self._max_episode_steps or info['task_success'] or self._cur_invalid_actions >= self._max_invalid_actions:
            done = True
        
        # add env feedback
        info['instruction'] = self.episode_language_instruction
        info['env_step'] = self._current_step
        info['env_feedback'] = self.get_env_feedback(event)
        info['episode_elapsed_seconds'] = time.time() - self._episode_start_time
        info['last_action_success'] = float(event['success'])
        info['object_states'] = {
                                    "cooled_objects" : self.env.cooled_objects,
                                    "heated_objects" : self.env.heated_objects,
                                    "cleaned_objects" : self.env.cleaned_objects,
                                    "visible_objs": [obj['objectType'] for obj in self.env.last_event.metadata['objects'] if obj['visible']]
                                }
        info['action_id'] = action
        info['action_description'] = self.language_skill_set[action] if type(action) == int else action
        info['reasoning'] = reasoning
        self.episode_log.append(info)
        return obs, reward, done, info
    
    def get_env_feedback(self, info):
        """
        Generate feedback message for the current step.
        Args:
            info (dict): Action execution information
        Returns:
            str: Descriptive message about step outcome
        """
        msg = ''
        if info["success"]:
            msg += f"Last action executed successfully."
        else:
            if 'is not visible' in info['message'] and '|' in info['message']:
                recep_id = info['message'].split('because it is in ')[1].split('. Note')[0]
                if recep_id not in self.id_to_name_dict:
                    pos = recep_id.split('|')[0]
                else:
                    pos = self.id_to_name_dict[recep_id]
                message = info['message'].split(recep_id)[0] + pos + '. Go there to pick the object instead.'
            else:
                message = info['message']
            msg += f"Last action is invalid. {message}"
        return msg
    
    def seed(self, seed=None):
        self.env.random_initilize(seed)

    def save_image(self, *args, **kwargs):
        """Save current agent view as a PNG image."""
        episode_idx = self._current_episode_num if not len(self.selected_indexes) else self.selected_indexes[self._current_episode_num - 1] + 1
        
        folder = self.log_path + '/images/episode_{}'.format(episode_idx)
        if not os.path.exists(folder):
            os.makedirs(folder)
        img = Image.fromarray(self.env.last_event.frame)
        if self.detection:
            img = utils.draw_boxes(img, self.env.last_event.instance_detections2D, name_translation=self.id_to_name_dict)

        # time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        image_path = os.path.join(folder, 'episode_{}_step_{}.png'.format(episode_idx, self._current_step)) #, time_stamp))
        img.save(image_path)
        return image_path

    def save_episode_log(self):
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        # time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        episode_idx = self._current_episode_num if not len(self.selected_indexes) else self.selected_indexes[self._current_episode_num - 1] + 1
        filename = 'episode_{}_step_{}.json'.format(episode_idx, self._current_step) #, time_stamp)
        if len(self.episode_log):
            with open(os.path.join(self.log_path, filename), 'w') as f:
                for item in self.episode_log:
                    if 'object_states' in item:
                        item.pop('object_states')
                    try:
                        json.dump(item, f, ensure_ascii=False)
                    except:
                        import pdb;pdb.set_trace()
                    f.write('\n')  


    def close(self):
        """Terminate the environment."""
        self.env.stop()

    

if __name__ == "__main__":
    """
    Example usage of the EBAlfEnv environment.
    Demonstrates environment interaction with random actions.
    """
    env = EBAlfEnv(eval_set='base', down_sample_ratio=1.0, selected_indexes=[])
    env.reset()
    print([(i, name) for i, name in enumerate(env.language_skill_set)])
    for _ in range(30):
        # Select  action
        action = int(input('action id: ')) #env.action_space.sample()
        if action in env.language_skill_set:
            action = env.language_skill_set.index(action)
        else:
            action = int(action)
            if action < 0:
                break
        
        print(env.language_skill_set[action])
        
        # Execute action
        obs, reward, done, info = env.step(action)
        print(reward, done, info)
        # Optional rendering and image saving
        env.save_image()
        if done:
            break
    env.close()

