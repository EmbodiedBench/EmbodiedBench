import ai2thor.controller
import gym
import numpy as np
import time
from PIL import Image
import json
import os
import sys
import math
from ai2thor.platform import CloudRendering
from embodiedbench.envs.eb_navigation.utils import draw_target_box, draw_boxes
from embodiedbench.main import logger
import copy

SUCCESS_THRESHOLD = 1

ValidEvalSets = [
    'base', 'common_sense', 'complex_instruction', 'visual_appearance', 'long_horizon'
]


DISCRETE_SKILLSET = [
    "Move forward by 0.25",
    "Move backward by 0.25",
    "Move rightward by 0.25",
    "Move leftward by 0.25",
    "Rotate to the right by 90 degrees.",
    "Rotate to the left by 90 degrees.",
    "Tilt the camera upward by 30 degrees.",
    "Tilt the camera downward by 30 degrees.",
    # "Crouch to be lower",
    # "Stand to be taller",
    # "Complete the current task."
]

# "Move forward by 0.25 meter.",
# "Move backward by 0.25 meter.",
# "Move right by 0.25 meter.",
# "Move left by 0.25 meter.",


class EBNavigationEnv(gym.Env):
    def __init__(self, eval_set='base', exp_name='test_base', down_sample_ratio=1.0, fov = 100, multiview = False, boundingbox = False, multistep = False,  resolution = 600, selected_indexes =[]):
        """
        A wrapper for AI2-THOR ManipulaTHOR environment.

        :param config: Dictionary containing initialization parameters for the controller.
        """
        self.resolution = resolution
        self.config = {
            "agentMode": "default",
            "gridSize": 0.1,
            "visibilityDistance": 10,
            "renderDepthImage": True,
            "renderInstanceSegmentation": True,
            "width": self.resolution,
            "height": self.resolution,
            "fieldOfView": fov,
            "platform": CloudRendering
        }
        self.env = ai2thor.controller.Controller(**self.config)

        # load dataset
        assert eval_set in ValidEvalSets
        self.down_sample_ratio = down_sample_ratio
        self.data_path = os.path.join(os.path.dirname(__file__), f"datasets/{eval_set}.json")
        self.dataset = self._load_dataset(eval_set)
        if len(selected_indexes):
            self.dataset = [self.dataset[i] for i in selected_indexes]

        self.selected_indexes = selected_indexes

        # Episode tracking
        self.number_of_episodes = len(self.dataset)
        self._reset = False
        self._current_episode_num = 0
        self._current_step = 0
        self._max_episode_steps = 20
        self._episode_start_time = 0
        self.is_holding = False
        self.episode_log = []
        self.episode_language_instruction = ""
        self.episode_data = None

        self._last_event = None

        self.standing = True

        # set action space
        self.language_skill_set = DISCRETE_SKILLSET
        self.action_space = gym.spaces.Discrete(len(self.language_skill_set))

        # set log and verbosity(0 for concise)
        self.feedback_verbosity = 0
        self.log_path = 'running/eb_nav/{}'.format(exp_name)

        self.multiview = multiview
        self.boundingbox = boundingbox
        self.multistep = multistep
        self.img_paths = []

    def _load_dataset(self, eval_set):
        with open(self.data_path) as f:
            dataset_split = json.load(f)
        dataset = dataset_split["tasks"]
        if 0 <= self.down_sample_ratio < 1:
            select_every = round(1 / self.down_sample_ratio)
            dataset = dataset[0:len(dataset):select_every]
        return dataset

    def reset(self, **kwargs):
        """
        Reset the environment.

        :param scene: Optionally set the scene for reset.
        :return: The initial observation.
        """
        # self.save_episode_log()
        assert self._current_episode_num < self.number_of_episodes

        # start reset environment 
        traj_data = self.dataset[self._current_episode_num]
        self.episode_data = traj_data
        self.episode_language_instruction = traj_data["instruction"]

        scene_name = traj_data["scene"]
        logger.info(f"Restoring scene {scene_name}...")
        self._last_event = self.env.reset(
            scene=scene_name
        )

        if self.multiview:
            event = self.env.step(action="GetMapViewCameraProperties", raise_for_failure=True)
            pose = copy.deepcopy(event.metadata["actionReturn"])
            pose["orthographic"] = True

            # add the camera to the scene
            self.env.step(
                action="AddThirdPartyCamera",
                **pose,
                skyboxColor="white",
                raise_for_failure=True,
            )

        pose = traj_data["agentPose"]
        self.env.step(
            action="Teleport",
            position={
                "x": pose["position"]["x"],
                "y": pose["position"]["y"],
                "z": pose["position"]["z"]
            },
            rotation={
                "x": 0,
                "y": pose["rotation"],
                "z": 0
            },
            horizon=pose["horizon"],
            standing=True
        )

        # finish reset environment 
        # reset episode information
        self._current_episode_num += 1
        self._current_step = 0

        self.standing = True
        obs = {
            'head_rgb': self.env.last_event.frame
        }
        self._reset = True
        self.episode_log = []
        self._episode_start_time = time.time()

        self.img_paths = []

        return obs
    
    def discrete_action_mapper(self, action_index):
        """
        Maps a discrete action index to the corresponding iTHOR environment action.

        Parameters:
            env: The AI2-THOR environment object.
            action_index: An integer representing the action index.

        Raises:
            ValueError: If the action index is invalid.
        """

        if action_index == 0:  # Move forward by 0.25 meter
            self._last_event = self.env.step(action="MoveAhead", moveMagnitude=0.25)
        elif action_index == 1:  # Move backward by 0.25 meter
            self._last_event = self.env.step(action="MoveBack", moveMagnitude=0.25)
        elif action_index == 2:  # Move right by 0.25 meter
            self._last_event = self.env.step(action="MoveRight", moveMagnitude=0.25)
        elif action_index == 3:  # Move left by 0.25 meter
            self._last_event = self.env.step(action="MoveLeft", moveMagnitude=0.25)
        elif action_index == 4:  # Rotate clockwise by 45 degrees
            self._last_event = self.env.step(action="RotateRight", degrees=90)
        elif action_index == 5:  # Rotate counterclockwise by 45 degrees
            self._last_event = self.env.step(action="RotateLeft", degrees=90)
        elif action_index == 6:  # Tilt the camera upward by 30 degrees
            self._last_event = self.env.step(action="LookUp", degrees=30)
        elif action_index == 7:  # Tilt the camera downward by 30 degrees
            self._last_event = self.env.step(action="LookDown", degrees=30)
        # elif action_index == 8:  # Crouch to be lower
        #     self._last_event = self.env.step(action="Crouch")
        #     self.standing = False
        # elif action_index == 9:  # Stand to be taller
        #     self._last_event = self.env.step(action="Stand")
        #     self.standing = True
        # elif action_index == 8:  # Complete the current task
        #     self._last_event = self.env.step(action="Done")
        else:
            print(f"Invalid action index: {action_index}")

    def _mark_action_failed(self, action_name, message):
        if self.env.last_event is None:
            return

        self.env.last_event.metadata['lastActionSuccess'] = False
        self.env.last_event.metadata['errorMessage'] = message
        self.env.last_event.metadata['lastAction'] = action_name
        self._last_event = self.env.last_event

    def named_action_mapper(self, action_name):
        normalized = str(action_name).strip()
        normalized_no_period = normalized[:-1] if normalized.endswith('.') else normalized

        if normalized_no_period == "Move forward by 0.25" or normalized in ("MoveAhead", "MoveAhead_25"):
            self._last_event = self.env.step(action="MoveAhead", moveMagnitude=0.25)
        elif normalized_no_period == "Move backward by 0.25" or normalized in ("MoveBack", "MoveBack_25"):
            self._last_event = self.env.step(action="MoveBack", moveMagnitude=0.25)
        elif normalized_no_period == "Move rightward by 0.25" or normalized in ("MoveRight", "MoveRight_25"):
            self._last_event = self.env.step(action="MoveRight", moveMagnitude=0.25)
        elif normalized_no_period == "Move leftward by 0.25" or normalized in ("MoveLeft", "MoveLeft_25"):
            self._last_event = self.env.step(action="MoveLeft", moveMagnitude=0.25)
        elif normalized_no_period == "Rotate to the right by 90 degrees" or normalized in ("RotateRight", "RotateRight_90"):
            self._last_event = self.env.step(action="RotateRight", degrees=90)
        elif normalized_no_period == "Rotate to the left by 90 degrees" or normalized in ("RotateLeft", "RotateLeft_90"):
            self._last_event = self.env.step(action="RotateLeft", degrees=90)
        elif normalized_no_period == "Tilt the camera upward by 30 degrees" or normalized == "LookUp":
            self._last_event = self.env.step(action="LookUp", degrees=30)
        elif normalized_no_period == "Tilt the camera downward by 30 degrees" or normalized == "LookDown":
            self._last_event = self.env.step(action="LookDown", degrees=30)
        elif normalized_no_period == "Tilt the camera upward by 15 degrees" or normalized == "LookUp_15":
            self._last_event = self.env.step(action="LookUp", degrees=15)
        elif normalized_no_period == "Tilt the camera downward by 15 degrees" or normalized == "LookDown_15":
            self._last_event = self.env.step(action="LookDown", degrees=15)
        else:
            self._mark_action_failed(normalized, f"Unsupported navigation action: {normalized}")

    def measure_success(self):
        # success measurement
        agent_position = self.env.last_event.metadata["agent"]["position"]
        target_object_id = self.episode_data["targetObjectIds"]
        target_position = self.episode_data["target_position"]

        # for obj in self.env.last_event.metadata["objects"]:
        #     if obj["objectId"] == target_object_id:
        #         target_position = obj["position"]
        #         break

        dist = math.sqrt(
            (agent_position["x"] - target_position["x"])**2 +
            (agent_position["z"] - target_position["z"])**2
        )
        success = (dist <= SUCCESS_THRESHOLD)
        return float(success), dist

        

    def step(self, action, reasoning=''):
        """
        Perform an action in the environment.

        :param action: The name of the action to perform.
        :param kwargs: Additional parameters for the action.
        :return: Event.
        """

        assert self._reset, 'Reset env before stepping'
        info = {}

        self._current_step += 1
        action_id = -3
        action_description = str(action)

        if isinstance(action, int):
            action_id = action
            if 0 <= action < len(self.language_skill_set):
                action_description = self.language_skill_set[action]
                self.discrete_action_mapper(action)
            else:
                self._mark_action_failed(
                    action_description,
                    f"Invalid discrete action id: {action}",
                )
        elif isinstance(action, str):
            self.named_action_mapper(action)
        else:
            self._mark_action_failed(
                action_description,
                f"Unsupported action type: {type(action).__name__}",
            )

        obs = {
                    'head_rgb': self.env.last_event.frame,
                }
        reward, distance = self.measure_success()
        done = bool(reward > 0 or self._current_step >= self._max_episode_steps)

        ## test calculate reward
        info['distance'] = distance
        info['env_feedback'] = self.get_env_feedback(self._last_event, requested_action=action_description)
        info['reasoning'] = reasoning
        # info['reflection'] = reasoning['reasoning_and_reflection']
        # info['plan'] = reasoning['language_plan']
        info['instruction'] = self.episode_language_instruction
        info['env_step'] = self._current_step
        info['episode_elapsed_seconds'] = time.time() - self._episode_start_time
        info['task_success'] = reward
        info['last_action_success'] = self.env.last_event.metadata['lastActionSuccess']
        info['action_id'] = action_id if isinstance(action, int) else -3
        info['action_description'] = action_description
        # info['reasoning'] = reasoning

        self.episode_log.append(info)

        return obs, reward, done, info
        
    def get_env_feedback(self, event, requested_action=None):
        """
        To extract relevant information from the event to construct a feedback dictionary.

        :param event: self._last_event
        :return: A dictionary containing structured feedback.
        """
        if self.feedback_verbosity == 1:
            feedback = {
                "lastActionSuccess": event.metadata.get("lastActionSuccess", None),
                "errorMessage": event.metadata.get("errorMessage", None),
                "lastAction": event.metadata.get("lastAction", None),

                "agent": {
                    "position": event.metadata.get("agent", {}).get("position", {}),
                    "rotation": event.metadata.get("agent", {}).get("rotation", {}),
                    "is_standing": self.standing
                }
            }
        else:
            # Does not provide the specific reason why the action fails if so
            feedback = {
                "lastActionSuccess": event.metadata.get("lastActionSuccess", None),
                "lastAction": event.metadata.get("lastAction", None),
                "errorMessage": event.metadata.get("errorMessage", None),

                "agent": {
                    "is_standing": self.standing
                }
            }

        msg = ''
        action_label = requested_action or feedback['lastAction']
        if feedback["lastActionSuccess"]:
            msg += f"Last action {action_label} executed successfully."
        else:
            msg += f"Last action {action_label} is invalid. {feedback['errorMessage']}"
        return msg

    def seed(self, seed=None):
        self.env.random_initilize(seed)


    def save_image(self, *args, **kwargs):
        """Save current agent view as a PNG image."""
        episode_idx = self._current_episode_num if not len(self.selected_indexes) else self.selected_indexes[self._current_episode_num - 1] + 1

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        if self.multiview:
            img1 = Image.fromarray(self.env.last_event.frame)
            img2 = Image.fromarray(self.env.last_event.third_party_camera_frames[-1])
            time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            # image_path = 'episode_{}_step_{}_{}.png'.format(self._current_episode_num, self._current_step, time_stamp)
            image_path1 = os.path.join(self.log_path, 'episode_{}_step_{}_{}_front.png'.format(episode_idx, self._current_step, time_stamp))
            image_path2 = os.path.join(self.log_path, 'episode_{}_step_{}_{}_top.png'.format(episode_idx, self._current_step, time_stamp))
            img1.save(image_path1)
            img2.save(image_path2)
            return [image_path1, image_path2]
        
        elif self.multistep:
            
            img = Image.fromarray(self.env.last_event.frame)
            time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            # image_path = 'episode_{}_step_{}_{}.png'.format(self._current_episode_num, self._current_step, time_stamp)
            image_path = os.path.join(self.log_path, 'episode_{}_step_{}_{}_front.png'.format(episode_idx, self._current_step, time_stamp))
            img.save(image_path)
            self.img_paths.append(image_path)
            if self._current_step<3:
                return self.img_paths
            else:
                return self.img_paths[-3:]

        else:
            if not self.boundingbox:
                img = Image.fromarray(self.env.last_event.frame)
                time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                # image_path = 'episode_{}_step_{}_{}.png'.format(self._current_episode_num, self._current_step, time_stamp)
                image_path = os.path.join(self.log_path, 'episode_{}_step_{}_{}_front.png'.format(episode_idx, self._current_step, time_stamp))
                img.save(image_path)
                return image_path
            else:
                img = Image.fromarray(self.env.last_event.frame)
                time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                # image_path = 'episode_{}_step_{}_{}.png'.format(self._current_episode_num, self._current_step, time_stamp)
                image_path = os.path.join(self.log_path, 'episode_{}_step_{}_{}_front_bb.png'.format(episode_idx, self._current_step, time_stamp))
                # if self.target_only:
                # draw_target_box(img, self.env.last_event.instance_detections2D, self.episode_data["targetObjectIds"], image_path)
                # else:
                draw_boxes(img,self.env.last_event.instance_detections2D, image_path)
                # img.save(image_path)
                return image_path

    def save_episode_log_per_step(self, flag):

        episode_idx = self._current_episode_num if not len(self.selected_indexes) else self.selected_indexes[self._current_episode_num - 1] + 1

        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        time_stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        filename = 'episode_{}.json'.format(episode_idx)
        if len(self.episode_log):
            with open(os.path.join(self.log_path, filename), 'a') as f:
                if flag == 1:
                    f.write('\n\n')
                    for item in self.episode_log:
                        if 'object_states' in item:
                            item.pop('object_states')
                        try:
                            json.dump(item, f, ensure_ascii=False)
                        except:
                            import pdb;pdb.set_trace()
                        f.write('\n') 
                else:
                    for item in self.episode_log:
                        if 'object_states' in item:
                            item.pop('object_states')
                        try:
                            json.dump(item, f, ensure_ascii=False)
                        except:
                            import pdb;pdb.set_trace()
                        f.write('\n') 

    def save_episode_log(self):
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        episode_idx = self._current_episode_num if not len(self.selected_indexes) else self.selected_indexes[self._current_episode_num - 1] + 1
        filename = 'episode_{}_step_{}.json'.format(episode_idx, self._current_step)
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
        """Close the environment."""
        self.env.stop()


if __name__ == "__main__":

    env = EBNavigationEnv("base")
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
        obs, reward, done, info = env.step(action, "")
        print(reward, done, info)
        # Optional rendering and image saving
        env.save_image()
        if done:
            break
    env.close()
