# import torch
import re
import os
import numpy as np
import cv2
import json
# import lmdeploy
# from lmdeploy import pipeline, GenerationConfig, PytorchEngineConfig
from openai import OpenAI
from embodiedbench.planner.planner_config.generation_guide import llm_generation_guide, vlm_generation_guide
from embodiedbench.planner.planner_utils import local_image_to_data_url, truncate_message_prompts
# from embodiedbench.planner.eb_navigation.RemoteModel_claude import RemoteModel
from embodiedbench.planner.remote_model import RemoteModel
from embodiedbench.planner.custom_model import CustomModel
from embodiedbench.evaluator.config.visual_icl_examples.eb_navigation.ebnav_visual_icl import create_example_json_list
from embodiedbench.planner.planner_utils import template, template_lang
from embodiedbench.main import logger

template = template
template_lang = template_lang

MESSAGE_WINDOW_LEN = 5

class EBNavigationPlanner():
    def __init__(self, model_name = '', model_type = 'remote', actions = [], system_prompt = '', examples = '', n_shot=1, obs_key='head_rgb', chat_history=False, language_only=False, multiview = False, multistep = False, visual_icl = False, tp=1, truncate=False, kwargs={}):
        self.model_name = model_name
        self.model_type = model_type
        self.obs_key = obs_key
        self.system_prompt = system_prompt
        self.n_shot = n_shot
        self.chat_history = chat_history # whether to includ all the chat history for prompting
        self.truncate = truncate # whether to truncate message history when chat_history is True
        self.set_actions(actions)
        self.planner_steps = 0
        self.output_json_error = 0

        self.kwargs = kwargs
        self.action_key = kwargs.pop('action_key', 'action_id')

        self.multiview = multiview
        self.multistep = multistep
        self.visual_icl = visual_icl  # 是否在提示中加入视觉 in-context learning 示例

        if not self.visual_icl:
            self.examples = examples[:n_shot]
            self.language_only = language_only
        else:
            self.examples = []
            self.language_only = False
            if language_only:
                self.icl_text_only = True
            else:
                self.icl_text_only = False


        self.first_prompt = f'''To achieve the task, 1. Reason about the current visual state and your final goal, and 2. Reflect on the effect of previous actions. 3. Summarize how you learn from the Strategy and Examples provided \
\nAim for about 1-2 actions in this step. !!!Notice: you cannot assess the situation until the whole plan in this planning step is finished executed, so plan accordingly.\
\nAt last, output the action id(s) (0 ~ {len(self.actions)-1}) from the available actions to execute. 

The input given to you is {'an first person view observation' if not self.multistep else 'latest 3 steps of the first person view observations'} {'and a overhead view of the house where the silver circle represents where you locates (Notice:The part hanging on the outside is your arm, and it is on your right side)' if self.multiview else ''}. Plan accordingly based on the visual observation.

You are supposed to output in JSON.{template_lang if self.language_only else template}'''

        self.following_prompt = f'''To achieve the task, 1. Reason about the current visual state and your final goal, and 2. Reflect on the effect of previous actions. 3. Summarize how you learn from the Strategy and Examples provided \
\nAim for about 5-6 actions in this step to be closer to the target object. !!!Notice: you cannot assess the situation until the whole plan in this planning step is finished executed, so plan accordingly.\
\nAt last, output the action id(s) (0 ~ {len(self.actions)-1}) from the available actions to execute. 

The input given to you is {'an first person view observation' if not self.multistep else 'latest 3 steps of the first person view observations'} {'and a overhead view of the house where the silver circle represents where you locates (Notice:The part hanging on the outside is your arm, and it is on your right side)' if self.multiview else ''}. Plan accordingly based on the visual observation.

You are supposed to output in JSON.{template_lang if self.language_only else template}'''

        
        if model_type == 'custom':
            self.model = CustomModel(model_name, language_only)
        else:
            self.model = RemoteModel(model_name, model_type, language_only, tp=tp)

    
    def set_actions(self, actions):
        self.actions = actions
        self.available_action_str = self.get_availabel_action_prompt(actions)

    def get_availabel_action_prompt(self, available_actions):
        available_action_str = ''
        for i in range(len(available_actions)):
            available_action_str += '\naction id ' + str(i) + ': ' + str(available_actions[i]) 
            if i < len(available_actions) - 1:
                available_action_str += ', '
        return available_action_str


    def process_prompt(self, user_instruction, prev_act_feedback=[]):

        user_instruction = user_instruction.rstrip('.')

        if len(prev_act_feedback) == 0:
            if self.n_shot >= 1:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '\n\n'.join([f'## Task Execution Example {i}: \n {x}' for i,x in enumerate(self.examples)])) 
            else:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '')

            prompt += f'\n\n## Now the human instruction is: {user_instruction}.'

            prompt += self.first_prompt
     
        elif self.chat_history:

            # This is to support the sliding window feature
            if self.n_shot >= 1:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '\n\n'.join([f'## Task Execution Example  {i}: \n {x}' for i,x in enumerate(self.examples)])) 
            else:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '')

            prompt += f'\n\n## The human instruction is: {user_instruction}.'

            prompt += '\n\n The action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                prompt += '\n Step {}, action id {}, {}, env feedback: {}'.format(i, action_feedback[0], self.actions[action_feedback[0]], action_feedback[1])

            prompt += f"\n\n{self.following_prompt}"

        else:
            if self.n_shot >= 1:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '\n\n'.join([f'## Task Execution Example  {i}: \n {x}' for i,x in enumerate(self.examples)])) 
            else:
                prompt = self.system_prompt.format(len(self.actions)-1, self.available_action_str, '')

            prompt += f'\n\n## Now the human instruction is: {user_instruction}.'

            prompt += '\n\n The action history:'
            for i, action_feedback in enumerate(prev_act_feedback):
                prompt += '\n Step {}, action id {}, {}, env feedback: {}'.format(i, action_feedback[0], self.actions[action_feedback[0]], action_feedback[1])
            
            prompt += f"\n\n{self.following_prompt}"

        return prompt
    

    def get_message(self, image, prompt, messages=[]):

        if self.language_only:
            current_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}],
            }
        elif self.multiview:
            data_url1 = local_image_to_data_url(image_path=image[0])
            data_url2 = local_image_to_data_url(image_path=image[1])
            current_message = {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url1,
                        }
                    }, 
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url2,
                        }
                    },
                    {"type": "text", "text": prompt}],
            }
        elif self.multistep:
            content = []
            for img_path in image:
                data_url = local_image_to_data_url(image_path=img_path)
                content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": data_url,
                            }
                        }) 
            content.append({"type": "text", "text": prompt})
            current_message = {
                "role":"user",
                "content":content
            }
        elif self.visual_icl:
            content = []
            content.append({"type": "text", "text": prompt})
            visual_example = create_example_json_list((not self.icl_text_only))
            content.extend(visual_example)
            content.append({"type": "text", "text": "Below is your current step observation, please starting planning to navigate to the target object by learning from the above-mentioned strategy and in-context learning examples. ### Output nothing else but a JSON string following the above mentioned format ###"})
            data_url = local_image_to_data_url(image_path=image)
            content.append({
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                        }
                    }) 
            current_message = {
                "role":"user",
                "content":content
            }
        else:
            data_url = local_image_to_data_url(image_path=image)
            current_message = {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": data_url,
                        }
                    }, 
                    {"type": "text", "text": prompt}],
            }

        messages = messages + [current_message]

        return messages[-MESSAGE_WINDOW_LEN:]


    def reset(self):
        # at the beginning of the episode
        self.episode_messages = []
        self.episode_act_feedback = []
        self.planner_steps = 0
        self.output_json_error = 0

    def language_to_action(self, output_text):
        pattern = r'\*\*\d+\*\*'
        match = re.search(pattern, output_text)
        if match:
            action = int(match.group().strip('*'))
        else:
            print('random action')
            action = np.random.randint(len(self.actions))
        return action
    
    def json_to_action(self, output_text, json_key='executable_plan'):
        valid = True
        try:
            json_object = json.loads(output_text)
            action = [x[self.action_key] for x in json_object[json_key]]
            if not len(action):
                print('empty plan, using random action instead')
                action = np.random.randint(len(self.actions))
        except json.JSONDecodeError as e:
            print("Failed to decode JSON:", e)
            print('random action')
            self.output_json_error += 1
            action = np.random.randint(len(self.actions))
            valid = False
        except Exception as e:
            # Catch-all for any other unexpected errors not handled specifically
            print("An unexpected error occurred:", e)
            print('Using random action due to an unexpected error')
            action = np.random.randint(len(self.actions))
            valid = False
        return action, valid

        
    def act_custom(self, prompt, obs):
        assert type(obs) == str # input image path
        out = self.model.respond(prompt, obs)
        out = out.replace("'",'"')
        out = out.replace('\"s ', "\'s ")
        out = out.replace('```json', '').replace('```', '')
        logger.debug(f"Model Output:\n{out}\n")
        self.planner_steps += 1
        action, valid = self.json_to_action(out)
        if valid:
            return action, out
        else:
            out = '''{"visual_state_description":"invalid json, random action", "reasoning_and_reflection":"invalid json, random action",
                   "language_plan":"invalid json, random action"}'''
            return action, out


    def act(self, observation, user_instruction):
        if type(observation) == dict:
            obs = observation[self.obs_key]
        else:
            obs = observation # input image path
        
        prompt = self.process_prompt(user_instruction, prev_act_feedback=self.episode_act_feedback)
        if self.model_type == 'custom':
            return self.act_custom(prompt, obs)

        if len(self.episode_messages) == 0:
             self.episode_messages = self.get_message(obs, prompt)
        else:
            if self.chat_history:
                self.episode_messages = self.get_message(obs, prompt, self.episode_messages)
            else:
                self.episode_messages = self.get_message(obs, prompt)
        
        # Apply truncation if chat_history and truncate are both True
        messages_to_send = self.episode_messages
        if self.chat_history and self.truncate:
            messages_to_send = truncate_message_prompts(self.episode_messages)
        
        for entry in messages_to_send:
            for content_item in entry["content"]:
                if content_item["type"] == "text":
                    text_content = content_item["text"]
                    logger.debug(f"Model Input:\n{text_content}\n")

        try:
            out = self.model.respond(messages_to_send)
        except Exception as e:
            print(e)
            if 'qwen' in self.model_name:
                return -2,'''{"visual_state_description":"qwen model generate empty action due to inappropriate content check", "reasoning_and_reflection":"invalid json, random action",
                   "language_plan":"invalid json, random action"}'''

        if self.chat_history:
            self.episode_messages.append(
                {
                "role": "assistant",
                "content": [{"type": "text", "text": out}],
                }
            )
            
        logger.debug(f"Model Output:\n{out}\n")
        action, valid = self.json_to_action(out)
        self.planner_steps += 1
        if valid:
            return action, out
        else:
            out = '''{"visual_state_description":"invalid json, random action", "reasoning_and_reflection":"invalid json, random action",
                   "language_plan":"invalid json, random action"}'''
            return action, out

    def update_info(self, info):
        """Update episode feedback history."""
        self.episode_act_feedback.append([
            info['action_id'],
            info['env_feedback']
        ])


        

