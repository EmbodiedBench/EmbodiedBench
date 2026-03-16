import re
import os
import numpy as np
from tqdm import tqdm
import json
import copy
import argparse
from embodiedbench.evaluator.config.system_prompts import eb_manipulation_system_prompt
from embodiedbench.envs.eb_manipulation.EBManEnv import EBManEnv, EVAL_SETS, ValidEvalSets
from embodiedbench.envs.eb_manipulation.eb_man_utils import form_object_coord_for_input, draw_bounding_boxes, draw_xyz_coordinate
from embodiedbench.planner.manip_planner import ManipPlanner
from embodiedbench.evaluator.config.eb_manipulation_example import vlm_examples_baseline, llm_examples, vlm_examples_ablation
from embodiedbench.main import logger

class EB_ManipulationEvaluator():
    def __init__(self, config):
        self.model_name = config['model_name']
        self.eval_set = ValidEvalSets[0]
        self.config = config
        self.env = None
        self.planner = None

    def load_demonstration(self):
        all_examples = {}
        if self.config['language_only'] == 1:
            # visual icl baseline
            if self.config['visual_icl'] == 1:
                for variation in EVAL_SETS[self.eval_set]:
                    all_examples[variation] = vlm_examples_ablation[variation.split('_')[0]]
            # language only
            else:
                for variation in EVAL_SETS[self.eval_set]:
                    all_examples[variation] = llm_examples[variation.split('_')[0]]
        else:
            # main baseline
            if self.config['detection_box'] == 1 and self.config['multiview'] == 0 and self.config['visual_icl'] == 0 and self.config['multistep'] == 0:
                for variation in EVAL_SETS[self.eval_set]:
                    all_examples[variation] = vlm_examples_baseline[variation.split('_')[0]]
            # ablation study
            else:
                for variation in EVAL_SETS[self.eval_set]:
                    all_examples[variation] = vlm_examples_ablation[variation.split('_')[0]]

        return all_examples

    def save_episode_metric(self, episode_info):
        filename = 'episode_{}_res.json'.format(self.env._current_episode_num)
        res_path = os.path.join(self.env.log_path, 'results')
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        with open(os.path.join(res_path, filename), 'w', encoding='utf-8') as f:
            json.dump(episode_info, f, ensure_ascii=False)
    
    def save_planner_outputs(self, reasoning_list):
        filename = 'planner_output_episode_{}.txt'.format(self.env._current_episode_num)
        res_path = os.path.join(self.env.log_path, 'results')
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        with open(os.path.join(res_path, filename), 'w', encoding='utf-8') as f:
            for s in reasoning_list:
                f.write(s + "\n")
    
    def print_task_eval_results(self, filename):
        folder_path = f"{self.log_path}/results"
        total_number_of_task = 0
        success_number_of_task = 0
        planner_steps = 0
        output_format_error = 0

        for file_name in sorted(os.listdir(folder_path)):
            if file_name.endswith(".json") and file_name.startswith("episode"):
                file_path = os.path.join(folder_path, file_name)
                
                # Open and load the JSON file
                with open(file_path, 'r', encoding='utf-8') as json_file:
                    data = json.load(json_file)
                    task_success = data["task_success"]
                    if data["planner_output_error"] > 0:    
                        output_format_error += 1
                    if task_success == 1:
                        success_number_of_task += 1
                    planner_steps += data["planner_steps"]
                    total_number_of_task += 1

        task_log = {}
        task_log['save_path'] = self.log_path
        task_log["total_num_tasks"] = total_number_of_task
        task_log["num_success"] = success_number_of_task
        task_log["success_rate"] = success_number_of_task / total_number_of_task
        task_log["avg_planner_steps"] = planner_steps / total_number_of_task
        task_log["output_format_error"] = output_format_error

        res_path = os.path.join(self.env.log_path, 'results')
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        with open(os.path.join(res_path, filename), 'w', encoding='utf-8') as f:
            json.dump(task_log, f, ensure_ascii=False)

    def evaluate(self):
        progress_bar = tqdm(total=self.env.number_of_episodes, desc="Episodes")
        while self.env._current_episode_num < self.env.number_of_episodes:
            logger.info(f"Evaluating episode {self.env._current_episode_num} ...")
            episode_info = {'reward': [], 'action_success': []}
            image_history = []

            _, obs = self.env.reset()
            if self.config['multiview']:
                camera_views = ['front_rgb', 'wrist_rgb']
            else:
                camera_views = ['front_rgb']
            img_path_list = self.env.save_image(camera_views)

            avg_obj_coord, all_avg_point_list, camera_extrinsics_list, camera_intrinsics_list = form_object_coord_for_input(vars(copy.deepcopy(obs)), self.env.task_class, camera_views)
            if not self.config['language_only']:
                for i, img_path in enumerate(img_path_list):
                    if 'front_rgb' in img_path:
                        img_path_list[i] = draw_xyz_coordinate(img_path, self.config['resolution'])
            if self.config['detection_box'] and not self.config['language_only']:
                img_path_list = draw_bounding_boxes(img_path_list, all_avg_point_list, camera_extrinsics_list, camera_intrinsics_list)
            if self.config['multistep']:
                image_history.append(img_path_list[0])
            user_instruction = self.env.episode_language_instruction
            print(f"Instruction: {user_instruction}")
            self.planner.reset()
            done = False
            reasoning_list = []

            while not done:
                if self.config['multistep']:
                    action, reasoning = self.planner.act(image_history, user_instruction, str(avg_obj_coord), self.env.current_task_variation)
                else:
                    action, reasoning = self.planner.act(img_path_list, user_instruction, str(avg_obj_coord), self.env.current_task_variation)
                print(f"Planner Output Action: {action}")
                reasoning_list.append(reasoning)
                if len(action) == 0:
                    episode_info['reward'].append(0)
                    episode_info['action_success'].append(0)
                    info = {'task_success': 0, 'episode_elapsed_seconds': 0}
                    break
                else:
                    for action_single in action[:min(self.env._max_episode_steps - self.env._current_step, len(action))]:
                        obs, reward, done, info = self.env.step(action_single)
                        print(f"Executed action: {action_single}, Task success: {info['task_success']}")
                        logger.debug(f"reward: {reward}")
                        logger.debug(f"terminate: {done}\n")
                        self.planner.update_info(info)
                        img_path_list = self.env.save_image(camera_views)
                        for img_path in img_path_list:
                            if self.config['multistep']:
                                image_history.append(img_path)
                        episode_info['reward'].append(reward)
                        episode_info['action_success'].append(info['action_success'])
                        if done:
                            break
                
                avg_obj_coord, all_avg_point_list, camera_extrinsics_list, camera_intrinsics_list = form_object_coord_for_input(copy.deepcopy(obs), self.env.task_class, camera_views)
                if not done:
                    if not self.config['language_only']:
                        for i, img_path in enumerate(img_path_list):
                            if 'front_rgb' in img_path:
                                img_path_list[i] = draw_xyz_coordinate(img_path, self.config['resolution'])
                    if self.config['detection_box'] and not self.config['language_only']:
                        img_path_list = draw_bounding_boxes(img_path_list, all_avg_point_list, camera_extrinsics_list, camera_intrinsics_list)
                        if self.config['multistep']:
                            if image_history[-1].split('.png')[0] in img_path_list[0]:
                                image_history.pop()
                                image_history.append(img_path_list[0])
            
            # evaluation metrics
            episode_info['instruction'] = user_instruction
            episode_info['avg_reward'] = np.mean(episode_info['reward'])
            episode_info['task_success'] = info['task_success']
            episode_info['num_steps'] = self.env._current_step
            episode_info['planner_steps'] = self.planner.planner_steps
            episode_info['planner_output_error'] = self.planner.output_json_error
            episode_info["episode_elapsed_seconds"] = info["episode_elapsed_seconds"]
            self.save_episode_metric(episode_info)
            self.save_planner_outputs(reasoning_list)
            progress_bar.update()
        self.print_task_eval_results(filename="summary.json")
        self.env.close()
    
    def evaluate_main(self):
        valid_eval_sets = self.config.get('eval_sets', ValidEvalSets)
        valid_eval_sets = list(valid_eval_sets)
        if type(valid_eval_sets) == list and len(valid_eval_sets) == 0:
            valid_eval_sets = ValidEvalSets
        
        for eval_set in valid_eval_sets:
            if self.env is not None:
                self.env.close()
            self.eval_set = eval_set
            logger.info(f'Current eval set: {eval_set}')
            if "/" in self.model_name:
                real_model_name = self.model_name.split('/')[1]
            else:
                real_model_name = self.model_name
            if 'exp_name' not in self.config or self.config['exp_name'] is None:
                self.log_path = 'running/eb_manipulation/{}/n_shot={}_resolution={}_detection_box={}_multiview={}_multistep={}_visual_icl={}/{}'.format(
                                                                                                    real_model_name, 
                                                                                                    self.config['n_shots'], 
                                                                                                    self.config['resolution'], 
                                                                                                    self.config['detection_box'],
                                                                                                    self.config['multiview'],
                                                                                                    self.config['multistep'],
                                                                                                    self.config['visual_icl'],
                                                                                                    self.eval_set)
            else:
                self.log_path = 'running/eb_manipulation/{}/{}/{}'.format(real_model_name, self.config["exp_name"], self.eval_set)
            self.env = EBManEnv(eval_set=self.eval_set, img_size=(self.config['resolution'], self.config['resolution']), down_sample_ratio=self.config["down_sample_ratio"], log_path=self.log_path)
            ic_examples = self.load_demonstration()
            self.planner = ManipPlanner(model_name=self.model_name,
                                        model_type=self.config['model_type'],
                                        system_prompt=eb_manipulation_system_prompt, 
                                        examples=ic_examples, 
                                        n_shot=self.config["n_shots"], 
                                        chat_history=self.config["chat_history"],
                                        language_only=self.config["language_only"],
                                        multiview=self.config["multiview"],
                                        multistep=self.config["multistep"],
                                        visual_icl=self.config["visual_icl"],
                                        tp=self.config["tp"])
            self.evaluate()
            with open(os.path.join(self.log_path, 'config.txt'), 'w') as f:
                f.write(str(self.config))
                
    def check_config_valid(self):
        if self.config['multiview'] + self.config['multistep'] + self.config['visual_icl'] + self.config['chat_history'] > 1:
            raise ValueError("Currently, we only support one of multiview, multistep, visual_icl, chat_history feature at a time.")
        
        if self.config['language_only']:
            if self.config['multiview'] or self.config['multistep']:
                logger.warning("Language only mode should not have multiview or multistep enabled. Setting these arguments to False ...")
                self.config['multiview'] = 0
                self.config['multistep'] = 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run evaluation with specified model name.")
    parser.add_argument('--model_name', type=str, required=True, help="Name of the model to evaluate.")
    parser.add_argument('--down_sample_ratio', type=float, default=1.0, help="Down sample ratio for the eval set.")
    parser.add_argument('--model_type', type=str, default='remote', help="Type of the model to evaluate.")
    parser.add_argument('--language_only', type=int, default=0, help="Whether to use language only.")
    parser.add_argument('--eval_sets', type=lambda s: s.split(','), help='Comma-separated list of evaluation sets.')
    parser.add_argument('--chat_history', type=int, default=0, help='Whether to use chat history.')
    parser.add_argument('--n_shots', type=int, default=10)
    parser.add_argument('--multiview', type=int, default=0)
    parser.add_argument('--detection_box', type=int, default=1)
    parser.add_argument('--multistep', type=int, default=0)
    parser.add_argument('--resolution', type=int, default=600)
    parser.add_argument('--exp_name', type=str)
    parser.add_argument('--visual_icl', type=int, default=0)
    parser.add_argument('--tp', type=int, default=1, help='number of tensor parallel splits of the model parameters')
    args = parser.parse_args()

    print("\n******** Evaluating eval set: {}, model: {} ********".format(args.eval_sets, args.model_name))
    config = {
        'model_name': args.model_name,
        'model_type': args.model_type,
        'eval_sets': args.eval_sets,
        'n_shots': args.n_shots,
        'resolution': args.resolution,
        'language_only': args.language_only,
        'down_sample_ratio': args.down_sample_ratio,
        'chat_history': args.chat_history,
        'detection_box': args.detection_box,
        'multiview': args.multiview,
        'multistep': args.multistep,
        'visual_icl': args.visual_icl,
        'exp_name': args.exp_name,
        'tp': args.tp,
        'selected_indexes': [0, 12]
    }
    print("printing config ...")
    for config_key in config:
        print(f"{config_key}: {config[config_key]}")
    evaluator = EB_ManipulationEvaluator(config)
    evaluator.check_config_valid()
    evaluator.evaluate_main()
