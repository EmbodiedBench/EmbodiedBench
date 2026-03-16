import json
import os
import sys
import time
import traceback

import numpy as np
from tqdm import tqdm

from embodiedbench.envs.eb_navigation.EBNavEnv import EBNavigationEnv, ValidEvalSets
from embodiedbench.evaluator.config.eb_navigation_example import examples
from embodiedbench.evaluator.config.system_prompts import eb_navigation_system_prompt
from embodiedbench.evaluator.summarize_result import average_json_values
from embodiedbench.main import logger
from embodiedbench.planner.nav_planner import EBNavigationPlanner


system_prompt = eb_navigation_system_prompt


class EB_NavigationEvaluator():
    def __init__(self, config):
        self.model_name = config['model_name']
        self.eval_sets = config["eval_sets"]
        self.eval_set = None
        self.config = config

        self.env = None
        self.planner = None
        self.max_invalid_planner_outputs = 10

    def check_config_valid(self):
        if self.config.get('multiview', 0):
            logger.warning("multiview is not supported by the updated eb-nav planner. Disabling it.")
            self.config['multiview'] = 0
        if self.config.get('visual_icl', 0):
            logger.warning("visual_icl is not supported by the updated eb-nav planner. Disabling it.")
            self.config['visual_icl'] = 0
        if self.config.get('truncate', 0):
            logger.warning("truncate is ignored by the updated eb-nav planner. Disabling it.")
            self.config['truncate'] = 0

        if self.config.get('multistep', 0) + self.config.get('chat_history', 0) > 1:
            raise ValueError("Only one of multistep or chat_history can be enabled at a time.")

        if self.config.get('memory_compression', 0) and (
            self.config.get('chat_history', 0) or self.config.get('multistep', 0)
        ):
            raise ValueError("memory_compression cannot be enabled together with chat_history or multistep.")

        if self.config['language_only'] and self.config.get('multistep', 0):
            logger.warning("Language only mode should not have multistep enabled. Setting it to False.")
            self.config['multistep'] = 0

    def save_episode_metric(self, episode_info):
        episode_idx = self.env._current_episode_num if not len(self.env.selected_indexes) else self.env.selected_indexes[self.env._current_episode_num - 1] + 1
        filename = 'episode_{}_final_res.json'.format(episode_idx)
        res_path = os.path.join(self.env.log_path, 'results')
        if not os.path.exists(res_path):
            os.makedirs(res_path)
        with open(os.path.join(res_path, filename), 'w', encoding='utf-8') as f:
            json.dump(episode_info, f, ensure_ascii=False)

    def evaluate_main(self):
        valid_eval_sets = self.config.get('eval_sets', ValidEvalSets)
        self.eval_sets = list(valid_eval_sets)
        if type(self.eval_sets) == list and len(self.eval_sets) == 0:
            self.eval_sets = ValidEvalSets

        for eval_set in self.eval_sets:
            if self.env is not None:
                self.env.close()
            self.eval_set = eval_set
            logger.info(f'Current eval set: {eval_set}')
            exp_name = f"{self.model_name.split('/')[-1]}_{self.config['exp_name']}/{eval_set}" if len(self.config['exp_name']) else f"{self.model_name.split('/')[-1]}/{eval_set}"

            self.env = EBNavigationEnv(
                eval_set=self.eval_set,
                down_sample_ratio=self.config['down_sample_ratio'],
                exp_name=exp_name,
                multiview=self.config.get('multiview', 0),
                boundingbox=self.config['detection_box'],
                multistep=self.config.get('multistep', 0),
                resolution=self.config['resolution'],
                selected_indexes=self.config.get('selected_indexes', []),
            )

            model_type = self.config.get('model_type', 'remote')
            use_easyr1_format = self.config.get('easyr1_format')
            if use_easyr1_format is None:
                use_easyr1_format = (model_type == 'custom' or 'easyr1' in self.model_name.lower())

            self.planner = EBNavigationPlanner(
                model_name=self.model_name,
                model_type=model_type,
                actions=self.env.language_skill_set,
                system_prompt=system_prompt,
                examples=examples,
                n_shot=self.config['n_shots'],
                obs_key='head_rgb',
                chat_history=self.config['chat_history'],
                language_only=self.config['language_only'],
                multiview=self.config.get('multiview', 0),
                multistep=self.config.get('multistep', 0),
                visual_icl=self.config.get('visual_icl', 0),
                tp=self.config.get('tp', 1),
                truncate=self.config.get('truncate', False),
                use_feedback=self.config.get('env_feedback', True),
                memory_compression=self.config.get('memory_compression', 0),
                segment_len=self.config.get('segment_len', 1),
                use_easyr1_format=use_easyr1_format,
            )

            self.evaluate()
            average_json_values(os.path.join(self.env.log_path, 'results'), selected_key=None)
            with open(os.path.join(self.env.log_path, 'config.txt'), 'w') as f:
                f.write(str(self.config))

    def _execute_env_action(self, action_single, reasoning, episode_info, previous_obs):
        obs, reward, done, info = self.env.step(action_single, reasoning=reasoning)
        if isinstance(action_single, int):
            action_str = self.env.language_skill_set[action_single]
        else:
            action_str = str(action_single)
        print(f"Executed action: {action_str}, Task success: {info['task_success']}")
        logger.debug(f"reward: {reward}")
        logger.debug(f"terminate: {done}\n")
        self.planner.update_info(info, previous_obs=previous_obs, current_obs=obs)
        img_path = self.env.save_image(obs)
        episode_info['reward'].append(reward)
        episode_info['num_invalid_actions'] += int(not info['last_action_success'])
        return obs, done, info, img_path

    def evaluate(self):
        progress_bar = tqdm(total=self.env.number_of_episodes, desc="Episodes")
        memory_mode = bool(self.config.get('memory_compression', 0))
        while self.env._current_episode_num < self.env.number_of_episodes:
            logger.info(f"Evaluating episode {self.env._current_episode_num} ...")
            episode_info = {'reward': [], 'num_invalid_actions': 0, 'empty_plan': 0}
            obs = self.env.reset()
            img_path = self.env.save_image(obs)
            user_instruction = self.env.episode_language_instruction
            print(f"Instruction: {user_instruction}")

            self.planner.reset()
            self.planner.set_actions(self.env.language_skill_set)
            done = False
            invalid_planner_outputs = 0
            info = {
                'task_success': 0,
                'env_step': 0,
                'episode_elapsed_seconds': 0,
            }

            while not done:
                try:
                    action, reasoning = self.planner.act(img_path, user_instruction)
                    print(f"Planner Output Action: {action}")

                    if action == -2:
                        episode_info['empty_plan'] = 1
                        self.env.episode_log.append({
                            'last_action_success': 0.0,
                            'action_id': -2,
                            'action_description': 'empty plan',
                            'reasoning': reasoning,
                        })
                        break

                    if action == -1:
                        invalid_planner_outputs += 1
                        episode_info['reward'].append(-1)
                        episode_info['num_invalid_actions'] += 1
                        self.env.episode_log.append({
                            'last_action_success': 0.0,
                            'action_id': -1,
                            'action_description': 'invalid action',
                            'reasoning': reasoning,
                        })
                        if invalid_planner_outputs >= self.max_invalid_planner_outputs:
                            break
                        continue

                    invalid_planner_outputs = 0

                    if memory_mode:
                        action_single = action[0] if isinstance(action, list) else action
                        obs, done, info, img_path = self._execute_env_action(action_single, reasoning, episode_info, obs)
                        if not done and not info['last_action_success']:
                            print("Invalid action. Replanning from compressed memory.")
                    elif isinstance(action, list):
                        step_budget = min(self.env._max_episode_steps - self.env._current_step, len(action))
                        for action_single in action[:step_budget]:
                            obs, done, info, img_path = self._execute_env_action(action_single, reasoning, episode_info, obs)
                            if done or not info['last_action_success']:
                                print("Invalid action or task complete. If invalid then replanning.")
                                break
                    else:
                        obs, done, info, img_path = self._execute_env_action(action, reasoning, episode_info, obs)
                except Exception:
                    traceback.print_exc()
                    time.sleep(1)

            episode_info['instruction'] = user_instruction
            episode_info['reward'] = float(np.mean(episode_info['reward'])) if episode_info['reward'] else 0.0
            episode_info['task_success'] = info.get('task_success', 0)
            episode_info['num_steps'] = info.get("env_step", self.env._current_step)
            episode_info['planner_steps'] = self.planner.planner_steps
            episode_info['planner_output_error'] = self.planner.output_json_error
            episode_info["num_invalid_actions"] = episode_info['num_invalid_actions']
            episode_info["num_invalid_action_ratio"] = episode_info['num_invalid_actions'] / info["env_step"] if info.get('env_step', 0) > 0 else 0
            episode_info["episode_elapsed_seconds"] = info.get("episode_elapsed_seconds", time.time() - self.env._episode_start_time)

            self.env.save_episode_log()
            self.save_episode_metric(episode_info)
            progress_bar.update()


if __name__ == '__main__':
    config = {
        'model_name': sys.argv[2],
        'down_sample_ratio': 1,
        'model_type': 'remote',
        'language_only': False,
        'dataset': sys.argv[1],
        'chat_history': True,
        'action_num_per_plan': 5,
        'fov': 100,
        'n_shots': int(sys.argv[4]),
        'sleep_time': 0,
        'multiview': 0,
        'detection_box': 0,
        'target_only': 0,
        'multistep': 0,
        'resolution': 600,
        'purpose': "retest",
        'exp_name': sys.argv[3],
        'icl_abl': 0,
        'visual': 0,
        'env_feedback': 1,
        'easyr1_format': None,
        'memory_compression': 0,
        'segment_len': 1,
    }
    evaluator = EB_NavigationEvaluator(config)
    evaluator.check_config_valid()
    evaluator.evaluate_main()
