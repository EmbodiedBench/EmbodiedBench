import os, math, re
import textwrap

import numpy as np
from scipy import spatial
from PIL import Image, ImageDraw, ImageFont
import logging

from embodiedbench.envs.eb_alfred.env.thor_env import ThorEnv
from embodiedbench.envs.eb_alfred.gen import constants
from embodiedbench.envs.eb_alfred.gen.utils.game_util import get_objects_with_name_and_prop
from embodiedbench.envs.eb_alfred.utils import natural_word_to_ithor_name


log = logging.getLogger(__name__)

log.setLevel(level=logging.ERROR)

class ThorConnector(ThorEnv):
    def __init__(self, x_display=constants.X_DISPLAY,
                 player_screen_height=constants.DETECTION_SCREEN_HEIGHT,
                 player_screen_width=constants.DETECTION_SCREEN_WIDTH,
                 quality='MediumCloseFitShadows',
                 build_path=constants.BUILD_PATH):
        super().__init__(x_display, player_screen_height, player_screen_width, quality, build_path)
        self.font = self._load_debug_font(24)
        self.agent_height = 0.9
        self.cur_receptacle = None
        self.reachable_positions, self.reachable_position_kdtree = None, None
        self.sliced = False
        self.task = None
        self.put_count_dict = {}

    @staticmethod
    def _load_debug_font(font_size):
        # Prefer an explicit override, then try a few common Linux/macOS font paths.
        env_font = os.environ.get("EMBODIEDBENCH_FONT_PATH")
        candidate_paths = [
            env_font,
            "/usr/share/fonts/truetype/ubuntu/UbuntuMono-B.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/Supplemental/Menlo.ttc",
            "/System/Library/Fonts/Supplemental/Courier New.ttf",
        ]

        for path in candidate_paths:
            if not path:
                continue
            try:
                return ImageFont.truetype(path, font_size)
            except OSError:
                continue

        # Final fallback to PIL built-in bitmap font to avoid hard failure.
        return ImageFont.load_default()

    def restore_scene(self, object_poses, object_toggles, dirty_and_empty):
        # print(object_poses)
        super().restore_scene(object_poses, object_toggles, dirty_and_empty)
        self.reachable_positions, self.reachable_position_kdtree = self.get_reachable_positions()
        self.cur_receptacle = None

    def get_reachable_positions(self):
        free_positions = super().step(dict(action="GetReachablePositions")).metadata["actionReturn"]
        free_positions = np.array([[p['x'], p['y'], p['z']] for p in free_positions])
        kd_tree = spatial.KDTree(free_positions)
        return free_positions, kd_tree

    def write_step_on_img(self, cfg, idx, description):
        img = Image.fromarray(self.last_event.frame)
        text = str(idx) + ':' + description['action']
        lines = textwrap.wrap(text, width=20)
        y_text = 6
        draw = ImageDraw.Draw(img)
        for line in lines:
            width, height = self.font.getsize(line)
            draw.text((6, y_text), line, font=self.font, fill=(255, 255, 255))
            y_text += height
        if cfg is True:
            if not description['success']:
                text_msg = 'error : ' + description['message']
                lines = textwrap.wrap(text_msg, width=20)
                for line in lines:
                    width, height = self.font.getsize(line)
                    draw.text((6, y_text + 6), line, font=self.font, fill=(255, 0, 0))
                    y_text += height
        return img


    def find_close_reachable_position(self, loc, nth=1):
        d, i = self.reachable_position_kdtree.query(loc, k=nth + 1)
        selected = i[nth - 1]
        return self.reachable_positions[selected]

    @staticmethod
    def _format_reason(reason_code, message):
        return f"[{reason_code}] {message}"

    @staticmethod
    def _normalize_sentence(text):
        if text is None:
            return ''
        normalized = str(text).strip()
        if len(normalized) == 0:
            return ''
        if normalized[-1] not in '.!?':
            normalized += '.'
        return normalized

    @staticmethod
    def _format_target_label(target_obj=None, target_name=None):
        if target_obj is not None:
            object_type = target_obj.get('objectType') or target_obj.get('name')
            object_id = target_obj.get('objectId')
            if object_type and object_id:
                return f'{object_type} ({object_id})'
            if object_id:
                return str(object_id)
            if object_type:
                return str(object_type)
        if target_name:
            return str(target_name)
        return None

    @staticmethod
    def _format_object_state(target_obj):
        if not target_obj:
            return ''

        state_keys = (
            'visible',
            'pickupable',
            'openable',
            'isOpen',
            'toggleable',
            'isToggled',
            'sliceable',
            'receptacle',
            'distance',
        )
        parts = []
        for key in state_keys:
            if key not in target_obj:
                continue
            value = target_obj.get(key)
            if key == 'distance' and isinstance(value, (int, float)):
                value = round(value, 3)
            parts.append(f'{key}={value}')

        parent_receptacles = target_obj.get('parentReceptacles')
        if parent_receptacles:
            parts.append(f'parentReceptacles={parent_receptacles}')

        return ', '.join(parts)

    @staticmethod
    def _suggest_recovery(action_name):
        suggestions = {
            'pick up': 'moving closer, opening the containing receptacle, or putting down the currently held object first',
            'put the held object into': 'opening the receptacle, moving closer, or choosing another visible receptacle instance',
            'drop': 'checking that the robot is holding an object before dropping it',
            'open': 'stepping back, centering the object in view, or selecting another visible openable object',
            'close': 'making sure the object is open and fully visible before trying again',
            'turn on': 'moving closer and selecting a visible toggleable object that is currently off',
            'turn off': 'moving closer and selecting a visible toggleable object that is currently on',
            'slice': 'holding a knife, moving closer, and keeping the target fully visible before trying again',
        }
        return suggestions.get(action_name, 'adjusting the viewpoint and trying again')

    def _format_interaction_failure(
        self,
        reason_code,
        action_name,
        target_obj=None,
        target_name=None,
        point=None,
        search_radius=None,
        raw_error=None,
        extra_context=None,
        hint=None,
    ):
        parts = []
        target_label = self._format_target_label(target_obj=target_obj, target_name=target_name)
        if target_label is None:
            parts.append(f'Failed to {action_name}.')
        else:
            parts.append(f'Failed to {action_name} {target_label}.')

        if point is not None:
            point_text = f'Selected point ({point[0]}, {point[1]})'
            if search_radius is not None:
                point_text += f' with search radius {search_radius}'
            parts.append(point_text + '.')

        extra_sentence = self._normalize_sentence(extra_context)
        if extra_sentence:
            parts.append(extra_sentence)

        object_state = self._format_object_state(target_obj)
        if object_state:
            parts.append(f'Object state: {object_state}.')

        raw_error_sentence = self._normalize_sentence(raw_error)
        if raw_error_sentence:
            parts.append(f'AI2-THOR error: {raw_error_sentence}')

        recovery_hint = hint or self._suggest_recovery(action_name)
        recovery_sentence = self._normalize_sentence(f'Try {recovery_hint}')
        if recovery_sentence:
            parts.append(recovery_sentence)

        return self._format_reason(reason_code, ' '.join(parts))

    def _normalize_point(self, point):
        if point is None or not isinstance(point, (list, tuple)) or len(point) != 2:
            return None
        try:
            x = int(round(float(point[0])))
            y = int(round(float(point[1])))
        except (TypeError, ValueError):
            return None
        return x, y

    def _get_object_by_id(self, object_id):
        if self.last_event is None:
            return None

        metadata_objects = self.last_event.metadata.get('objects', [])
        if object_id is None:
            return None

        target_id = str(object_id).strip()
        for obj in metadata_objects:
            if obj.get('objectId') == target_id:
                return obj

        # Conservative fallback for transient id mismatch: only use unique objectType.
        if '|' in target_id:
            target_type = target_id.split('|', 1)[0]
            type_matches = [obj for obj in metadata_objects if obj.get('objectType') == target_type]
            if len(type_matches) == 1:
                return type_matches[0]

        return None

    def _mark_action_failed(self):
        if self.last_event is not None:
            self.last_event.metadata['lastActionSuccess'] = False

    def _format_hidden_target_reason(self, target_obj, x, y):
        parent_receptacles = target_obj.get('parentReceptacles')
        if parent_receptacles:
            parent_recep = parent_receptacles[0]
            is_open = self.get_object_prop(parent_recep, 'isOpen', self.last_event.metadata)
            is_openable = self.get_object_prop(parent_recep, 'openable', self.last_event.metadata)
            if is_openable and not is_open:
                return self._format_interaction_failure(
                    'inside_closed_receptacle',
                    'interact with',
                    target_obj=target_obj,
                    point=(x, y),
                    extra_context=f'Target object is inside closed receptacle {parent_recep}.',
                    hint='opening that receptacle before trying again',
                )
        return self._format_interaction_failure(
            'target_not_visible',
            'interact with',
            target_obj=target_obj,
            point=(x, y),
            extra_context='Target object is currently not visible.',
            hint='navigating until the object is visible before interacting',
        )

    @staticmethod
    def _sort_point_candidates(candidates):
        return sorted(candidates, key=lambda o: (not o.get('visible', False), o.get('distance', 1e9)))

    def _candidate_object_ids_at_radius(self, x, y, radius):
        if self.last_event is None or self.last_event.instance_segmentation_frame is None:
            return []

        instance_segs = np.array(self.last_event.instance_segmentation_frame)
        color_to_object_id = self.last_event.color_to_object_id
        img_h, img_w = instance_segs.shape[:2]
        x0 = max(0, x - radius)
        x1 = min(img_w, x + radius + 1)
        y0 = max(0, y - radius)
        y1 = min(img_h, y + radius + 1)
        patch = instance_segs[y0:y1, x0:x1]
        if patch.size == 0:
            return []

        colors = patch.reshape(-1, 3)
        color_hist = {}
        for color in colors:
            color_key = tuple(int(v) for v in color)
            color_hist[color_key] = color_hist.get(color_key, 0) + 1

        candidate_ids = []
        for color_key, _ in sorted(color_hist.items(), key=lambda kv: kv[1], reverse=True):
            obj_id = color_to_object_id.get(color_key)
            if obj_id is not None and obj_id not in candidate_ids:
                candidate_ids.append(obj_id)

        return candidate_ids

    def _resolve_point_candidates(self, point, max_search_radius=3, required_property=None):
        normalized_point = self._normalize_point(point)
        if normalized_point is None:
            self._mark_action_failed()
            return None, self._format_reason('invalid_point_format', f'Point {point} is invalid. Expected [x, y].')
        x, y = normalized_point

        if self.last_event is None or self.last_event.frame is None:
            self._mark_action_failed()
            return None, self._format_reason('missing_observation', 'No current observation frame is available.')

        img_h, img_w = self.last_event.frame.shape[:2]
        if x < 0 or x >= img_w or y < 0 or y >= img_h:
            self._mark_action_failed()
            return None, self._format_reason('point_out_of_bounds', f'Point ({x}, {y}) is outside image bounds [0,{img_w - 1}]x[0,{img_h - 1}].')

        candidate_ids_seen = []
        first_metadata_ctx = None
        first_visible_ctx = None
        first_match_ctx = None

        def build_ctx(radius, candidate_ids, candidate_objects, matched_candidates):
            return {
                'x': x,
                'y': y,
                'used_radius': radius,
                'candidate_ids': list(candidate_ids),
                'candidates': self._sort_point_candidates(candidate_objects),
                'matched_candidates': self._sort_point_candidates(matched_candidates),
            }

        for radius in range(max_search_radius + 1):
            candidate_ids = self._candidate_object_ids_at_radius(x, y, radius)
            if not candidate_ids:
                continue

            candidate_ids_seen = list(candidate_ids)
            candidate_objects = [self._get_object_by_id(obj_id) for obj_id in candidate_ids]
            candidate_objects = [obj for obj in candidate_objects if obj is not None]

            deduped_candidates = []
            seen_object_ids = set()
            for obj in candidate_objects:
                object_id = obj.get('objectId')
                if object_id in seen_object_ids:
                    continue
                seen_object_ids.add(object_id)
                deduped_candidates.append(obj)
            candidate_objects = deduped_candidates

            if not candidate_objects:
                continue

            if required_property is None:
                matched_candidates = candidate_objects
            else:
                matched_candidates = [obj for obj in candidate_objects if obj.get(required_property, False)]
            visible_candidates = [obj for obj in candidate_objects if obj.get('visible', False)]
            visible_matched_candidates = [obj for obj in matched_candidates if obj.get('visible', False)]

            current_ctx = build_ctx(radius, candidate_ids, candidate_objects, matched_candidates)
            if first_metadata_ctx is None:
                first_metadata_ctx = current_ctx
            if first_visible_ctx is None and visible_candidates:
                first_visible_ctx = build_ctx(radius, candidate_ids, visible_candidates, [])
            if first_match_ctx is None and matched_candidates:
                first_match_ctx = current_ctx
            if visible_matched_candidates:
                return build_ctx(radius, candidate_ids, candidate_objects, visible_matched_candidates), None

        if len(candidate_ids_seen) == 0:
            self._mark_action_failed()
            return None, self._format_reason('point_on_non_object', f'No object was detected near point ({x}, {y}).')

        if first_visible_ctx is not None:
            return first_visible_ctx, None

        if first_match_ctx is not None:
            return first_match_ctx, None

        if first_metadata_ctx is None:
            self._mark_action_failed()
            metadata_ids = [obj.get('objectId') for obj in self.last_event.metadata.get('objects', []) if obj.get('objectId')]
            sample_candidate_ids = candidate_ids_seen[:3]
            sample_metadata_ids = metadata_ids[:3]
            return None, self._format_reason(
                'point_on_non_interactable',
                (
                    'Only non-interactable scene structure was detected near the selected point. '
                    f'candidate_ids={sample_candidate_ids}, metadata_count={len(metadata_ids)}, metadata_sample={sample_metadata_ids}.'
                )
            )

        return first_metadata_ctx, None

    def pick_by_point(self, point, max_search_radius=3):
        point_ctx, err = self._resolve_point_candidates(
            point,
            max_search_radius=max_search_radius,
            required_property='pickupable',
        )
        if err:
            return err
        x, y = point_ctx['x'], point_ctx['y']

        if len(self.last_event.metadata.get('inventoryObjects', [])) > 0:
            holding_obj = self.last_event.metadata['inventoryObjects'][0]['objectType']
            self._mark_action_failed()
            return self._format_reason('holding_object', f'Robot is already holding {holding_obj}. Put down or drop it first.')

        pickup_candidates = point_ctx.get('matched_candidates', [])
        if len(pickup_candidates) == 0:
            top_obj = point_ctx['candidates'][0]
            self._mark_action_failed()
            return self._format_reason(
                'not_pickupable',
                f'Point ({x}, {y}) selected {top_obj["objectType"]}, but this object cannot be picked up.'
            )

        target_obj = self._sort_point_candidates(pickup_candidates)[0]
        if not target_obj.get('visible', False):
            self._mark_action_failed()
            return self._format_hidden_target_reason(target_obj, x, y)

        super().step(dict(action="PickupObject", objectId=target_obj['objectId'], forceAction=False))
        if not self.last_event.metadata['lastActionSuccess']:
            error_msg = self.last_event.metadata.get('errorMessage', 'PickupObject failed.')
            return self._format_interaction_failure(
                'pickup_api_failed',
                'pick up',
                target_obj=target_obj,
                point=(x, y),
                search_radius=point_ctx['used_radius'],
                raw_error=error_msg,
            )
        return ''

    def put_by_point(self, point, max_search_radius=3):
        if len(self.last_event.metadata.get('inventoryObjects', [])) == 0:
            self._mark_action_failed()
            return self._format_reason('not_holding_object', 'Robot is not holding any object.')

        point_ctx, err = self._resolve_point_candidates(
            point,
            max_search_radius=max_search_radius,
            required_property='receptacle',
        )
        if err:
            return err
        x, y = point_ctx['x'], point_ctx['y']

        receptacle_candidates = point_ctx.get('matched_candidates', [])
        if len(receptacle_candidates) == 0:
            top_obj = point_ctx['candidates'][0]
            self._mark_action_failed()
            return self._format_reason(
                'not_receptacle',
                f'Point ({x}, {y}) selected {top_obj["objectType"]}, but this object is not a receptacle.'
            )

        target_obj = self._sort_point_candidates(receptacle_candidates)[0]
        if target_obj.get('openable', False) and not target_obj.get('isOpen', False):
            self._mark_action_failed()
            return self._format_reason(
                'receptacle_closed',
                f'Target receptacle {target_obj["objectType"]} is closed at point ({x}, {y}). Open it before putting.'
            )

        ret = self.put(target_obj['objectId'])
        if len(ret) > 0:
            self._mark_action_failed()
            return ret
        return ''

    def open_by_point(self, point, max_search_radius=3):
        point_ctx, err = self._resolve_point_candidates(
            point,
            max_search_radius=max_search_radius,
            required_property='openable',
        )
        if err:
            return err
        x, y = point_ctx['x'], point_ctx['y']

        openable_candidates = point_ctx.get('matched_candidates', [])
        if len(openable_candidates) == 0:
            top_obj = point_ctx['candidates'][0]
            self._mark_action_failed()
            return self._format_reason(
                'not_openable',
                f'Point ({x}, {y}) selected {top_obj["objectType"]}, but this object is not openable.'
            )
        target_obj = self._sort_point_candidates(openable_candidates)[0]
        if not target_obj.get('visible', False):
            self._mark_action_failed()
            return self._format_hidden_target_reason(target_obj, x, y)
        if target_obj.get('isOpen', False):
            self._mark_action_failed()
            return self._format_reason('already_open', f'{target_obj["objectType"]} is already open.')

        super().step(dict(action="OpenObject", objectId=target_obj['objectId']))
        if not self.last_event.metadata['lastActionSuccess']:
            return self._format_interaction_failure(
                'open_api_failed',
                'open',
                target_obj=target_obj,
                point=(x, y),
                search_radius=point_ctx['used_radius'],
                raw_error=self.last_event.metadata.get('errorMessage', 'OpenObject failed.'),
            )
        return ''

    def close_by_point(self, point, max_search_radius=3):
        point_ctx, err = self._resolve_point_candidates(
            point,
            max_search_radius=max_search_radius,
            required_property='openable',
        )
        if err:
            return err
        x, y = point_ctx['x'], point_ctx['y']

        openable_candidates = point_ctx.get('matched_candidates', [])
        if len(openable_candidates) == 0:
            top_obj = point_ctx['candidates'][0]
            self._mark_action_failed()
            return self._format_reason(
                'not_openable',
                f'Point ({x}, {y}) selected {top_obj["objectType"]}, but this object is not openable.'
            )
        target_obj = self._sort_point_candidates(openable_candidates)[0]
        if not target_obj.get('visible', False):
            self._mark_action_failed()
            return self._format_hidden_target_reason(target_obj, x, y)
        if not target_obj.get('isOpen', False):
            self._mark_action_failed()
            return self._format_reason('already_closed', f'{target_obj["objectType"]} is already closed.')

        super().step(dict(action="CloseObject", objectId=target_obj['objectId']))
        if not self.last_event.metadata['lastActionSuccess']:
            return self._format_interaction_failure(
                'close_api_failed',
                'close',
                target_obj=target_obj,
                point=(x, y),
                search_radius=point_ctx['used_radius'],
                raw_error=self.last_event.metadata.get('errorMessage', 'CloseObject failed.'),
            )
        return ''

    def toggleon_by_point(self, point, max_search_radius=3):
        point_ctx, err = self._resolve_point_candidates(
            point,
            max_search_radius=max_search_radius,
            required_property='toggleable',
        )
        if err:
            return err
        x, y = point_ctx['x'], point_ctx['y']

        toggle_candidates = point_ctx.get('matched_candidates', [])
        if len(toggle_candidates) == 0:
            top_obj = point_ctx['candidates'][0]
            self._mark_action_failed()
            return self._format_reason(
                'not_toggleable',
                f'Point ({x}, {y}) selected {top_obj["objectType"]}, but this object is not toggleable.'
            )
        target_obj = self._sort_point_candidates(toggle_candidates)[0]
        if not target_obj.get('visible', False):
            self._mark_action_failed()
            return self._format_hidden_target_reason(target_obj, x, y)
        if target_obj.get('isToggled', False):
            self._mark_action_failed()
            return self._format_reason('already_on', f'{target_obj["objectType"]} is already turned on.')

        super().step(dict(action="ToggleObjectOn", objectId=target_obj['objectId']))
        if not self.last_event.metadata['lastActionSuccess']:
            return self._format_interaction_failure(
                'toggle_on_api_failed',
                'turn on',
                target_obj=target_obj,
                point=(x, y),
                search_radius=point_ctx['used_radius'],
                raw_error=self.last_event.metadata.get('errorMessage', 'ToggleObjectOn failed.'),
            )
        return ''

    def toggleoff_by_point(self, point, max_search_radius=3):
        point_ctx, err = self._resolve_point_candidates(
            point,
            max_search_radius=max_search_radius,
            required_property='toggleable',
        )
        if err:
            return err
        x, y = point_ctx['x'], point_ctx['y']

        toggle_candidates = point_ctx.get('matched_candidates', [])
        if len(toggle_candidates) == 0:
            top_obj = point_ctx['candidates'][0]
            self._mark_action_failed()
            return self._format_reason(
                'not_toggleable',
                f'Point ({x}, {y}) selected {top_obj["objectType"]}, but this object is not toggleable.'
            )
        target_obj = self._sort_point_candidates(toggle_candidates)[0]
        if not target_obj.get('visible', False):
            self._mark_action_failed()
            return self._format_hidden_target_reason(target_obj, x, y)
        if not target_obj.get('isToggled', False):
            self._mark_action_failed()
            return self._format_reason('already_off', f'{target_obj["objectType"]} is already turned off.')

        super().step(dict(action="ToggleObjectOff", objectId=target_obj['objectId']))
        if not self.last_event.metadata['lastActionSuccess']:
            return self._format_interaction_failure(
                'toggle_off_api_failed',
                'turn off',
                target_obj=target_obj,
                point=(x, y),
                search_radius=point_ctx['used_radius'],
                raw_error=self.last_event.metadata.get('errorMessage', 'ToggleObjectOff failed.'),
            )
        return ''

    def slice_by_point(self, point, max_search_radius=3):
        point_ctx, err = self._resolve_point_candidates(
            point,
            max_search_radius=max_search_radius,
            required_property='sliceable',
        )
        if err:
            return err
        x, y = point_ctx['x'], point_ctx['y']

        inv = self.last_event.metadata.get('inventoryObjects', [])
        if len(inv) == 0 or 'Knife' not in inv[0].get('objectType', ''):
            self._mark_action_failed()
            return self._format_reason('missing_knife', 'Robot should hold a knife before slicing.')

        slice_candidates = point_ctx.get('matched_candidates', [])
        if len(slice_candidates) == 0:
            top_obj = point_ctx['candidates'][0]
            self._mark_action_failed()
            return self._format_reason(
                'not_sliceable',
                f'Point ({x}, {y}) selected {top_obj["objectType"]}, but this object is not sliceable.'
            )
        target_obj = self._sort_point_candidates(slice_candidates)[0]
        if not target_obj.get('visible', False):
            self._mark_action_failed()
            return self._format_hidden_target_reason(target_obj, x, y)

        super().step(dict(action="SliceObject", objectId=target_obj['objectId']))
        if not self.last_event.metadata['lastActionSuccess']:
            return self._format_interaction_failure(
                'slice_api_failed',
                'slice',
                target_obj=target_obj,
                point=(x, y),
                search_radius=point_ctx['used_radius'],
                raw_error=self.last_event.metadata.get('errorMessage', 'SliceObject failed.'),
            )
        self.sliced = True
        return ''

    @staticmethod
    def _extract_point_from_text(normalized_text):
        lower_text = normalized_text.lower()
        if not any(k in lower_text for k in ('point', 'pixel', 'coord', 'coordinate', '@', ' at ')):
            return None
        point_match = re.search(r'(-?\d+)\s*[,，]\s*(-?\d+)', normalized_text)
        if point_match is None:
            return None
        return [int(point_match.group(1)), int(point_match.group(2))]

    def llm_skill_interact(self, instruction):
        if isinstance(instruction, dict):
            action_type = str(instruction.get('action', '')).strip().lower()
            ret = 'instruction not supported'
            action_handler_map = {
                'pickup_by_point': self.pick_by_point,
                'pick_by_point': self.pick_by_point,
                'pickup_at_point': self.pick_by_point,
                'pickup_point': self.pick_by_point,
                'put_by_point': self.put_by_point,
                'put_at_point': self.put_by_point,
                'open_by_point': self.open_by_point,
                'close_by_point': self.close_by_point,
                'toggleon_by_point': self.toggleon_by_point,
                'toggleoff_by_point': self.toggleoff_by_point,
                'slice_by_point': self.slice_by_point,
            }
            if action_type in action_handler_map:
                ret = action_handler_map[action_type](instruction.get('point'))
            if ret == 'instruction not supported' and self.last_event is not None:
                self.last_event.metadata['lastActionSuccess'] = False

            last_action_success = self.last_event.metadata.get('lastActionSuccess', False) if self.last_event is not None else len(ret) <= 0
            return {
                'action': instruction,
                'success': last_action_success,
                'message': ret
            }

        normalized = instruction.strip()
        normalized_no_period = normalized[:-1] if normalized.endswith('.') else normalized

        if normalized.startswith("find "):
            obj_name = normalized.replace('find a ', '').replace('find an ', '')
            self.cur_receptacle = obj_name
            if self.last_event is not None:
                self.last_event.metadata['lastActionSuccess'] = True  # always set to success
            ret = ''
        elif normalized_no_period == "Move forward by 0.25" or normalized == "MoveAhead":
            self.step(dict(action="MoveAhead", moveMagnitude=0.25))
            ret = "" if self.last_event.metadata['lastActionSuccess'] else self.last_event.metadata.get('errorMessage', '')
        elif normalized_no_period == "Move backward by 0.25" or normalized == "MoveBack":
            self.step(dict(action="MoveBack", moveMagnitude=0.25))
            ret = "" if self.last_event.metadata['lastActionSuccess'] else self.last_event.metadata.get('errorMessage', '')
        elif normalized_no_period == "Move rightward by 0.25" or normalized == "MoveRight":
            self.step(dict(action="MoveRight", moveMagnitude=0.25))
            ret = "" if self.last_event.metadata['lastActionSuccess'] else self.last_event.metadata.get('errorMessage', '')
        elif normalized_no_period == "Move leftward by 0.25" or normalized == "MoveLeft":
            self.step(dict(action="MoveLeft", moveMagnitude=0.25))
            ret = "" if self.last_event.metadata['lastActionSuccess'] else self.last_event.metadata.get('errorMessage', '')
        elif normalized_no_period == "Rotate to the right by 90 degrees" or normalized == "RotateRight":
            self.step(dict(action="RotateRight", degrees=90))
            ret = "" if self.last_event.metadata['lastActionSuccess'] else self.last_event.metadata.get('errorMessage', '')
        elif normalized_no_period == "Rotate to the left by 90 degrees" or normalized == "RotateLeft":
            self.step(dict(action="RotateLeft", degrees=90))
            ret = "" if self.last_event.metadata['lastActionSuccess'] else self.last_event.metadata.get('errorMessage', '')
        elif normalized_no_period == "Tilt the camera upward by 30 degrees" or normalized == "LookUp":
            self.step(dict(action="LookUp", degrees=30))
            ret = "" if self.last_event.metadata['lastActionSuccess'] else self.last_event.metadata.get('errorMessage', '')
        elif normalized_no_period == "Tilt the camera downward by 30 degrees" or normalized == "LookDown":
            self.step(dict(action="LookDown", degrees=30))
            ret = "" if self.last_event.metadata['lastActionSuccess'] else self.last_event.metadata.get('errorMessage', '')
        elif normalized == "LookUp_15":
            self.step(dict(action="LookUp", degrees=15))
            ret = "" if self.last_event.metadata['lastActionSuccess'] else self.last_event.metadata.get('errorMessage', '')
        elif normalized == "LookDown_15":
            self.step(dict(action="LookDown", degrees=15))
            ret = "" if self.last_event.metadata['lastActionSuccess'] else self.last_event.metadata.get('errorMessage', '')
        elif normalized.startswith("pick up "):
            obj_name = normalized.replace('pick up the ', '')
            maybe_point = self._extract_point_from_text(normalized)
            if maybe_point is not None:
                ret = self.pick_by_point(maybe_point)
            else:
                ret = self.pick(natural_word_to_ithor_name(obj_name))
        elif normalized.startswith("put down "):
            maybe_point = self._extract_point_from_text(normalized)
            if maybe_point is not None:
                ret = self.put_by_point(maybe_point)
            elif self.cur_receptacle is None:
                ret = self.drop()
            else:
                m = re.match(r'put down (.+)', normalized)

                if self.cur_receptacle  in self.put_count_dict:
                    self.put_count_dict[self.cur_receptacle ] += 1
                else:
                    self.put_count_dict[self.cur_receptacle ] = 1

                receptacle = self.cur_receptacle
                ret = self.put(natural_word_to_ithor_name(receptacle))

                if len(ret) > 16 and self.put_count_dict[receptacle] >= 3:
                    # if put down failed, then drop the object
                    self.drop()
                    ret += f'. The robot dropped the object instead.'
                    self.last_event.metadata['lastActionSuccess'] = False
        elif normalized.startswith("open "):
            maybe_point = self._extract_point_from_text(normalized)
            if maybe_point is not None:
                ret = self.open_by_point(maybe_point)
            else:
                obj_name = normalized.replace('open the ', '')
                ret = self.open(natural_word_to_ithor_name(obj_name))
        elif normalized.startswith("close "):
            maybe_point = self._extract_point_from_text(normalized)
            if maybe_point is not None:
                ret = self.close_by_point(maybe_point)
            else:
                obj_name = normalized.replace('close the ', '')
                ret = self.close(natural_word_to_ithor_name(obj_name))
        elif normalized.startswith("turn on "):
            maybe_point = self._extract_point_from_text(normalized)
            if maybe_point is not None:
                ret = self.toggleon_by_point(maybe_point)
            else:
                obj_name = normalized.replace('turn on the ', '')
                ret = self.toggleon(natural_word_to_ithor_name(obj_name))
        elif normalized.startswith("turn off "):
            maybe_point = self._extract_point_from_text(normalized)
            if maybe_point is not None:
                ret = self.toggleoff_by_point(maybe_point)
            else:
                obj_name = normalized.replace('turn off the ', '')
                ret = self.toggleoff(natural_word_to_ithor_name(obj_name))
        elif normalized.startswith("slice "):
            maybe_point = self._extract_point_from_text(normalized)
            if maybe_point is not None:
                ret = self.slice_by_point(maybe_point)
            else:
                obj_name = normalized.replace('slice the ', '')
                ret = self.slice(natural_word_to_ithor_name(obj_name))
                self.sliced = True
        elif normalized.startswith("drop"):
            ret = self.drop()
        else:
            ret = 'instruction not supported'

        if ret == 'instruction not supported' and self.last_event is not None:
            self.last_event.metadata['lastActionSuccess'] = False

        last_action_success = False
        if self.last_event is not None:
            last_action_success = self.last_event.metadata.get('lastActionSuccess', False)
        else:
            last_action_success = len(ret) <= 0

        if not last_action_success:
            log.warning(f"llm_skill_interact failed")
            log.warning(f"errorMessage: {self.last_event.metadata['errorMessage']}")
            log.warning(f"returned msg: {ret}")
        else:
            log.info(f"Last action succeeded")

        ret_dict = {
            'action': instruction,
            'success': last_action_success,
            'message': ret
        }

        return ret_dict

    def get_object_prop(self, name, prop, metadata):
        for obj in metadata['objects']:
            if name in obj['objectId']:
                return obj[prop]
        return None

    @staticmethod
    def angle_diff(x, y):
        x = math.radians(x)
        y = math.radians(y)
        return math.degrees(math.atan2(math.sin(x - y), math.cos(x - y)))
    
    def nav_obj(self, target_obj: str, prefer_sliced=False):
        objects = self.last_event.metadata['objects']
        action_name = 'object navigation'
        ret_msg = ''
        print(f'{action_name} ({target_obj})')

        # get the object location
        if '|' in target_obj:
            obj_id = target_obj
            target_obj = target_obj.split('|')[0]
            tmp_id, tmp_obj_data = self.get_obj_id_from_name(target_obj, priority_in_visibility=True, priority_sliced=prefer_sliced)
            # if sliced object 
            if 'Sliced' in tmp_id and obj_id in tmp_id:
                obj_id = tmp_id
                obj_data = tmp_obj_data
        else:
            obj_id, obj_data = self.get_obj_id_from_name(target_obj, priority_in_visibility=True, priority_sliced=prefer_sliced)

        # find object index from id
        obj_idx = -1
        for i, o in enumerate(objects):
            if o['objectId'] == obj_id:
                obj_idx = i
                break
        if obj_idx == -1:
            ret_msg = f'Cannot find {target_obj}. This object may not exist in this scene. Try to explore other instances instead.'
        else:
            # teleport sometimes fails even with reachable positions. if fails, repeat with the next closest reachable positions.
            max_attempts = 20
            teleport_success = False

            # get obj location
            loc = objects[obj_idx]['position']
            obj_rot = objects[obj_idx]['rotation']['y']

            # # do not move if the object is already visible and close
            # if objects[obj_idx]['visible'] and objects[obj_idx]['distance'] < 1.0:
            #     log.info('Object is already visible')
            #     max_attempts = 0
            #     teleport_success = True

            # try teleporting
            reachable_pos_idx = 0
            for i in range(max_attempts):
                reachable_pos_idx += 1
                if i == 10 and (target_obj == 'Fridge' or target_obj == 'Microwave'):
                    reachable_pos_idx -= 10

                closest_loc = self.find_close_reachable_position([loc['x'], loc['y'], loc['z']], reachable_pos_idx)
                # calculate desired rotation angle (see https://github.com/allenai/ai2thor/issues/806)
                rot_angle = math.atan2(-(loc['x'] - closest_loc[0]), loc['z'] - closest_loc[2])
                if rot_angle > 0:
                    rot_angle -= 2 * math.pi
                rot_angle = -(180 / math.pi) * rot_angle  # in degrees

                if i < 10 and (target_obj == 'Fridge' or target_obj == 'Microwave'):  # not always correct, but better than nothing
                    angle_diff = abs(self.angle_diff(rot_angle, obj_rot))
                    if target_obj == 'Fridge' and \
                            not ((90 - 20 < angle_diff < 90 + 20) or (270 - 20 < angle_diff < 270 + 20)):
                        continue
                    if target_obj == 'Microwave' and \
                            not ((180 - 20 < angle_diff < 180 + 20) or (0 - 20 < angle_diff < 0 + 20)):
                        continue

                # calculate desired horizon angle
                camera_height = self.agent_height + constants.CAMERA_HEIGHT_OFFSET
                xz_dist = math.hypot(loc['x'] - closest_loc[0], loc['z'] - closest_loc[2])
                hor_angle = math.atan2((loc['y'] - camera_height), xz_dist)
                hor_angle = (180 / math.pi) * hor_angle  # in degrees
                hor_angle *= 0.9  # adjust angle for better view
                # hor_angle = -30
                # hor_angle = 0

                # teleport ### Full
                super().step(dict(action="TeleportFull", x=closest_loc[0], y=self.agent_height, z=closest_loc[2], rotation=rot_angle, horizon=-hor_angle))

                if not self.last_event.metadata['lastActionSuccess']:
                    log.warning(
                        f"TeleportFull action failed: {self.last_event.metadata['errorMessage']}, trying again...")
                else:
                    teleport_success = True
                    break

            if not teleport_success:
                ret_msg = f'Cannot move to {target_obj}'

        return ret_msg

    def get_obj_id_from_name(self, obj_name, only_pickupable=False, only_toggleable=False, priority_sliced=False, get_inherited=False,
                             parent_receptacle_penalty=True, priority_in_visibility=False, exclude_obj_id=None):
        obj_id = None
        obj_data = None
        min_distance = 1e+8

        if any(i.isdigit() for i in obj_name):
            for i in range(len(self.last_event.metadata['objects'])):
                if obj_name in self.last_event.metadata['objects'][i]['name']:
                    obj_id = self.last_event.metadata['objects'][i]['objectId']
                    obj_data = self.last_event.metadata['objects'][i]
                    break
            return obj_id, obj_data
        for obj in self.last_event.metadata['objects']:
            if obj['objectId'] == exclude_obj_id:
                continue
            
            if (only_pickupable is False or obj['pickupable']) and \
                    (only_toggleable is False or obj['toggleable']) and \
                    obj['objectId'].split('|')[0].casefold() == obj_name.casefold() and \
                    (get_inherited is False or len(obj['objectId'].split('|')) == 5):
                
                if obj["distance"] < min_distance:
                    penalty_advantage = 0  # low priority for objects in closable receptacles such as fridge, microwave
                    if parent_receptacle_penalty and obj['parentReceptacles']:
                        for p in obj['parentReceptacles']:
                            is_open = self.get_object_prop(p, 'isOpen', self.last_event.metadata)
                            openable = self.get_object_prop(p, 'openable', self.last_event.metadata)
                            if openable is True and is_open is False:
                                penalty_advantage += 100000
                                break

                    if obj_name.casefold() == 'stoveburner':
                        # try to find an empty stove
                        if len(obj['receptacleObjectIds']) > 0:
                            penalty_advantage += 10000

                    if priority_in_visibility and obj['visible'] is False:
                        penalty_advantage += 1000

                    if priority_sliced and '_Slice' in obj['name']:
                        penalty_advantage += -100  # prefer sliced objects; this prevents picking up non-sliced objects

                    if obj["distance"] + penalty_advantage < min_distance:
                        min_distance = obj["distance"] + penalty_advantage
                        obj_data = obj
                        obj_id = obj["objectId"]

        return obj_id, obj_data

    def pick(self, obj_name):
        obj_id, obj_data = self.get_obj_id_from_name(obj_name, only_pickupable=True, priority_in_visibility=True, priority_sliced=self.sliced)

        ret_msg = ''
        log.info(f'pick {obj_id}')

        if obj_id is None:
            ret_msg = self._format_reason(
                'pickup_target_not_found',
                f'Cannot find {obj_name} to pick up. The object may be out of view, inside a receptacle, or a different instance may need to be targeted.'
            )
        else:
            if obj_data['visible'] is False and obj_data['parentReceptacles'] is not None and len(obj_data['parentReceptacles']) > 0:
                recep_name = obj_data['parentReceptacles'][0]
                ret_msg = self._format_reason(
                    'target_not_visible',
                    (
                        f'{obj_name} is not visible because it is in {recep_name}. '
                        f'Open or navigate to that receptacle first. Note: multiple instances of {recep_name} may exist.'
                    )
                )

                # try anyway
                super().step(dict(
                    action="PickupObject",
                    objectId=obj_id,
                    forceAction=False
                ))
            else:
                super().step(dict(
                    action="PickupObject",
                    objectId=obj_id,
                    forceAction=False
                ))
                
                if not self.last_event.metadata['lastActionSuccess']:
                    if len(self.last_event.metadata['inventoryObjects']) > 0:
                        holding_obj_id = self.last_event.metadata['inventoryObjects'][0]['objectId']
                        holding_obj_type = self.last_event.metadata['inventoryObjects'][0]['objectType']
                        ret_msg = self._format_reason(
                            'holding_object',
                            f'Cannot pick up {self._format_target_label(obj_data, obj_name)} because the robot is already holding {holding_obj_type} ({holding_obj_id}). Put it down or drop it first.'
                        )
                    else:
                        ret_msg = self._format_interaction_failure(
                            'pickup_api_failed',
                            'pick up',
                            target_obj=obj_data,
                            raw_error=self.last_event.metadata.get('errorMessage', 'PickupObject failed.'),
                        )

            if self.last_event.metadata['lastActionSuccess']:
                ret_msg = ''

        return ret_msg

    def put(self, receptacle_name):
        # assume the agent always put the object currently holding
        ret_msg = ''
        orig_receptacle_name = receptacle_name

        if len(self.last_event.metadata['inventoryObjects']) == 0:
            ret_msg = self._format_reason(
                'not_holding_object',
                'Robot is not holding any object. Pick something up before trying to put it down.'
            )
            return ret_msg
        else:
            holding_obj_id = self.last_event.metadata['inventoryObjects'][0]['objectId']
            holding_obj_type = self.last_event.metadata['inventoryObjects'][0]['objectType']

        
        halt = False
        last_recep_id = None
        exclude_obj_id = None
        last_error = ''
        last_target_obj = None
        target_not_found = False
        for k in range(2):  # try closest and next closest one
            for j in range(7):  # move/look around or rotate obj
                for i in range(2):  # try inherited receptacles too (e.g., sink basin, bath basin)
                    if k == 1 and exclude_obj_id is None:
                        exclude_obj_id = last_recep_id  # previous recep id

                    # for the second round, find another receptacle
                    if k == 0 and '|' in orig_receptacle_name: 
                        if i == 1:
                            continue
                        recep_id = orig_receptacle_name
                        receptacle_name = orig_receptacle_name.split('|')[0]
                    else:
                        if 'Sink' in receptacle_name or 'Bathtub' in receptacle_name: # sink base
                            if i == 0:
                                recep_id, _ = self.get_obj_id_from_name(receptacle_name, get_inherited=True, exclude_obj_id=exclude_obj_id)
                            else:
                                recep_id, _ = self.get_obj_id_from_name(receptacle_name, exclude_obj_id=exclude_obj_id)
                        else:
                            if i == 0:
                                recep_id, _ = self.get_obj_id_from_name(receptacle_name, exclude_obj_id=exclude_obj_id)
                            else:
                                recep_id, _ = self.get_obj_id_from_name(receptacle_name, get_inherited=True, exclude_obj_id=exclude_obj_id)

                    if not recep_id:
                        target_not_found = True
                        ret_msg = self._format_reason(
                            'put_target_not_found',
                            (
                                f'Could not find receptacle {receptacle_name}. '
                                'It may be closed, out of view, or a different instance may need to be targeted.'
                            )
                        )
                        continue

                    print(f'put {holding_obj_id} on {recep_id}')
                    last_target_obj = self._get_object_by_id(recep_id)

                    # look up (put action fails when a receptacle is not visible)
                    if j == 1:
                        super().step(dict(action="LookUp"))
                        super().step(dict(action="LookUp"))
                    elif j == 2:
                        super().step(dict(action="LookDown"))
                        super().step(dict(action="LookDown"))
                        super().step(dict(action="LookDown"))
                        super().step(dict(action="LookDown"))
                    elif j == 3:
                        super().step(dict(action="LookUp"))
                        super().step(dict(action="LookUp"))
                        super().step(dict(action="MoveBack"))
                    elif j == 4:
                        super().step(dict(action="MoveAhead"))
                        for r in range(4):
                            super().step(dict(action="MoveRight"))
                    elif j == 5:
                        for r in range(8):
                            super().step(dict(action="MoveLeft"))
                    elif j == 6:
                        for r in range(4):
                            super().step(dict(action="MoveRight"))
                        super().step(dict(  # this somehow make putobject success in some cases
                            action="RotateHand",
                            x=40
                        ))

                    super().step(dict(action="PutObject",objectId=holding_obj_id, receptacleObjectId=recep_id, forceAction=True))
                    last_recep_id = recep_id

                    if not self.last_event.metadata['lastActionSuccess']:
                        logging.warning(f"PutObject action failed: {self.last_event.metadata['errorMessage']}, trying again...")
                        last_error = self.last_event.metadata.get('errorMessage', 'PutObject failed.')
                        ret_msg = self._format_interaction_failure(
                            'put_api_failed',
                            'put the held object into',
                            target_obj=last_target_obj,
                            target_name=receptacle_name,
                            raw_error=last_error,
                            extra_context=f'Held object: {holding_obj_type} ({holding_obj_id}).',
                        )
                    else:
                        ret_msg = ''
                        halt = True
                        break
                if halt:
                    break
            if halt:
                break

        if not halt and target_not_found and len(last_error) == 0:
            return ret_msg

        return ret_msg

    def drop(self):
        log.info(f'drop')
        ret_msg = ''
        super().step(dict(
            action="DropHandObject",
            forceAction=True
        ))

        if not self.last_event.metadata['lastActionSuccess']:
            if len(self.last_event.metadata['inventoryObjects']) == 0:
                ret_msg = self._format_reason(
                    'not_holding_object',
                    'Robot is not holding any object, so there is nothing to drop.'
                )
            else:
                holding_obj = self.last_event.metadata['inventoryObjects'][0]
                ret_msg = self._format_interaction_failure(
                    'drop_api_failed',
                    'drop',
                    target_name=f'held object {holding_obj["objectType"]} ({holding_obj["objectId"]})',
                    raw_error=self.last_event.metadata.get('errorMessage', 'DropHandObject failed.'),
                )
        else:
            ret_msg = ''

        return ret_msg

    def open(self, obj_name):
        log.info(f'open {obj_name}')
        ret_msg = ''
        # obj_id, _ = self.get_obj_id_from_name(obj_name)
        # get the object location
        if '|' in obj_name:
            obj_id = obj_name
            obj_name = obj_name.split('|')[0]
        else:
            obj_id, _ = self.get_obj_id_from_name(obj_name)


        if obj_id is None:
            ret_msg = self._format_reason(
                'open_target_not_found',
                f'Cannot find {obj_name} to open. The object may be out of view or another instance may need to be targeted.'
            )
        else:
            open_flag = False
            target_obj = self._get_object_by_id(obj_id)
            last_error = ''
            for ob in self.last_event.metadata['objects']:
                if ob['objectId'] == obj_id and ob['openable'] and ob['isOpen']:
                    open_flag = True
                    break

            for i in range(4):
                super().step(dict(
                    action="OpenObject",
                    objectId=obj_id,
                ))

                if not self.last_event.metadata['lastActionSuccess']:
                    last_error = self.last_event.metadata.get('errorMessage', 'OpenObject failed.')
                    target_obj = self._get_object_by_id(obj_id)
                    log.warning(
                        f"OpenObject action failed: {last_error}, moving backward and trying again...")
                    if open_flag:
                        ret_msg = self._format_reason(
                            'already_open',
                            f'{self._format_target_label(target_obj, obj_name)} is already open. No open action is needed.'
                        )
                    else:
                        ret_msg = self._format_interaction_failure(
                            'open_api_failed',
                            'open',
                            target_obj=target_obj,
                            target_name=obj_name,
                            raw_error=last_error,
                        )

                    # move around to avoid self-collision
                    if i == 0:
                        super().step(dict(action="MoveBack"))
                    elif i == 1:
                        super().step(dict(action="MoveBack"))
                        super().step(dict(action="MoveRight"))
                    elif i == 2:
                        super().step(dict(action="MoveLeft"))
                        super().step(dict(action="MoveLeft"))
                else:
                    ret_msg = ''
                    break

        return ret_msg

    def close(self, obj_name):
        log.info(f'close {obj_name}')
        ret_msg = ''
        if '|' in obj_name:
            obj_id = obj_name
            obj_name = obj_name.split('|')[0]
        else:
            obj_id, _ = self.get_obj_id_from_name(obj_name)

        if obj_id is None:
            ret_msg = self._format_reason(
                'close_target_not_found',
                f'Cannot find {obj_name} to close. The object may be out of view or another instance may need to be targeted.'
            )
        else:
            target_obj = self._get_object_by_id(obj_id)
            super().step(dict(
                action="CloseObject",
                objectId=obj_id,
            ))

            if not self.last_event.metadata['lastActionSuccess']:
                ret_msg = self._format_interaction_failure(
                    'close_api_failed',
                    'close',
                    target_obj=target_obj,
                    target_name=obj_name,
                    raw_error=self.last_event.metadata.get('errorMessage', 'CloseObject failed.'),
                )
            
                for ob in self.last_event.metadata['objects']:
                    if ob['objectId'] == obj_id and ob['openable'] and not ob['isOpen']:
                        ret_msg = self._format_reason(
                            'already_closed',
                            f'{self._format_target_label(target_obj, obj_name)} is already closed. No close action is needed.'
                        )
                        break

        return ret_msg

    def toggleon(self, obj_name):
        log.info(f'toggle on {obj_name}')
        ret_msg = ''
        obj_id, obj_data = self.get_obj_id_from_name(obj_name, only_toggleable=True)
        if obj_id is None:
            ret_msg = self._format_reason(
                'toggle_on_target_not_found',
                f'Cannot find {obj_name} to turn on. The object may be out of view or not currently interactable.'
            )
        else:
            try:
                super().step(dict(
                    action="ToggleObjectOn",
                    objectId=obj_id,
                ))
                if not self.last_event.metadata['lastActionSuccess']:
                    if obj_data is not None and obj_data.get('isToggled', False):
                        ret_msg = self._format_reason(
                            'already_on',
                            f'{self._format_target_label(obj_data, obj_name)} is already turned on.'
                        )
                    else:
                        ret_msg = self._format_interaction_failure(
                            'toggle_on_api_failed',
                            'turn on',
                            target_obj=obj_data,
                            target_name=obj_name,
                            raw_error=self.last_event.metadata.get('errorMessage', 'ToggleObjectOn failed.'),
                        )
            except Exception as exc:
                ret_msg = self._format_interaction_failure(
                    'toggle_on_api_failed',
                    'turn on',
                    target_obj=obj_data,
                    target_name=obj_name,
                    raw_error=str(exc) or 'ToggleObjectOn failed.',
                )
                self.last_event.metadata['lastActionSuccess'] = False

        return ret_msg

    def toggleoff(self, obj_name):
        log.info(f'toggle off {obj_name}')
        ret_msg = ''
        obj_id, obj_data = self.get_obj_id_from_name(obj_name, only_toggleable=True)
        if obj_id is None:
            ret_msg = self._format_reason(
                'toggle_off_target_not_found',
                f'Cannot find {obj_name} to turn off. The object may be out of view or not currently interactable.'
            )
        else:
            super().step(dict(
                action="ToggleObjectOff",
                objectId=obj_id,
            ))

            if not self.last_event.metadata['lastActionSuccess']:
                if obj_data is not None and not obj_data.get('isToggled', False):
                    ret_msg = self._format_reason(
                        'already_off',
                        f'{self._format_target_label(obj_data, obj_name)} is already turned off.'
                    )
                else:
                    ret_msg = self._format_interaction_failure(
                        'toggle_off_api_failed',
                        'turn off',
                        target_obj=obj_data,
                        target_name=obj_name,
                        raw_error=self.last_event.metadata.get('errorMessage', 'ToggleObjectOff failed.'),
                    )

        return ret_msg

    def slice(self, obj_name):
        log.info(f'slice {obj_name}')
        ret_msg = ''
        obj_id, obj_data = self.get_obj_id_from_name(obj_name)
        if obj_id is None:
            ret_msg = self._format_reason(
                'slice_target_not_found',
                f'Cannot find {obj_name} to slice. The object may be out of view or another instance may need to be targeted.'
            )
        else:
            inventory = self.last_event.metadata.get('inventoryObjects', [])
            if len(inventory) == 0 or 'Knife' not in inventory[0].get('objectType', ''):
                return self._format_reason(
                    'missing_knife',
                    f'Cannot slice {self._format_target_label(obj_data, obj_name)} because the robot is not holding a knife.'
                )
            super().step(dict(
                action="SliceObject",
                objectId=obj_id,
            ))

            if not self.last_event.metadata['lastActionSuccess']:
                ret_msg = self._format_interaction_failure(
                    'slice_api_failed',
                    'slice',
                    target_obj=obj_data,
                    target_name=obj_name,
                    raw_error=self.last_event.metadata.get('errorMessage', 'SliceObject failed.'),
                )

        return ret_msg
