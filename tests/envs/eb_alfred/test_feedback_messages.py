import importlib
import sys
import types
import unittest


def _install_stub_dependencies():
    gym_mod = types.ModuleType("gym")

    class DummyEnv:
        pass

    class DummyDiscrete:
        def __init__(self, n):
            self.n = n

    gym_mod.Env = DummyEnv
    gym_mod.spaces = types.SimpleNamespace(Discrete=DummyDiscrete)
    sys.modules["gym"] = gym_mod

    numpy_mod = types.ModuleType("numpy")
    numpy_mod.array = lambda value: value
    sys.modules["numpy"] = numpy_mod

    scipy_mod = types.ModuleType("scipy")
    spatial_mod = types.ModuleType("scipy.spatial")

    class DummyKDTree:
        def __init__(self, *_args, **_kwargs):
            pass

        def query(self, *_args, **_kwargs):
            return 0, [0]

    spatial_mod.KDTree = DummyKDTree
    scipy_mod.spatial = spatial_mod
    sys.modules["scipy"] = scipy_mod
    sys.modules["scipy.spatial"] = spatial_mod

    pil_mod = types.ModuleType("PIL")
    image_mod = types.ModuleType("PIL.Image")
    image_mod.fromarray = lambda value: value
    draw_mod = types.ModuleType("PIL.ImageDraw")
    draw_mod.Draw = lambda *_args, **_kwargs: types.SimpleNamespace(text=lambda *_a, **_k: None)
    font_mod = types.ModuleType("PIL.ImageFont")

    class DummyFont:
        def getsize(self, text):
            return (len(text), 10)

    font_mod.truetype = lambda *_args, **_kwargs: DummyFont()
    font_mod.load_default = lambda: DummyFont()
    pil_mod.Image = image_mod
    pil_mod.ImageDraw = draw_mod
    pil_mod.ImageFont = font_mod
    sys.modules["PIL"] = pil_mod
    sys.modules["PIL.Image"] = image_mod
    sys.modules["PIL.ImageDraw"] = draw_mod
    sys.modules["PIL.ImageFont"] = font_mod

    thor_env_mod = types.ModuleType("embodiedbench.envs.eb_alfred.env.thor_env")

    class DummyThorEnv:
        def __init__(self, *_args, **_kwargs):
            pass

        def step(self, action):
            if hasattr(self, "_step_handler"):
                return self._step_handler(action)
            return getattr(self, "last_event", None)

    thor_env_mod.ThorEnv = DummyThorEnv
    sys.modules["embodiedbench.envs.eb_alfred.env.thor_env"] = thor_env_mod
    env_pkg = types.ModuleType("embodiedbench.envs.eb_alfred.env")
    env_pkg.thor_env = thor_env_mod
    sys.modules["embodiedbench.envs.eb_alfred.env"] = env_pkg

    constants_mod = types.ModuleType("embodiedbench.envs.eb_alfred.gen.constants")
    constants_mod.X_DISPLAY = "1"
    constants_mod.DETECTION_SCREEN_HEIGHT = 300
    constants_mod.DETECTION_SCREEN_WIDTH = 300
    constants_mod.BUILD_PATH = ""
    gen_pkg = types.ModuleType("embodiedbench.envs.eb_alfred.gen")
    gen_pkg.constants = constants_mod
    sys.modules["embodiedbench.envs.eb_alfred.gen"] = gen_pkg
    sys.modules["embodiedbench.envs.eb_alfred.gen.constants"] = constants_mod

    game_util_mod = types.ModuleType("embodiedbench.envs.eb_alfred.gen.utils.game_util")
    game_util_mod.get_objects_with_name_and_prop = lambda *_args, **_kwargs: []
    gen_utils_pkg = types.ModuleType("embodiedbench.envs.eb_alfred.gen.utils")
    gen_utils_pkg.game_util = game_util_mod
    sys.modules["embodiedbench.envs.eb_alfred.gen.utils"] = gen_utils_pkg
    sys.modules["embodiedbench.envs.eb_alfred.gen.utils.game_util"] = game_util_mod

    utils_mod = types.ModuleType("embodiedbench.envs.eb_alfred.utils")
    utils_mod.natural_word_to_ithor_name = lambda value: value
    utils_mod.alfred_objs = []
    utils_mod.alfred_open_obj = []
    utils_mod.alfred_pick_obj = []
    utils_mod.alfred_slice_obj = []
    utils_mod.alfred_toggle_obj = []
    utils_mod.alfred_recep = []
    utils_mod.draw_boxes = lambda image, *_args, **_kwargs: image
    utils_mod.load_task_json = lambda *_args, **_kwargs: {}
    utils_mod.dotdict = lambda value: value
    sys.modules["embodiedbench.envs.eb_alfred.utils"] = utils_mod

    preprocess_mod = types.ModuleType("embodiedbench.envs.eb_alfred.data.preprocess")
    preprocess_mod.Dataset = object
    data_pkg = types.ModuleType("embodiedbench.envs.eb_alfred.data")
    data_pkg.preprocess = preprocess_mod
    sys.modules["embodiedbench.envs.eb_alfred.data"] = data_pkg
    sys.modules["embodiedbench.envs.eb_alfred.data.preprocess"] = preprocess_mod

    main_mod = types.ModuleType("embodiedbench.main")
    main_mod.logger = types.SimpleNamespace(info=lambda *_args, **_kwargs: None)
    sys.modules["embodiedbench.main"] = main_mod


_install_stub_dependencies()
sys.modules.pop("embodiedbench.envs.eb_alfred.thor_connector", None)
sys.modules.pop("embodiedbench.envs.eb_alfred.EBAlfEnv", None)
thor_connector_module = importlib.import_module("embodiedbench.envs.eb_alfred.thor_connector")
alfred_env_module = importlib.import_module("embodiedbench.envs.eb_alfred.EBAlfEnv")

ThorConnector = thor_connector_module.ThorConnector
EBAlfEnv = alfred_env_module.EBAlfEnv


class DummyEvent:
    def __init__(self, metadata):
        self.metadata = metadata
        self.frame = [[0]]
        self.instance_segmentation_frame = None
        self.color_to_object_id = {}


class AlfredFeedbackFormattingTests(unittest.TestCase):
    def test_open_by_point_failure_includes_state_and_hint(self):
        connector = ThorConnector.__new__(ThorConnector)
        target = {
            "objectType": "Microwave",
            "objectId": "Microwave|1",
            "visible": True,
            "openable": True,
            "isOpen": False,
            "distance": 0.7,
        }
        connector.last_event = DummyEvent(
            {
                "lastActionSuccess": True,
                "errorMessage": "",
                "objects": [target],
                "inventoryObjects": [],
            }
        )
        connector._resolve_point_candidates = lambda *_args, **_kwargs: (
            {
                "x": 312,
                "y": 184,
                "used_radius": 2,
                "matched_candidates": [target],
                "candidates": [target],
            },
            None,
        )

        def step_handler(action):
            connector.last_event.metadata.update(
                {
                    "lastActionSuccess": False,
                    "errorMessage": "Object failed to open/close successfully.",
                    "objects": [target],
                }
            )
            return connector.last_event

        connector._step_handler = step_handler

        message = connector.open_by_point([312, 184], max_search_radius=3)

        self.assertIn("[open_api_failed]", message)
        self.assertIn("Failed to open Microwave", message)
        self.assertIn("point (312, 184)", message)
        self.assertIn("search radius 2", message)
        self.assertIn("visible=True", message)
        self.assertIn("openable=True", message)
        self.assertIn("isOpen=False", message)
        self.assertIn("AI2-THOR error: Object failed to open/close successfully.", message)
        self.assertIn("Try stepping back", message)

    def test_open_failure_by_name_includes_target_state_and_raw_error(self):
        connector = ThorConnector.__new__(ThorConnector)
        target = {
            "objectType": "Microwave",
            "objectId": "Microwave|1",
            "visible": True,
            "openable": True,
            "isOpen": False,
            "distance": 0.5,
        }
        connector.last_event = DummyEvent(
            {
                "lastActionSuccess": True,
                "errorMessage": "",
                "objects": [target],
                "inventoryObjects": [],
            }
        )
        connector.get_obj_id_from_name = lambda *_args, **_kwargs: ("Microwave|1", target)

        def step_handler(action):
            if action["action"] == "OpenObject":
                connector.last_event.metadata.update(
                    {
                        "lastActionSuccess": False,
                        "errorMessage": "Object failed to open/close successfully.",
                        "objects": [target],
                    }
                )
            else:
                connector.last_event.metadata.update(
                    {
                        "lastActionSuccess": True,
                        "errorMessage": "",
                        "objects": [target],
                    }
                )
            return connector.last_event

        connector._step_handler = step_handler

        message = connector.open("Microwave|1")

        self.assertIn("[open_api_failed]", message)
        self.assertIn("Failed to open Microwave", message)
        self.assertIn("Microwave|1", message)
        self.assertIn("visible=True", message)
        self.assertIn("openable=True", message)
        self.assertIn("isOpen=False", message)
        self.assertIn("AI2-THOR error: Object failed to open/close successfully.", message)
        self.assertIn("Try stepping back", message)

    def test_env_feedback_wraps_richer_invalid_message(self):
        env = EBAlfEnv.__new__(EBAlfEnv)
        env.id_to_name_dict = {}

        feedback = env.get_env_feedback(
            {
                "success": False,
                "message": (
                    "[open_api_failed] Failed to open Microwave (Microwave|1). "
                    "Object state: visible=True, openable=True, isOpen=False."
                ),
            }
        )

        self.assertEqual(
            feedback,
            "Last action is invalid. [open_api_failed] Failed to open Microwave "
            "(Microwave|1). Object state: visible=True, openable=True, isOpen=False.",
        )


if __name__ == "__main__":
    unittest.main()
