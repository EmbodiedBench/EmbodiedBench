from embodiedbench.main import logger
from embodiedbench.planner.vlm_planner import VLMPlanner


class EBNavigationPlanner(VLMPlanner):
    def __init__(
        self,
        model_name='',
        model_type='remote',
        actions=None,
        system_prompt='',
        examples=None,
        n_shot=1,
        obs_key='head_rgb',
        chat_history=False,
        language_only=False,
        multiview=False,
        multistep=False,
        visual_icl=False,
        tp=1,
        truncate=False,
        use_feedback=True,
        memory_compression=False,
        segment_len=1,
        use_easyr1_format=False,
        kwargs=None,
    ):
        unsupported_features = []
        if multiview:
            unsupported_features.append('multiview')
        if visual_icl:
            unsupported_features.append('visual_icl')
        if truncate:
            unsupported_features.append('truncate')
        if unsupported_features:
            logger.warning(
                "EBNavigationPlanner ignores unsupported legacy features: %s",
                ', '.join(unsupported_features),
            )

        planner_kwargs = dict(kwargs or {})
        planner_kwargs.setdefault('easyr1_variant', 'navigation')

        super().__init__(
            model_name=model_name,
            model_type=model_type,
            actions=actions or [],
            system_prompt=system_prompt,
            examples=examples or [],
            n_shot=n_shot,
            obs_key=obs_key,
            chat_history=chat_history,
            language_only=language_only,
            use_feedback=use_feedback,
            multistep=multistep,
            tp=tp,
            memory_compression=memory_compression,
            segment_len=segment_len,
            enable_point_actions=False,
            use_easyr1_format=use_easyr1_format,
            kwargs=planner_kwargs,
        )
