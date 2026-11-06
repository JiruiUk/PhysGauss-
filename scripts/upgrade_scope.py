"""Persistent benchmark scope shared by queue, reports and monitor."""
from pidg_benchmark import PROJECT, SCENES, read_json


def selected_scenes(queue=None):
    path = (queue if queue is not None else PROJECT / 'variants/upgrade') / 'benchmark_scope.json'
    scenes = read_json(path)['scenes'] if path.exists() else list(SCENES)
    if (not isinstance(scenes, list) or not scenes or
            any(not isinstance(scene, str) or scene not in SCENES for scene in scenes) or
            len(set(scenes)) != len(scenes)):
        raise ValueError('Invalid benchmark scope: choose unique known scenes')
    return scenes
