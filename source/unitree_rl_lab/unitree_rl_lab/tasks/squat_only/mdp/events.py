"""周期スクワットタスクのリセット時イベント。

現在このタスク独自のイベントは無い (リセットは Isaac Lab の
``reset_scene_to_default`` をそのまま使う)。
``mdp/__init__.py`` が ``from .events import *`` しているのでファイルは残す。
"""

from __future__ import annotations
