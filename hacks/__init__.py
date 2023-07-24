import importlib

import imgui

from ff_draw.plugins import FFDrawPlugin
from .cam import Cam
from .afk import Afk

modules = [
    '.cam.Cam',
    '.afk.Afk',
    '.anilock.AniLock',
]


class Hacks(FFDrawPlugin):
    def __init__(self, main):
        super().__init__(main)
        self.modules = {}
        for m in modules:
            path, cls_name = m.rsplit('.', 1)
            mod = importlib.import_module(path, __package__)
            self.modules[cls_name] = getattr(mod, cls_name)(self)

    def draw_panel(self):
        for n, m in self.modules.items():
            if imgui.collapsing_header(n)[0]:
                imgui.push_id(n)
                m.draw_panel()
                imgui.pop_id()
