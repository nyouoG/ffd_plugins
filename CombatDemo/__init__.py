import imgui

from ff_draw.plugins import FFDrawPlugin
from .mem import CombatMem

strategy_map = {}


def register_strategy(class_job_id, is_pvp=False):
    def wrapper(func):
        strategy_map[(class_job_id, is_pvp)] = func
        return func

    return wrapper


@register_strategy(32)
def dark_knight_pve(m: CombatMem):
    if (me := m.me) is None: return 4
    if (target := m.targets.current) is None: return 5
    if not m.is_enemy(me, target): return 6
    if m.action_state.stack_has_action: return 7
    if me.level >= 86 and me.status.has_status(749) and not m.action_state.get_cool_down_by_action(25755).remain:
        return m.action_state.use_action(25755)
    if m.action_state.get_cool_down_by_action(3639).remain <= .1:  # 罩子
        return m.use_action_pos(3639, target.pos)
    gcd_remain = m.action_state.get_cool_down_by_action(3617).remain
    if gcd_remain > .5: return 8

    if me.level >= 26 and m.action_state.combo_action == 3623:
        return m.action_state.use_action(3632, target.id)
    if me.level >= 2 and m.action_state.combo_action == 3617:
        return m.action_state.use_action(3623, target.id)
    return m.action_state.use_action(3617, target.id)


class CombatDemo(FFDrawPlugin):
    def __init__(self, main):
        super().__init__(main)
        self.mem = CombatMem(self)
        self.enable = False
        self.res = 0

    def update(self, _):
        if not self.enable: return 1
        if (me := self.mem.me) is None: return 2
        is_pvp = self.mem.is_pvp
        if (strategy := strategy_map.get((me.class_job, is_pvp))) is None: return 3
        self.res = strategy(self.mem)

    def draw_panel(self):
        _, self.enable = imgui.checkbox('Enable', self.enable)
        imgui.text(f"res:{self.res}")

        if (me := self.mem.me):
            for status_id, param, remain, source_id in me.status:
                if status_id:
                    imgui.text(f"{status_id} {param} {remain} {source_id}")


