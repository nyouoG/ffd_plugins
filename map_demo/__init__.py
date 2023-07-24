import functools
import json
import math
import pathlib

import glm
import imgui

from ff_draw.gui.default_style import set_style, pop_style
from ff_draw.main import FFDraw
from ff_draw.plugins import FFDrawPlugin
from fpt4.utils.sqpack.utils import icon_path

with open(pathlib.Path(__file__).parent / 'map_range.json', 'r', encoding='utf-8') as f:
    map_range = {data['id']: {
        m['id']: sorted((m['p1'], m['p2'])) for m in data['maps']
    } for data in json.load(f)}


@functools.cache
def simple_get_map_id(tid):
    try:
        return FFDraw.instance.sq_pack.sheets.territory_type_sheet[tid].map.key
    except (KeyError, AttributeError):
        return 0


def between(a, b, n):
    return a <= n <= b or b <= n <= a


def vec_between(a, b, n):
    return all(between(v1, v2, vp) for v1, vp, v2 in zip(a, n, b))


pi_2 = math.pi / 2


def get_map_id(tid, pos: glm.vec3):
    if tid in map_range:
        for map_id, (p1, p2) in map_range[tid].items():
            if vec_between(p1, p2, pos): return map_id
    return simple_get_map_id(tid)


class MapWindow:
    def __init__(self, main: 'MapDemo'):
        self.main = main
        self.window = None

    def on_want_close(self, w):
        self.main.window = None
        self.window = None
        return True

    def before_draw_window(self, w):
        if self.window is None: self.window = w
        set_style(self.main.main.gui.panel.style_color)

    def after_draw_window(self, w):
        pop_style()

    def draw(self, w):
        self.main._draw_panel()


waymark_icons = [
    61241,  # A
    61242,  # B
    61243,  # C
    61247,  # D
    61244,  # 1
    61245,  # 2
    61246,  # 3
    61248,  # 4
]
icon_size = 36

target_icon_attack_1 = 61201
target_icon_bind_1 = 61211
target_icon_stop_1 = 61221
target_icon_square = 61231  # square, circle,cross,triangle
target_icons = [
    target_icon_attack_1 + 0,  # attack 1
    target_icon_attack_1 + 1,  # attack 2
    target_icon_attack_1 + 2,  # attack 3
    target_icon_attack_1 + 3,  # attack 4
    target_icon_attack_1 + 4,  # attack 5
    target_icon_bind_1 + 0,  # bind 1
    target_icon_bind_1 + 1,  # bind 2
    target_icon_bind_1 + 2,  # bind 3
    target_icon_stop_1 + 0,  # stop 1
    target_icon_stop_1 + 1,  # stop 2
    target_icon_square,  # square
    target_icon_square + 1,  # circle
    target_icon_square + 2,  # cross
    target_icon_square + 3,  # triangle
    target_icon_attack_1 + 5,  # attack 6
    target_icon_attack_1 + 6,  # attack 7
    target_icon_attack_1 + 7,  # attack 8
]


class MapDemo(FFDrawPlugin):
    def __init__(self, main):
        super().__init__(main)
        self.map_scale = 1
        self.map_id = 0
        self.window = None

    def on_unload(self):
        try:
            self.window.window.close()
        except:
            pass

    def _draw_panel(self):
        me = self.main.mem.actor_table.me
        if not me: return
        tid = self.main.mem.territory_info.territory_id
        size = imgui.get_window_size()[0] - imgui.get_style().window_padding[0] * 2
        m_pos = me.pos
        if map_id := get_map_id(tid, m_pos):
            red = imgui.get_color_u32_rgba(1, 0, 0, 1)
            green = imgui.get_color_u32_rgba(0, 1, 0, 1)
            blue = imgui.get_color_u32_rgba(0, 0, 1, 1)
            start_pos = glm.vec2(*imgui.get_cursor_screen_pos())
            m = self.main.sq_pack.sheets.map_sheet[map_id]
            offset = glm.vec2(m.offset_x, m.offset_y)
            _v1 = m.scale / 204800
            p2n = lambda p: (p.xz + offset) * _v1 + .5
            uv_center = p2n(m_pos)
            if map_id != self.map_id:
                self.map_scale = (128 * _v1) * 2
                self.map_id = map_id
            up_left_uv = uv_center - self.map_scale / 2
            down_right_uv = uv_center + self.map_scale / 2

            n2m = lambda n: start_pos + (n - up_left_uv) / self.map_scale * size
            scale_size = lambda v: v * _v1 / self.map_scale * size
            cnt = 0
            self.main.gui.game_image.map_image(map_id, size, size, exc_handling=1, uv0=tuple(up_left_uv), uv1=tuple(down_right_uv))
            end_pos = glm.vec2(*imgui.get_cursor_screen_pos())
            half_icon_size = icon_size / 2
            draw_list = imgui.get_window_draw_list()

            # draw waymark
            for i, icon_id in enumerate(waymark_icons):
                if (waymark := self.main.mem.marking.way_mark(i)).is_enable and vec_between(up_left_uv, down_right_uv, (dn_pos := p2n(waymark.pos))):
                    try:
                        texture, _ = self.main.gui.game_image.get_game_texture(icon_path(icon_id, True))
                    except TypeError:
                        continue
                    dm_pos = n2m(dn_pos)
                    draw_list.add_image(texture, tuple(dm_pos - half_icon_size), tuple(dm_pos + half_icon_size))

            # init target icon map
            target_icon_map = {}
            for i, icon_id in enumerate(target_icons):
                if target_id := self.main.mem.marking.head_mark_target(i + 1):
                    target_icon_map[target_id] = icon_id
            # iterate actor and draw
            for actor in self.main.mem.actor_table.iter_actor_by_type(1):
                pos = actor.pos
                if get_map_id(tid, pos) != map_id: continue
                cnt += 1
                dn_pos = p2n(pos)
                if not vec_between(up_left_uv, down_right_uv, dn_pos): continue
                dm_pos = n2m(dn_pos)
                color = green if actor.address == me.address else red
                if icon_id := target_icon_map.get(actor.id):
                    try:
                        texture, _ = self.main.gui.game_image.get_game_texture(icon_path(icon_id, True))
                    except TypeError:
                        draw_list.add_text(*dm_pos, color, f"{icon_id}")
                    else:
                        draw_list.add_image(texture, tuple(dm_pos - half_icon_size), tuple(dm_pos + half_icon_size))
                draw_list.add_circle(*dm_pos, 5, color, 12, 2)
                facing = actor.facing - pi_2  # the map is rotated 90 degrees
                draw_list.add_line(*dm_pos, *(dm_pos + glm.vec2(glm.cos(facing), -glm.sin(facing)) * 10), color, 2)

            draw_list.add_circle(*n2m(p2n(m_pos)), scale_size(128), blue, 0, 2)  # 加载圈

            imgui.set_cursor_pos((end_pos.x, end_pos.y))
            imgui.text(f'{cnt=}')

            _, self.map_scale = imgui.slider_float('scale', self.map_scale, 0.01, 1, '%.2f', .01)
        if 0:
            imgui.text(f'{tid=} {map_id=} pos={m_pos.x:.2f},{m_pos.y:.2f},{m_pos.z:.2f}')
            if tid in map_range:
                for map_id, (p1, p2) in map_range[tid].items():
                    is_select = all(v1 >= vp >= v2 for v1, vp, v2 in zip(p1, m_pos, p2))
                    msg = f'{map_id=} p1={p1[0]:.2f},{p1[1]:.2f},{p1[2]:.2f} p2={p2[0]:.2f},{p2[1]:.2f},{p2[2]:.2f}'
                    if is_select:
                        imgui.text(f'[{msg}]')
                    else:
                        imgui.text(msg)

    def draw_panel(self):
        changed, new_val = imgui.checkbox('sub_window', self.window is not None)
        if changed:
            if new_val:
                self.window = MapWindow(self)
                self.main.gui.window_manager.new_window(
                    'map window',
                    self.window.draw,
                    before_window_draw=self.window.before_draw_window,
                    after_window_draw=self.window.after_draw_window,
                    on_want_close=self.window.on_want_close,
                )
            else:
                try:
                    self.window.window.close()
                except:
                    pass
        if self.window is None:
            self._draw_panel()
