import ctypes
import io
import typing

import math
import pathlib
import re
import struct
import glm
import imgui

from ff_draw.mem.actor import Actor
from ff_draw.plugins import FFDrawPlugin
from ff_draw.omen import BaseOmen

from nylib.pattern import StaticPatternSearcher
from nylib.utils.win32 import memory as ny_mem
from raid_helper import utils as raid_utils, data as raid_data, get_shape_default_by_action_type


class DrawConfig(ctypes.Structure):
    _fields_ = [
        ('version', ctypes.c_uint32),
        ('flag', ctypes.c_uint32),
    ]


hook_key = '_in_game_draw_server_'


def make_shell(scanner: StaticPatternSearcher):
    def repl(m: re.Match):
        try:
            match m.group(2):
                case 'sp':
                    val, = scanner.find_point(m.group(3))
                case 'sp_nu':
                    val, = scanner.find_points(m.group(3))[0]
                case 'sa':
                    val = scanner.find_address(m.group(3))
                case 'sa_nu':
                    val = scanner.find_addresses(m.group(3))[0]
                case _:
                    raise ValueError(f'unknown shell pattern {m.group(2)}')
        except Exception as e:
            raise ValueError(f'unknown shell pattern {m.group(2)} {m.group(3)} {e}')
        return f'{m.group(1)}{val:#x}'

    work_dir = pathlib.Path(__file__).parent
    fn = work_dir / 'shell.py'
    shell_code = fn.read_text(encoding='utf-8')
    shell_code = re.sub(r'(\W)(sp(?:_nu)?|sa(?:_nu)?)\("(.*)"\)', repl, shell_code)
    # shell_code = re.sub(r'(sp(?:_nu)?|sa(?:_nu)?) =.*', lambda m: '# ' + m.group(0), shell_code)
    shell_code += f'''
def install():
    if hasattr(inject_server, {hook_key!r}):
        # return addressof(getattr(inject_server, {hook_key!r}).config)
        getattr(inject_server, {hook_key!r}).unload()
    setattr(inject_server, {hook_key!r}, (ds := DrawServer()))
    return addressof(ds.config)

res = install()
'''
    # (work_dir / 'compiled_shell.py').write_text(shell_code, encoding='utf-8')
    return shell_code, fn


empty_color = b'\x00\x00\x00\x00'


def vec4f_to_vec4i8(v: glm.vec4):
    return struct.pack('4B', int(v.x * 255), int(v.y * 255), int(v.z * 255), int(v.w * 255))


pi_2 = math.pi / 2
omen_scale_adjust = glm.vec3(1, 5, 1)
omen_scale_adjust_rect = glm.vec3(.5, 5, 1)
omen_color_adjust = glm.vec4(1, 1, 1, 5)

circle_shape = lambda t=0: donut_shape(t, 1) if t else b'vfx/omen/eff/ffd/circle.avfx'


def rect_shape(t=0):
    if t == 0:
        return b'vfx/omen/eff/ffd/rect.avfx'
    elif t == 1:
        return b'vfx/omen/eff/ffd/rect2.avfx'
    else:
        return b'c!vfx/omen/eff/ffd/rect2.avfx'


fan_shape = lambda degree: b'vfx/omen/eff/ffd/fan_%d.avfx' % degree
donut_shape = lambda inner, outer: b'vfx/omen/eff/ffd/donut_%d.avfx' % int(inner / outer * 0xffff)
fan_regex = re.compile(r'fan_(\d+).avfx$'.encode())
donut_regex = re.compile(r'donut_(\d+)(?:_(\d+))?.avfx$'.encode())
is_shape_circle = lambda shape: shape >> 16 == 1 if isinstance(shape, int) else (shape == b'vfx/omen/eff/ffd/circle.avfx' or shape.startswith(b'vfx/omen/eff/ffd/donut'))
is_shape_rect = lambda shape: shape >> 16 == 2 if isinstance(shape, int) else (shape.startswith(b'c!') or shape.startswith(b'vfx/omen/eff/ffd/rect'))
is_shape_fan = lambda shape: shape >> 16 == 5 if isinstance(shape, int) else shape.startswith(b'vfx/omen/eff/ffd/fan')


def draw_circle(
        radius: typing.Callable[[BaseOmen], float] | float,
        pos: typing.Callable[[BaseOmen], glm.vec3] | glm.vec3 | Actor,
        color: typing.Callable[[BaseOmen], glm.vec4 | str] | glm.vec4 | str = None,
        surface_color: typing.Callable[[BaseOmen], glm.vec4 | str] | glm.vec4 | str = None,
        line_color: typing.Callable[[BaseOmen], glm.vec4 | str] | glm.vec4 | str = None,
        label: typing.Callable[[BaseOmen], str] | str = '',
        label_color: typing.Callable[[BaseOmen], tuple[float, float, float,]] | tuple[float, float, float,] = None,
        duration: float = 0,
        inner_radius: typing.Callable[[BaseOmen], float] | float = 0,
        alpha: typing.Callable[[BaseOmen], float] | float = None,
):
    if isinstance(radius, int):
        scale = glm.vec3(radius, 1, radius)
    else:
        def scale(o: BaseOmen):
            r = o.get_maybe_callable(radius)
            return glm.vec3(r, 1, r)
    if isinstance(radius, int) and isinstance(inner_radius, int):
        if inner_radius:
            shape = b'vfx/omen/eff/ffd/donut_%d.avfx' % int(inner_radius / radius * 0xffff)
        else:
            shape = b'vfx/omen/eff/ffd/circle.avfx'
    else:
        def shape(o: BaseOmen):
            if _inner_radius := o.get_maybe_callable(radius):
                return b'vfx/omen/eff/ffd/donut_%d.avfx' % int(_inner_radius / o.get_maybe_callable(radius) * 0xffff)
            else:
                return b'vfx/omen/eff/ffd/circle.avfx'
    return raid_utils.create_game_omen(
        pos=pos, shape=shape, scale=scale, surface_color=surface_color, line_color=line_color,
        color=color, label=label, label_color=label_color, duration=duration, alpha=alpha,
    )


def draw_rect(
        width: typing.Callable[[BaseOmen], float] | float,
        length: typing.Callable[[BaseOmen], float] | float,
        pos: typing.Callable[[BaseOmen], glm.vec3] | glm.vec3 | Actor,
        facing: typing.Callable[[BaseOmen], float] | float | Actor = None,
        color: typing.Callable[[BaseOmen], glm.vec4 | str] | glm.vec4 | str = None,
        surface_color: typing.Callable[[BaseOmen], glm.vec4 | str] | glm.vec4 | str = None,
        line_color: typing.Callable[[BaseOmen], glm.vec4 | str] | glm.vec4 | str = None,
        label: typing.Callable[[BaseOmen], str] | str = '',
        label_color: typing.Callable[[BaseOmen], tuple[float, float, float,]] | tuple[float, float, float,] = None,
        duration: float = 0,
        arg=0,  # 0:normal, 1:include back, 2:cross
        alpha: typing.Callable[[BaseOmen], float] | float = None,
):
    if isinstance(width, int) and isinstance(length, int):
        scale = glm.vec3(width, 1, length)
    else:
        def scale(o: BaseOmen):
            return glm.vec3(o.get_maybe_callable(width), 1, o.get_maybe_callable(length))
    return raid_utils.create_game_omen(
        pos=pos,
        shape=rect_shape(arg),
        scale=scale,
        facing=facing,
        surface_color=surface_color, line_color=line_color, color=color,
        label=label, label_color=label_color,
        duration=duration,
        alpha=alpha,
    )


def draw_fan(
        degree: typing.Callable[[BaseOmen], float] | float,
        radius: typing.Callable[[BaseOmen], float] | float,
        pos: typing.Callable[[BaseOmen], glm.vec3] | glm.vec3 | Actor,
        facing: typing.Callable[[BaseOmen], float] | float | Actor = None,
        color: typing.Callable[[BaseOmen], glm.vec4 | str] | glm.vec4 | str = None,
        surface_color: typing.Callable[[BaseOmen], glm.vec4 | str] | glm.vec4 | str = None,
        line_color: typing.Callable[[BaseOmen], glm.vec4 | str] | glm.vec4 | str = None,
        label: typing.Callable[[BaseOmen], str] | str = '',
        label_color: typing.Callable[[BaseOmen], tuple[float, float, float,]] | tuple[float, float, float,] = None,
        duration: float = 0,
        alpha: typing.Callable[[BaseOmen], float] | float = None,
):
    if isinstance(radius, int):
        scale = glm.vec3(radius, 1, radius)
    else:
        def scale(o: BaseOmen):
            r = o.get_maybe_callable(radius)
            return glm.vec3(r, 1, r)

    if isinstance(degree, int):
        shape = fan_shape(degree)
    else:
        def shape(o: BaseOmen):
            return fan_shape(o.get_maybe_callable(degree))

    return raid_utils.create_game_omen(
        pos=pos,
        shape=shape,
        scale=scale,
        facing=facing,
        surface_color=surface_color, line_color=line_color, color=color,
        label=label, label_color=label_color,
        duration=duration,
        alpha=alpha,
    )


class InGameDraw(FFDrawPlugin):
    def __init__(self, main):
        super().__init__(main)
        self.main.gui.add_3d_shape, self.old_add_3d_shape = self.on_add_3d_shape, self.main.gui.add_3d_shape
        BaseOmen.render, self.old_omen_render = (lambda s, c=False: self.omen_render(s, c)), BaseOmen.render
        BaseOmen.destroy, self.old_omen_destory = (lambda s: self.omen_destroy(s)), BaseOmen.destroy

        raid_utils.draw_circle = draw_circle

        self.is_install = False
        self.draw_data = io.BytesIO()
        self.avfx_data = io.BytesIO()

        self.p_config = 0
        self._flag = 0
        self.hook_raid_utils()

    def hook_raid_utils(self):
        self.old_raid_utils_circle_shape, raid_utils.circle_shape = raid_utils.circle_shape, circle_shape
        self.old_raid_utils_rect_shape, raid_utils.rect_shape = raid_utils.rect_shape, rect_shape
        self.old_raid_utils_fan_shape, raid_utils.fan_shape = raid_utils.fan_shape, fan_shape
        self.old_raid_utils_donut_shape, raid_utils.donut_shape = raid_utils.donut_shape, donut_shape
        self.old_raid_utils_is_shape_circle, raid_utils.is_shape_circle = raid_utils.is_shape_circle, is_shape_circle
        self.old_raid_utils_is_shape_rect, raid_utils.is_shape_rect = raid_utils.is_shape_rect, is_shape_rect
        self.old_raid_utils_is_shape_fan, raid_utils.is_shape_fan = raid_utils.is_shape_fan, is_shape_fan
        self.old_raid_utils_draw_circle, raid_utils.draw_circle = raid_utils.draw_circle, draw_circle
        self.old_raid_utils_draw_rect, raid_utils.draw_rect = raid_utils.draw_rect, draw_rect
        self.old_raid_utils_draw_fan, raid_utils.draw_fan = raid_utils.draw_fan, draw_fan
        for k, v in list(raid_data.special_actions.items()):
            if isinstance(v, int):
                match v >> 16:
                    case 1:  # circle/donut
                        # raid_data.special_actions[k] = circle_shape(v & 0xffff)
                        dict.__setitem__(raid_data.special_actions, k, circle_shape((v & 0xffff) / 0xffff))
                    case 2:  # rect
                        # raid_data.special_actions[k] = rect_shape(v & 0xffff)
                        dict.__setitem__(raid_data.special_actions, k, rect_shape(v & 0xffff))
                    case 5:  # fan
                        # raid_data.special_actions[k] = fan_shape(v & 0xffff)
                        dict.__setitem__(raid_data.special_actions, k, fan_shape(v & 0xffff))

    def unhook_raid_utils(self):
        raid_utils.circle_shape = self.old_raid_utils_circle_shape
        raid_utils.rect_shape = self.old_raid_utils_rect_shape
        raid_utils.fan_shape = self.old_raid_utils_fan_shape
        raid_utils.donut_shape = self.old_raid_utils_donut_shape
        raid_utils.is_shape_circle = self.old_raid_utils_is_shape_circle
        raid_utils.is_shape_rect = self.old_raid_utils_is_shape_rect
        raid_utils.is_shape_fan = self.old_raid_utils_is_shape_fan
        raid_utils.draw_circle = self.old_raid_utils_draw_circle
        raid_utils.draw_rect = self.old_raid_utils_draw_rect
        raid_utils.draw_fan = self.old_raid_utils_draw_fan
        for k, v in list(raid_data.special_actions.items()):
            if isinstance(v, bytes):
                shape = v[2:] if (is_cross := v.startswith(b'c!')) else v
                if shape.startswith(b'vfx/omen/eff/ffd/'):
                    fn = shape[17:]
                    if fn == b'circle.avfx':
                        dict.__setitem__(raid_data.special_actions, k, 0x10000)
                    if fn == b'rect.avfx':
                        # raid_data.special_actions[k] = 0x20000
                        dict.__setitem__(raid_data.special_actions, k, 0x20000)
                    if fn == b'rect2.avfx':
                        # raid_data.special_actions[k] = 0x20002 if is_cross else 0x20001
                        dict.__setitem__(raid_data.special_actions, k, 0x20002 if is_cross else 0x20001)
                    if m := fan_regex.match(fn):
                        # raid_data.special_actions[k] = 0x50000 | int(m.group(1))
                        dict.__setitem__(raid_data.special_actions, k, 0x50000 | int(m.group(1)))
                    if m := donut_regex.match(fn):
                        # raid_data.special_actions[k] = 0x10000 | int(m.group(1))
                        dict.__setitem__(raid_data.special_actions, k, 0x10000 | int(m.group(1)))
        get_shape_default_by_action_type.cache_clear()

    def unload(self):
        try:
            self.main.gui.add_3d_shape = self.old_add_3d_shape
            self.main.gui.window_manager.draw_window.update = self.old_game_window_update
            BaseOmen.render = self.old_omen_render
            self.unhook_raid_utils()
            get_shape_default_by_action_type.cache_clear()
        except Exception as e:
            self.logger.error(f'unload failed: {e}', exc_info=True)

    def omen_destroy(self, omen: BaseOmen):
        if (oid := getattr(omen, '__avfx_id', None)) is not None:
            self.avfx_data.write(struct.pack("<II", oid, 0))
        if (oid := getattr(omen, '__avfx_id_x', None)) is not None:
            self.avfx_data.write(struct.pack("<II", oid, 0))
        self.old_omen_destory(omen)

    def omen_render(self, omen: BaseOmen, cross=False):
        if not omen.working:
            return self.omen_destroy(omen)
        if not omen.shape:
            return
        if not isinstance(omen.shape, bytes) or not omen.surface_color: return self.old_omen_render(omen, cross)
        shape = omen.shape[2:] if (is_cross := omen.shape.startswith(b'c!')) else omen.shape
        self._omen_render(omen, shape, False)
        if is_cross:
            self._omen_render(omen, shape, True)

    def _omen_render(self, omen: BaseOmen, shape: bytes, cross=False):
        id_key = '__avfx_id_x' if cross else '__avfx_id'
        facing = omen.facing + pi_2 if cross else omen.facing
        color = omen.surface_color * omen_color_adjust
        scale = omen.scale * omen_scale_adjust_rect if b'rect' in shape else omen.scale * omen_scale_adjust
        if hasattr(omen, id_key):
            omen_id = getattr(omen, id_key)
            self.avfx_data.write(struct.pack("<II", omen_id, 2))
            self.avfx_data.write(struct.pack("<64s12s12sf16s", shape, omen.pos.to_bytes(), scale.to_bytes(), facing, color.to_bytes()))

        else:
            omen_id = self.main.mem.inject_handle.run(
                f"res = -1\nif hasattr(inject_server, {hook_key!r}):\n    res = getattr(inject_server, {hook_key!r}).create_omen(*args)",
                shape, omen.pos.to_bytes(), scale.to_bytes(), facing, color.to_bytes()
            )
            if omen_id != -1:
                setattr(omen, id_key, omen_id)
                self.avfx_data.write(struct.pack("<II", omen_id, 1))

    def on_add_3d_shape(self, shape, transform: glm.mat4, surface_color=None, line_color=None, line_width=3.0, point_color=None, point_size=5.0):
        if shape >> 16 == 9:
            return self.old_add_3d_shape(shape, transform, surface_color, line_color, line_width, point_color, point_size)
        self.draw_data.write(struct.pack('I', shape))
        self.draw_data.write(transform.to_bytes())
        self.draw_data.write(vec4f_to_vec4i8(surface_color) if surface_color else empty_color)
        self.draw_data.write(vec4f_to_vec4i8(line_color) if line_color else empty_color)
        # return self.old_add_3d_shape(shape, transform, surface_color, line_color, line_width, point_color, point_size)

    def update(self, main):
        if not self.is_install:
            try:
                self.old_game_window_update = self.main.gui.window_manager.draw_window.update
                self.main.gui.window_manager.draw_window.update = self.on_game_window_update
                shell, file_name = make_shell(self.main.mem.scanner)
                self.p_config = self.main.mem.inject_handle.run(shell.encode('utf-8'), filename=file_name)
                config = ny_mem.read_memory(self.main.mem.handle, DrawConfig, self.p_config)
                config.version = self.data.setdefault('version', 2)
                config.flag = self.data.setdefault('flag', 0)
                ny_mem.write_memory(self.main.mem.handle, self.p_config, config)
            except Exception as e:
                self.logger.error(f'install failed: {e}', exc_info=True)
            finally:
                self.is_install = True

    def draw_panel(self):
        if self.p_config:
            config = ny_mem.read_memory(self.main.mem.handle, DrawConfig, self.p_config)
            hash_change = False

            _hash_change, _new_val = imgui.slider_int('version', config.version, 1, 2)
            if _hash_change:
                hash_change = True
                config.version = _new_val
                self.data['version'] = _new_val

            _hash_change, is_check = imgui.checkbox('FLAG_RECT_LINE', bool(config.flag & 1))
            if _hash_change:
                hash_change = True
                if is_check:
                    config.flag |= 1
                else:
                    config.flag &= ~1
                self.data['flag'] = config.flag

            if hash_change:
                self.storage.save()
                ny_mem.write_memory(self.main.mem.handle, self.p_config, config)

    def on_game_window_update(self):
        self.main.mem.inject_handle.run(f"if hasattr(inject_server, {hook_key!r}):getattr(inject_server, {hook_key!r}).swap(*args)", self.draw_data.getvalue(), self.avfx_data.getvalue())
        self.draw_data = io.BytesIO()
        self.avfx_data = io.BytesIO()
        return self.old_game_window_update()
