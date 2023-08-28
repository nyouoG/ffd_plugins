import ctypes
import io
import pathlib
import re
import struct
import glm
import imgui

from ff_draw.plugins import FFDrawPlugin

from nylib.pattern import StaticPatternSearcher
from nylib.utils.win32 import memory as ny_mem


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
        except KeyError:
            raise ValueError(f'unknown shell pattern {m.group(2)} {m.group(3)}')
        return f'{m.group(1)}{val:#x}'

    work_dir = pathlib.Path(__file__).parent
    shell_code = (work_dir / 'shell.py').read_text(encoding='utf-8')
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
    return shell_code


empty_color = b'\x00\x00\x00\x00'


def vec4f_to_vec4i8(v: glm.vec4):
    return struct.pack('4B', int(v.x * 255), int(v.y * 255), int(v.z * 255), int(v.w * 255))


class InGameDraw(FFDrawPlugin):
    def __init__(self, main):
        super().__init__(main)
        self.old_add_3d_shape = self.main.gui.add_3d_shape
        self.main.gui.add_3d_shape = self.on_add_3d_shape
        self.is_install = False
        self.draw_data = io.BytesIO()

        self.p_config = 0
        self._flag = 0

    def unload(self):
        try:
            self.main.gui.add_3d_shape = self.old_add_3d_shape
            self.main.gui.window_manager.draw_window.update = self.old_game_window_update
        except Exception as e:
            pass

    def on_add_3d_shape(self, shape, transform: glm.mat4, surface_color=None, line_color=None, line_width=3.0, point_color=None, point_size=5.0):
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
                shell = make_shell(self.main.mem.scanner)
                self.p_config = self.main.mem.inject_handle.run(shell.encode('utf-8'))
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
        self.main.mem.inject_handle.run(f"if hasattr(inject_server, {hook_key!r}):getattr(inject_server, {hook_key!r}).swap(args[0])", self.draw_data.getvalue())
        self.draw_data = io.BytesIO()
        return self.old_game_window_update()
