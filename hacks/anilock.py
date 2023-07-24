import struct
import typing

import imgui

import nylib.utils.win32.memory as ny_mem

if typing.TYPE_CHECKING:
    from . import Hacks


class AniLock:
    def __init__(self, main: 'Hacks'):
        self.main = main
        mem = main.main.mem
        self.handle = mem.handle
        self.local_lock = mem.scanner.find_address("41 C7 45 08 ? ? ? ? EB ? 41 C7 45 08")
        if struct.unpack('f', mem.scanner.get_original_text(self.local_lock + 4, 4)) == .5:
            self.normal_lock_addr = self.local_lock + 4
            self.seal_lock_addr = self.local_lock + 14
        else:
            self.normal_lock_addr = self.local_lock + 14
            self.seal_lock_addr = self.local_lock + 4
        self.original_seal_val, = struct.unpack('f', mem.scanner.get_original_text(self.seal_lock_addr, 4))
        self.original_normal_val, = struct.unpack('f', mem.scanner.get_original_text(self.normal_lock_addr, 4))

        self.main.logger.debug(f'ani_lock/local_lock: {self.normal_lock_addr:X} (original: {self.original_normal_val:.2f})')
        self.main.logger.debug(f'ani_lock/seal_lock: {self.seal_lock_addr:X} (original: {self.original_seal_val:.2f})')

        self.sync_normal_addr = mem.scanner.find_address("41 f6 44 24 ? ? 74 ? f3") + 0xf
        self.sync_normal_original = mem.scanner.get_original_text(self.sync_normal_addr, 8)
        self.main.logger.debug(f'ani_lock/sync_normal: {self.sync_normal_addr:X}')

        # sync seal not impl

        self.preset_data = main.data.setdefault('anilock', {})
        if 'state' in self.preset_data:
            self.state = self.preset_data['state']
        else:
            self.preset_data['state'] = self.state

    def set_local(self, val):
        if val == -1:
            ny_mem.write_float(self.handle, self.normal_lock_addr, self.original_normal_val)
            ny_mem.write_float(self.handle, self.seal_lock_addr, self.original_seal_val)
        else:
            ny_mem.write_float(self.handle, self.normal_lock_addr, min(val, self.original_normal_val))
            ny_mem.write_float(self.handle, self.seal_lock_addr, min(val, self.original_seal_val))

    def set_sync(self, mode):
        if mode:
            ny_mem.write_bytes(self.handle, self.sync_normal_addr, b'\x90' * 8)
        else:
            ny_mem.write_bytes(self.handle, self.sync_normal_addr, self.sync_normal_original)

    @property
    def state(self):
        if ny_mem.read_ubyte(self.handle, self.sync_normal_addr) == 0x90:
            return ny_mem.read_float(self.handle, self.normal_lock_addr)
        return -1

    @state.setter
    def state(self, value):
        self.set_local(value)
        self.set_sync(value != -1)

    def draw_panel(self):
        state = self.state
        changed, new_val = imgui.checkbox("##Enabled", state != -1)
        imgui.same_line()
        if changed:
            if state < 0:
                self.state = .5
            else:
                self.state = -1
        if state != -1:
            changed, new_val = imgui.slider_float("##Value", state, 0, .5, "%.2f", .01)
            if changed:
                self.state = new_val
