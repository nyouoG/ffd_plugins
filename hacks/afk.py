import typing

import imgui

import nylib.utils.win32.memory as ny_mem

if typing.TYPE_CHECKING:
    from . import Hacks


class Afk:
    def __init__(self, main: 'Hacks'):
        self.main = main
        mem = main.main.mem
        self.handle = mem.handle
        self.write_1 = mem.scanner.find_address("75 ? 0F 28 C7 0F 28 CF")
        self.write_2 = mem.scanner.find_address("F3 0F 11 51 ? 33 C9")
        self.write_2_original = mem.scanner.get_original_text(self.write_2, 5)
        self.preset_data = main.data.setdefault('afk', {})
        if 'enabled' in self.preset_data:
            self.is_enabled = self.preset_data['enabled']
        else:
            self.preset_data['enabled'] = self.is_enabled

        self.main.logger.debug(f'afk/write_1: {self.write_1:X}')
        self.main.logger.debug(f'afk/write_2: {self.write_2:X}')

    @property
    def is_enabled(self):
        return ny_mem.read_ubyte(self.handle, self.write_1) == 0xeb

    @is_enabled.setter
    def is_enabled(self, value):
        ny_mem.write_ubyte(self.handle, self.write_1, 0xeb if value else 0x75)
        ny_mem.write_bytes(self.handle, self.write_2, bytearray(b'\x90' * 5 if value else self.write_2_original))
        self.preset_data['enabled'] = value
        self.main.storage.save()

    def draw_panel(self):
        change, new_val = imgui.checkbox("Enabled", self.is_enabled)
        if change:
            self.is_enabled = new_val
