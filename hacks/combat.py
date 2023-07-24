import typing

import imgui

import nylib.utils.win32.memory as ny_mem

if typing.TYPE_CHECKING:
    from . import Hacks

shell_uninstall = '''
def uninstall(key):
    if hasattr(inject_server, key):
        getattr(inject_server, key).uninstall()
        delattr(inject_server, key)
'''

shell_uninstall_multi = '''
def uninstall_multi(key):
    if hasattr(inject_server, key):
        for hook in getattr(inject_server, key):
            hook.uninstall()
        delattr(inject_server, key)
'''


class GetRadius:
    key = '__hacks_hook__get_radius__'
    # shell param = key, address
    shell = '''
def install_get_radius_hook():
    import ctypes
    if hasattr(inject_server, key):
        return ctypes.addressof(getattr(getattr(inject_server, key), 'val'))
    from nylib.hook import create_hook
    val = ctypes.c_float(0)
    hook = create_hook(
        address, ctypes.c_float, [ctypes.c_void_p, ctypes.c_ubyte]
    )(
        lambda h, *a: max(h.original(*a) + val.value, 0)
    ).install_and_enable()
    setattr(hook, 'val', val)
    setattr(inject_server, key, hook)
    return ctypes.addressof(val)
res = install_get_radius_hook()
'''

    def __init__(self, main: 'Hacks'):
        self.main = main
        self.mem = main.main.mem
        self.handle = self.mem.handle
        self.hook_addr, = self.mem.scanner.find_point("E8 * * * * F3 0F 58 F0 F3 0F 10 05 ? ? ? ?")
        self.val_addr = 0

        self.preset_data = main.data.setdefault('combat/get_radius', {})
        if 'val' in self.preset_data:
            self.val = self.preset_data['val']
        else:
            self.preset_data['val'] = -1
            self.main.storage.save()

    @property
    def val(self):
        if self.val_addr:
            return ny_mem.read_float(self.handle, self.val_addr)
        return -1

    @val.setter
    def val(self, value):
        if value == -1:
            self.mem.inject_handle.run(f'key = {repr(self.key)};\n' + shell_uninstall)
            self.val_addr = 0
        else:
            if not self.val_addr:
                self.val_addr = self.mem.inject_handle.run(f'key = {repr(self.key)}; address = {hex(self.hook_addr)};\n' + self.shell)
            ny_mem.write_float(self.handle, self.val_addr, value)
        self.preset_data['val'] = value
        self.main.storage.save()

    def draw_panel(self):
        val = self.val
        changed, new_val = imgui.checkbox("##Enabled", self.val != -1)
        if changed:
            self.val = 0 if val < 0 else -1
        if val != -1:
            imgui.same_line()
            changed, new_val = imgui.slider_float("##Value", val, 0, 4, "%.2f", .1)
            if changed: self.val = new_val


class GetActionRange:
    key = '__hacks_hook__get_action_range__'
    # shell param = key, address
    shell = '''
def install_get_action_range_hook():
    import ctypes
    if hasattr(inject_server, key):
        return ctypes.addressof(getattr(getattr(inject_server, key), 'val'))
    from nylib.hook import create_hook
    val = ctypes.c_float(0)
    hook = create_hook(
        address, ctypes.c_float, [ctypes.c_uint]
    )(
        lambda h, *a: (res := h.original(*a)) and max(res + val.value, 0)
    ).install_and_enable()
    setattr(hook, 'val', val)
    setattr(inject_server, key, hook)
    return ctypes.addressof(val)
res = install_get_action_range_hook()
'''

    def __init__(self, main: 'Hacks'):
        self.main = main
        self.mem = main.main.mem
        self.handle = self.mem.handle
        self.hook_addr, = self.mem.scanner.find_point("e8 * * * * f3 0f 11 43 ? 80 ? ?")
        self.val_addr = 0

        self.preset_data = main.data.setdefault('combat/get_action_range', {})
        if 'val' in self.preset_data:
            self.val = self.preset_data['val']
        else:
            self.preset_data['val'] = -1
            self.main.storage.save()

    @property
    def val(self):
        if self.val_addr:
            return ny_mem.read_float(self.handle, self.val_addr)
        return -1

    @val.setter
    def val(self, value):
        if value == -1:
            self.mem.inject_handle.run(f'key = {repr(self.key)};\n' + shell_uninstall)
            self.val_addr = 0
        else:
            if not self.val_addr:
                self.val_addr = self.mem.inject_handle.run(f'key = {repr(self.key)}; address = {hex(self.hook_addr)};\n' + self.shell)
            ny_mem.write_float(self.handle, self.val_addr, value)
        self.preset_data['val'] = value
        self.main.storage.save()

    def draw_panel(self):
        val = self.val
        changed, new_val = imgui.checkbox("##Enabled", self.val != -1)
        if changed:
            self.val = 0 if val < 0 else -1
        if val != -1:
            imgui.same_line()
            changed, new_val = imgui.slider_float("##Value", val, 0, 4, "%.2f", .1)
            if changed: self.val = new_val


class Speed:
    key = '__hacks_hook__speed__'
    # shell param = key, address1, address2
    shell = '''
def install_speed_hook():
    import ctypes
    if hasattr(inject_server, key):
        hook1, hook2 = getattr(inject_server, key)
        return ctypes.addressof(getattr(hook1, 'accel')), ctypes.addressof(getattr(hook1, 'speed'))
    from nylib.hook import create_hook
    accel = ctypes.c_ubyte(0)
    speed = ctypes.c_float(1)

    def on_update_speed(h, a):
        current_speed = ctypes.c_float.from_address(a + 0x44)
        if accel.value:
            current_speed.value = 1e10
        res = h.original(a)
        if speed.value != 1:
            current_speed.value *= speed.value
        return res

    def on_get_fly_speed(h, a):
        return h.original(a) * speed.value

    hook1 = create_hook(
        address1, ctypes.c_void_p, [ctypes.c_size_t]
    )(on_update_speed).install_and_enable()

    hook2 = create_hook(
        address2, ctypes.c_float, [ctypes.c_size_t]
    )(on_get_fly_speed).install_and_enable()

    setattr(hook1, 'accel', accel)
    setattr(hook1, 'speed', speed)
    setattr(inject_server, key, (hook1, hook2))
    return ctypes.addressof(accel), ctypes.addressof(speed)
res = install_speed_hook()
'''

    def __init__(self, main: 'Hacks'):
        self.main = main
        self.mem = main.main.mem
        self.handle = self.mem.handle
        self.hook_addr_1 = self.mem.scanner.find_address("40 53 48 83 EC ? 80 79 ? ? 48 8B D9 0F 84 ? ? ? ? 48 89 7C 24 ?")
        self.hook_addr_2 = self.mem.scanner.find_address("40 ? 48 83 EC ? 48 8B ? 48 8B ? FF 90 ? ? ? ? 48 85 ? 75")
        self.max_accel_addr, self.speed_addr = self.mem.inject_handle.run(f'key = {repr(self.key)}; address1 = {hex(self.hook_addr_1)};address2 = {hex(self.hook_addr_2)};\n' + self.shell)

        self.preset_data = main.data.setdefault('combat/speed', {})
        if 'max_accel' in self.preset_data:
            self.max_accel = self.preset_data['max_accel']
        else:
            self.preset_data['max_accel'] = self.max_accel
            self.main.storage.save()

        if 'speed' in self.preset_data:
            self.speed = self.preset_data['speed']
        else:
            self.preset_data['speed'] = self.speed
            self.main.storage.save()

    @property
    def max_accel(self):
        return bool(ny_mem.read_ubyte(self.handle, self.max_accel_addr))

    @max_accel.setter
    def max_accel(self, value):
        ny_mem.write_ubyte(self.handle, self.max_accel_addr, int(value))
        self.preset_data['max_accel'] = value
        self.main.storage.save()

    @property
    def speed(self):
        return ny_mem.read_float(self.handle, self.speed_addr)

    @speed.setter
    def speed(self, value):
        ny_mem.write_float(self.handle, self.speed_addr, value)
        self.preset_data['speed'] = value
        self.main.storage.save()

    def draw_panel(self):
        changed, new_val = imgui.checkbox("Max Accel", self.max_accel)
        if changed: self.max_accel = new_val
        changed, new_val = imgui.slider_float("Speed", self.speed, 0, 3, "%.2f", .1)
        if changed: self.speed = new_val
