import typing

import imgui

import nylib.utils.win32.memory as ny_mem

if typing.TYPE_CHECKING:
    from . import Hacks


class CurrentMinMax:
    def __init__(self, handle, address):
        self.handle = handle
        self.address = address

    current = property(
        lambda self: ny_mem.read_float(self.handle, self.address),
        lambda self, value: ny_mem.write_float(self.handle, self.address, value)
    )

    min = property(
        lambda self: ny_mem.read_float(self.handle, self.address + 4),
        lambda self, value: ny_mem.write_float(self.handle, self.address + 4, value)
    )

    max = property(
        lambda self: ny_mem.read_float(self.handle, self.address + 8),
        lambda self, value: ny_mem.write_float(self.handle, self.address + 8, value)
    )


class Cam:
    def __init__(self, main: 'Hacks'):
        self.main = main
        mem = main.main.mem
        self.handle = mem.handle
        self._cam_base, = mem.scanner.find_point("48 8D 0D * * * * E8 ? ? ? ? 48 83 3D ? ? ? ? ? 74 ? E8 ? ? ? ?")
        self._zoom_offset, = mem.scanner.find_val("F3 0F ? ? * * * * 48 8B ? ? ? ? ? 48 85 ? 74 ? F3 0F ? ? ? ? ? ? 48 83 C1")
        self._fov_offset, = mem.scanner.find_val("F3 0F ? ? * * * * 0F 2F ? ? ? ? ? 72 ? F3 0F ? ? ? ? ? ? 48 8B")
        self._angle_offset, = mem.scanner.find_val("F3 0F 10 B3 * * * * 48 8D ? ? ? F3 44 ? ? ? ? ? ? ? F3 44")
        self.preset_data = main.data.setdefault('cam_preset', {})
        if 'zoom' in self.preset_data:
            self.cam_zoom.min, self.cam_zoom.max = self.preset_data['zoom']['min'], self.preset_data['zoom']['max']
        else:
            self.preset_data['zoom'] = {'min': self.cam_zoom.min, 'max': self.cam_zoom.max}

        if 'fov' in self.preset_data:
            self.cam_fov.min, self.cam_fov.max = self.preset_data['fov']['min'], self.preset_data['fov']['max']
        else:
            self.preset_data['fov'] = {'min': self.cam_fov.min, 'max': self.cam_fov.max}

        if 'angle' in self.preset_data:
            self.cam_angle.min, self.cam_angle.max = self.preset_data['angle']['min'], self.preset_data['angle']['max']
        else:
            self.preset_data['angle'] = {'min': self.cam_angle.min, 'max': self.cam_angle.max}

        main.storage.save()
        self.main.logger.debug(f'cam/cam_base: {self._cam_base:X}')
        self.main.logger.debug(f'cam/zoom_offset: {self._zoom_offset:X}')
        self.main.logger.debug(f'cam/fov_offset: {self._fov_offset:X}')
        self.main.logger.debug(f'cam/angle_offset: {self._angle_offset:X}')

    @property
    def cam_zoom(self):
        return CurrentMinMax(self.handle, ny_mem.read_address(self.handle, self._cam_base) + self._zoom_offset)

    @property
    def cam_fov(self):
        return CurrentMinMax(self.handle, ny_mem.read_address(self.handle, self._cam_base) + self._fov_offset)

    @property
    def cam_angle(self):
        return CurrentMinMax(self.handle, ny_mem.read_address(self.handle, self._cam_base) + self._angle_offset)

    def draw_panel(self):
        imgui.columns(4)
        imgui.next_column()
        imgui.text("Current")
        imgui.next_column()
        imgui.text("Min")
        imgui.next_column()
        imgui.text("Max")
        imgui.next_column()
        imgui.separator()

        zoom = self.cam_zoom
        imgui.text("Zoom")
        imgui.next_column()
        changed, new_zoom_current = imgui.drag_float("##zoom_current", zoom.current, 0.1, zoom.min, zoom.max, "%.1f")
        if changed: zoom.current = new_zoom_current
        imgui.next_column()
        changed, new_zoom_min = imgui.input_float('##zoom_min', zoom.min, .5, 5, "%.1f")
        if changed:
            zoom.min = new_zoom_min
            self.preset_data['zoom']['min'] = new_zoom_min
            self.main.storage.save()
        imgui.next_column()
        changed, new_zoom_max = imgui.input_float('##zoom_max', zoom.max, .5, 5, "%.1f")
        if changed:
            zoom.max = new_zoom_max
            self.preset_data['zoom']['max'] = new_zoom_max
            self.main.storage.save()
        imgui.next_column()

        fov = self.cam_fov
        imgui.text("FOV")
        imgui.next_column()
        changed, new_fov_current = imgui.drag_float("##fov_current", fov.current, 0.1, fov.min, fov.max, "%.1f")
        if changed: fov.current = new_fov_current
        imgui.next_column()
        changed, new_fov_min = imgui.input_float('##fov_min', fov.min, .1, 1, "%.1f")
        if changed:
            fov.min = new_fov_min
            self.preset_data['fov']['min'] = new_fov_min
            self.main.storage.save()
        imgui.next_column()
        changed, new_fov_max = imgui.input_float('##fov_max', fov.max, .1, 1, "%.1f")
        if changed:
            fov.max = new_fov_max
            self.preset_data['fov']['max'] = new_fov_max
            self.main.storage.save()
        imgui.next_column()

        angle = self.cam_angle
        imgui.text("Angle")
        imgui.next_column()
        changed, new_angle_current = imgui.drag_float("##angle_current", angle.current, 0.1, angle.min, angle.max, "%.1f")
        if changed: angle.current = new_angle_current
        imgui.next_column()
        changed, new_angle_min = imgui.input_float('##angle_min', angle.min, .1, 1, "%.1f")
        if changed:
            angle.min = new_angle_min
            self.preset_data['angle']['min'] = new_angle_min
            self.main.storage.save()
        imgui.next_column()
        changed, new_angle_max = imgui.input_float('##angle_max', angle.max, .1, 1, "%.1f")
        if changed:
            angle.max = new_angle_max
            self.preset_data['angle']['max'] = new_angle_max
            self.main.storage.save()
        imgui.next_column()

        imgui.columns(1)
