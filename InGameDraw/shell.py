import functools
import io
import logging
import pathlib
import re
import struct
import typing
import zlib

import time
import math
from ctypes import *

import glm
from nylib.hook import create_hook

logger = logging.getLogger('InGameDraw')

address_type = c_uint64
sp_nu = lambda sig: 0
sp = lambda sig: 0
sa_nu = lambda sig: 0
sa = lambda sig: 0


# region draw_raw

class VertexInput(Structure):
    _fields_ = [
        ('slot', c_ubyte),
        ('offset', c_ubyte),
        ('format', c_ubyte),
        ('usage', c_ubyte),
    ]


CommonInput = (VertexInput * 3)(
    VertexInput(slot=0, offset=0, format=0X23, usage=0X0),  # slot 0 offset 0 format float[3] usage position
    VertexInput(slot=0, offset=12, format=0X44, usage=0X3),  # slot 0 offset 12 format ubyte[4] usage color
    VertexInput(slot=0, offset=16, format=0X22, usage=0X8),  # slot 0 offset 16 format float[2] usage uv
)

kernel_device = address_type.from_address(sp("48 ? ? * * * * 33 ? 89 b8"))
kernel_create_vertex_info = CFUNCTYPE(
    address_type,  # ret vertex info*
    c_void_p,  # device *
    c_void_p,  # VertexInput *
    c_uint,  # input size
)(sp("e8 * * * * 45 ? ? 48 89 87 ? ? ? ? 48 ? ? ? 41"))


def free_vertex_info(vertex_info):
    p_func = address_type.from_address(address_type.from_address(vertex_info).value + 0x18).value
    CFUNCTYPE(c_void_p, address_type)(p_func)(vertex_info)


SERVER_SIZE = 0xC8
server_vfunc = sp("48 ? ? * * * * 48 89 58 ? 48 ? ? 48 89 58")
server_init = CFUNCTYPE(
    c_uint8,  # is_success
    c_void_p,  # server*
    c_int32,  # is_3d
    c_int32,  # view
    c_int32,  # sub view
    c_int32,  # pass
    c_uint32,  # sub priority
    c_size_t,  # vertex_size
    c_void_p,  # vertex info
    c_void_p,  # init info size_t[3] {command_buffer_size, vertex_buffer_size, index_buffer_size}
)(sp("e8 * * * * 84 ? 0f 84 ? ? ? ? 45 ? ? 33 ? b9 ? ? ? ? e8 ? ? ? ? 48 ? ? 74 ? 48"))

server_init_info = (c_size_t * 3)(
    10 * 1024 * 1024,  # command_buffer_size
    20 * 1024 * 1024,  # vertex_buffer_size
    10 * 1024 * 1024,  # index_buffer_size
)

server_init_shader = CFUNCTYPE(
    c_void_p,  # ret void*
    c_void_p,  # server*
)(sa("48 89 5c 24 ? 57 48 ? ? ? 45 ? ? 48 ? ? 48 ? ? ? ? 41 ? ? ? e8 ? ? ? ? 33 ? c7 44 24 ? ? ? ? ? 89 7c 24 ? 4c ? ? ? ? ? ? 45 ? ? 48 89 7c 24 ? 48 ? ? ? ? 48 ? ? ? ? e8 ? ? ? ? 48 89 43"))

server_close = CFUNCTYPE(
    c_void_p,  # ret void*
    c_void_p,  # server*
)(sp_nu("e8 * * * * 48 ? ? ? 48 ? ? 74 ? 48 ? ? ba ? ? ? ? ff ? 48 89 5f"))

server_begin_frame = CFUNCTYPE(
    c_void_p,  # ret void*
    c_void_p,  # server*
)(sa("48 89 5c 24 ? 57 48 ? ? ? 33 ? 48 ? ? 48 89 b9 ? ? ? ? 45"))

server_render_frame = CFUNCTYPE(
    c_void_p,  # ret void*
    c_void_p,  # server*
)(sa("48 89 5c 24 ? 48 89 6c 24 ? 48 89 74 24 ? 57 48 ? ? ? ? ? ? 48 ? ? ? ? ? ? 48 ? ? 48 89 84 24 ? ? ? ? 65"))

server_end_frame = CFUNCTYPE(
    c_void_p,  # ret void*
    c_void_p,  # server*
)(sa("40 ? 48 ? ? ? 48 ? ? 48 ? ? ? e8 ? ? ? ? 48 ? ? ? ? 48 ? ? e8"))


class BlendState(Structure):
    _fields_ = [
        ('enable', c_uint32, 1),
        ('color_blend_operation', c_uint32, 3),
        ('color_blend_factor_src', c_uint32, 4),
        ('color_blend_factor_dst', c_uint32, 4),
        ('alpha_blend_operation', c_uint32, 3),
        ('alpha_blend_factor_src', c_uint32, 4),
        ('alpha_blend_factor_dst', c_uint32, 4),
        ('color_write_enable', c_uint32, 4),
    ]


class SamplerState(Structure):
    _fields_ = [
        ('address_u', c_uint32, 2),
        ('address_v', c_uint32, 2),
        ('address_w', c_uint32, 2),
        ('filter', c_uint32, 2),
        ('mip_lod_bias', c_uint32, 10),
        ('min_lod', c_uint32, 4),
        ('max_anisotropy', c_uint32, 3),
        ('gamma_enable', c_uint32, 1),
        ('test_func', c_uint32, 4),
    ]


class Material(Structure):
    _fields_ = [
        ('blend_state', BlendState),
        ('p_texture', address_type),
        ('sampler_state', SamplerState),
        ('texture_remap_color', c_uint32, 3),
        ('texture_remap_alpha', c_uint32, 2),
        ('depth_test_enable', c_uint32, 1),
        ('depth_write_enable', c_uint32, 1),
        ('face_cull_enable', c_uint32, 1),
        ('face_cull_mode', c_uint32, 1),
    ]


p_base_mat_2d = sp("0f ? ? * * * * 48 89 9c 24 ? ? ? ? f2")
p_base_mat_3d = p_base_mat_2d + 0x18
base_mat_2d = Material.from_buffer_copy(Material.from_address(p_base_mat_2d))
base_mat_3d = Material.from_buffer_copy(Material.from_address(p_base_mat_3d))
base_mat_2d.face_cull_enable = 0
base_mat_3d.face_cull_enable = 0


class Vertex(Structure):
    _fields_ = [
        ('position', c_float * 3),
        ('color', c_ubyte * 4),
        ('uv', c_float * 2),
    ]


context_create_command = CFUNCTYPE(
    address_type,  # ret draw command*
    c_void_p,  # context*
    c_int32,  # type
    c_uint32,  # vertex count
    c_uint,  # priority
    c_void_p,  # material
)(sp(" e8 * * * * 4c ? ? 48 ? ? 0f 84 ? ? ? ? f3 ? ? ? ? f3 ? ? ? 48"))


class DrawCommand:
    line_list = 0X10
    line_strip = 0X11
    triangle_list = 0X20
    triangle_strip = 0X21
    quad_list = 0X22
    quad_strip = 0X23


def make_material_2d(p_tex):
    mat = Material.from_buffer_copy(base_mat_2d)
    mat.p_texture = p_tex
    return mat


def make_material_3d(p_tex):
    mat = Material.from_buffer_copy(base_mat_3d)
    mat.p_texture = p_tex
    return mat


def apply_vertex_color(p_command, vertexes: list[glm.vec3 | glm.vec2] | tuple[glm.vec3, ...] | tuple[glm.vec2, ...], colors: list[glm.u8vec4] | tuple[glm.u8vec4, ...] | glm.u8vec4):
    vector_count = len(vertexes)
    if isinstance(colors, glm.u8vec4):
        colors = [colors for _ in range(vector_count)]
    elif len(colors) < vector_count:
        if not isinstance(colors, list): colors = list(colors)
        colors.extend(colors[-1] for _ in range(vector_count - len(colors)))
    p_command_ = p_command
    for vertex, color in zip(vertexes, colors):
        if len(vertex) == 2:  vertex = glm.vec3(vertex, .5)
        memmove(p_command_, vertex.to_bytes(), 12)
        memmove(p_command_ + 12, color.to_bytes(), 4)
        p_command_ += 0x18
    return p_command


def apply_vertex_color_uv(p_command, vertexes: list[glm.vec3 | glm.vec2] | tuple[glm.vec3, ...] | tuple[glm.vec2, ...], colors: list[glm.u8vec4] | tuple[glm.u8vec4, ...] | glm.u8vec4, uvs: list[glm.vec2] | tuple[glm.vec2, ...]):
    vector_count = len(vertexes)
    assert len(uvs) == vector_count
    if isinstance(colors, glm.u8vec4):
        colors = [colors for _ in range(vector_count)]
    elif len(colors) < vector_count:
        if not isinstance(colors, list): colors = list(colors)
        colors.extend(colors[-1] for _ in range(vector_count - len(colors)))
    p_command_ = p_command
    for vertex, color, uv in zip(vertexes, colors, uvs):
        if len(vertex) == 2:  vertex = glm.vec3(vertex, .5)
        memmove(p_command_, vertex.to_bytes(), 12)
        memmove(p_command_ + 12, color.to_bytes(), 4)
        memmove(p_command_ + 16, uv.to_bytes(), 8)
        p_command_ += 0x18
    return p_command


class Server:
    def __init__(self, is_3d: bool, p_server=None):
        if p_server:
            self._is_self_allocated = False
            self._server = None
            self.p_server = p_server
        else:
            self._is_self_allocated = True
            self._server = create_string_buffer(SERVER_SIZE)
            self.p_server = addressof(self._server)
            address_type.from_address(self.p_server).value = server_vfunc
            if not (vertex_info := kernel_create_vertex_info(kernel_device.value, byref(CommonInput), len(CommonInput))):
                raise RuntimeError("create vertex info failed")
            if not server_init(self._server, int(is_3d), 30, 14, 4, 0, 24, vertex_info, addressof(server_init_info)):
                free_vertex_info(vertex_info)
                raise RuntimeError("init server failed")
            server_init_shader(self._server)
            free_vertex_info(vertex_info)
        self.is_3d = is_3d
        self.default_material = addressof(base_mat_3d if is_3d else base_mat_2d)

    def get_contex(self, idx=0):
        return address_type.from_address(self.p_server + 0xB8 + idx * 8).value

    def begin_frame(self):
        if self._is_self_allocated:
            server_begin_frame(self.p_server)

    def render_frame(self):
        if self._is_self_allocated:
            server_render_frame(self.p_server)

    def end_frame(self):
        if self._is_self_allocated:
            server_end_frame(self.p_server)

    def close(self):
        if self._is_self_allocated:
            server_close(self.p_server)

    def __del__(self):
        self.close()

    def create_command(self, command, vertex_count, priority=0, p_material=None):
        if p_material is None: p_material = self.default_material
        p_vertex = context_create_command(self.get_contex(), command, vertex_count, priority, p_material)
        assert p_vertex, "draw command failed"
        return p_vertex  # cast(p_vertex, POINTER(Vertex))

    def create_line_list_command(self, vertex_count, priority=0, p_material=None):
        assert vertex_count > 0, "vertex count must be greater than 0"
        assert vertex_count % 2 == 0, "vertex count must be even"
        return self.create_command(DrawCommand.line_list, vertex_count, priority, p_material)

    def create_line_strip_command(self, vertex_count, priority=0, p_material=None):
        assert vertex_count >= 2, "vertex count must be greater than 1"
        return self.create_command(DrawCommand.line_strip, vertex_count, priority, p_material)

    def create_triangle_list_command(self, vertex_count, priority=0, p_material=None):
        assert vertex_count > 0, "vertex count must be greater than 0"
        assert vertex_count % 3 == 0, "vertex count must be multiple of 3"
        return self.create_command(DrawCommand.triangle_list, vertex_count, priority, p_material)

    def create_triangle_strip_command(self, vertex_count, priority=0, p_material=None):
        assert vertex_count >= 3, "vertex count must be greater than 2"
        return self.create_command(DrawCommand.triangle_strip, vertex_count, priority, p_material)

    def create_quad_list_command(self, vertex_count, priority=0, p_material=None):
        assert vertex_count > 0, "vertex count must be greater than 0"
        assert vertex_count % 4 == 0, "vertex count must be multiple of 4"
        return self.create_command(DrawCommand.quad_list, vertex_count, priority, p_material)

    def create_quad_strip_command(self, vertex_count, priority=0, p_material=None):
        assert vertex_count >= 4, "vertex count must be greater than 3"
        return self.create_command(DrawCommand.quad_strip, vertex_count, priority, p_material)

    def draw_command(self, command, vectors, colors, priority=0, p_material=None, uvs=None):
        p_cmd = self.create_command(command, len(vectors), priority, p_material)
        if p_material and uvs:
            return apply_vertex_color_uv(p_cmd, vectors, colors, uvs)
        else:
            return apply_vertex_color(p_cmd, vectors, colors)

    def draw_line_list(self, vectors, colors, priority=0, p_material=None, uvs=None):
        p_cmd = self.create_line_list_command(len(vectors), priority, p_material)
        if p_material and uvs:
            return apply_vertex_color_uv(p_cmd, vectors, colors, uvs)
        else:
            return apply_vertex_color(p_cmd, vectors, colors)

    def draw_line_strip(self, vectors, colors, priority=0, p_material=None, uvs=None):
        p_cmd = self.create_line_strip_command(len(vectors), priority, p_material)
        if p_material and uvs:
            return apply_vertex_color_uv(p_cmd, vectors, colors, uvs)
        else:
            return apply_vertex_color(p_cmd, vectors, colors)

    def draw_triangle_list(self, vectors, colors, priority=0, p_material=None, uvs=None):
        p_cmd = self.create_triangle_list_command(len(vectors), priority, p_material)
        if p_material and uvs:
            return apply_vertex_color_uv(p_cmd, vectors, colors, uvs)
        else:
            return apply_vertex_color(p_cmd, vectors, colors)

    def draw_triangle_strip(self, vectors, colors, priority=0, p_material=None, uvs=None):
        p_cmd = self.create_triangle_strip_command(len(vectors), priority, p_material)
        if p_material and uvs:
            return apply_vertex_color_uv(p_cmd, vectors, colors, uvs)
        else:
            return apply_vertex_color(p_cmd, vectors, colors)

    def draw_quad_list(self, vectors, colors, priority=0, p_material=None, uvs=None):
        p_cmd = self.create_quad_list_command(len(vectors), priority, p_material)
        if p_material and uvs:
            return apply_vertex_color_uv(p_cmd, vectors, colors, uvs)
        else:
            return apply_vertex_color(p_cmd, vectors, colors)

    def draw_quad_strip(self, vectors, colors, priority=0, p_material=None, uvs=None):
        p_cmd = self.create_quad_strip_command(len(vectors), priority, p_material)
        if p_material and uvs:
            return apply_vertex_color_uv(p_cmd, vectors, colors, uvs)
        else:
            return apply_vertex_color(p_cmd, vectors, colors)


def apply_transform(transform: glm.mat4, vertices):
    return [glm.vec3(transform * v) for v in vertices]


empty_vec2 = glm.vec2(0, 0).to_bytes()


def lines(line_mode: int, line_vertexes: list[glm.vec4]):
    if line_mode == DrawCommand.line_list:
        for i in range(0, len(line_vertexes), 2):
            yield line_vertexes[i].xyz, line_vertexes[i + 1].xyz
    elif line_mode == DrawCommand.line_strip:
        for i in range(len(line_vertexes) - 1):
            yield line_vertexes[i].xyz, line_vertexes[i + 1].xyz
    else:
        raise ValueError(f'line mode {line_mode} not supported')


DIR_AB = glm.vec3(0, 0, 1)  # glm.normalize(glm.vec3(0, 0, 1) - glm.vec3(0, 0, 0))
ROT_MAT_SAME = glm.rotate(math.pi / 2, glm.vec3(0, 0, 1))
DIR_AB_Rev = -DIR_AB
ROT_MAT_REV = glm.rotate(math.pi / 2, glm.vec3(0, 0, -1)) * glm.rotate(math.pi, glm.vec3(0, 1, 0))


def line_to_transforms(line_mode: int, line_vertexes: list[glm.vec4], transform, width=.1):
    for a, b in lines(line_mode, line_vertexes):
        scale = glm.scale(glm.vec3(width, 1, glm.distance(a, b)))
        DIR_T = glm.normalize(b - a)
        if DIR_T == DIR_AB:
            rotate = ROT_MAT_SAME
        elif DIR_T == DIR_AB_Rev:
            rotate = ROT_MAT_REV
        else:
            rotate = glm.mat4_cast(glm.quatLookAtLH(DIR_T, DIR_AB))
        translate = glm.translate(a)
        yield transform * translate * rotate * scale


class Model:
    FLAG_RECT_LINE = 1
    fill_mode: int = DrawCommand.triangle_list
    fill_vertexes: list[glm.vec4] = []
    line_mode: int = DrawCommand.line_list
    line_vertexes: list[glm.vec4] = []

    @classmethod
    def draw(cls, server_3d: Server, transform: glm.mat4, fill_color=None, line_color=None, priority=0, p_material=None, uvs=None, flags=0):
        if fill_color and cls.fill_vertexes:
            server_3d.draw_command(cls.fill_mode, apply_transform(transform, cls.fill_vertexes), fill_color, priority, p_material, uvs)
        if line_color and cls.line_vertexes:
            if flags & cls.FLAG_RECT_LINE:
                for _transform in line_to_transforms(cls.line_mode, cls.line_vertexes, transform):
                    server_3d.draw_command(cls.line_mode, apply_transform(_transform, cls.line_vertexes), line_color, priority, p_material, uvs)
            else:
                server_3d.draw_command(cls.line_mode, apply_transform(transform, cls.line_vertexes), line_color, priority, p_material, uvs)

    @classmethod
    def compile(cls, transform: glm.mat4, fill_color: glm.u8vec4 | list[glm.u8vec4] = None, line_color: glm.u8vec4 = None, flags=0):
        if fill_color and cls.fill_vertexes:
            buf = io.BytesIO()
            if isinstance(fill_color, list):
                color_len = len(fill_color)
                for i, v in enumerate(cls.fill_vertexes):
                    buf.write(glm.vec3(transform * v).to_bytes())
                    buf.write(fill_color[i % color_len].to_bytes())
                    buf.write(empty_vec2)
            else:
                fill_bytes = fill_color.to_bytes()
                for v in cls.fill_vertexes:
                    buf.write(glm.vec3(transform * v).to_bytes())
                    buf.write(fill_bytes)
                    buf.write(empty_vec2)
            yield cls.fill_mode, len(cls.fill_vertexes), buf.getvalue()
        if line_color and cls.line_vertexes:
            if flags & cls.FLAG_RECT_LINE:
                for transform in line_to_transforms(cls.line_mode, cls.line_vertexes, transform):
                    yield from RectF.compile(transform, line_color)
            else:
                buf = io.BytesIO()
                line_bytes = line_color.to_bytes()
                for v in cls.line_vertexes:
                    buf.write(glm.vec3(transform * v).to_bytes())
                    buf.write(line_bytes)
                    buf.write(empty_vec2)
                yield cls.line_mode, len(cls.line_vertexes), buf.getvalue()


circle_step = 100


class Circle(Model):
    line_mode = DrawCommand.line_strip
    line_vertexes = []
    for i in range(circle_step):
        _a = i / circle_step * math.pi * 2
        line_vertexes.append(glm.vec4(math.cos(_a), 0, math.sin(_a), 1))
    line_vertexes.append(line_vertexes[0])

    fill_mode = DrawCommand.triangle_list
    fill_vertexes = []
    center = glm.vec4(0, 0, 0, 1)
    for i in range(circle_step):
        fill_vertexes.append(line_vertexes[(i + 1) % circle_step])
        fill_vertexes.append(line_vertexes[i])
        fill_vertexes.append(center)


class Triangle(Model):
    fill_mode = DrawCommand.triangle_list
    fill_vertexes = [
        glm.vec4(1, 0, 0, 1),
        glm.vec4(-1 / 3 ** .5, 0, 0, 1),
        glm.vec4(0, 0, 1 / 3 ** .5, 1),
    ]
    line_mode = DrawCommand.line_strip
    line_vertexes = fill_vertexes + [fill_vertexes[0]]


class RectF(Model):
    fill_mode = DrawCommand.triangle_strip
    fill_vertexes = [
        glm.vec4(.5, 0, 1, 1),
        glm.vec4(-.5, 0, 1, 1),
        glm.vec4(.5, 0, 0, 1),
        glm.vec4(-.5, 0, 0, 1),
    ]

    line_mode = DrawCommand.line_strip
    line_vertexes = [
        fill_vertexes[0],
        fill_vertexes[1],
        fill_vertexes[3],
        fill_vertexes[2],
        fill_vertexes[0],
    ]


class RectFB(Model):
    fill_mode = DrawCommand.triangle_strip
    fill_vertexes = [
        glm.vec4(.5, 0, 1, 1),
        glm.vec4(.5, 0, -1, 1),
        glm.vec4(-.5, 0, 1, 1),
        glm.vec4(-.5, 0, -1, 1),
    ]

    line_mode = DrawCommand.line_strip
    line_vertexes = [
        fill_vertexes[0],
        fill_vertexes[1],
        fill_vertexes[3],
        fill_vertexes[2],
        fill_vertexes[0],
    ]


@functools.cache
def Donut(inner_percent):
    scale = glm.scale(glm.vec3(inner_percent, 1, inner_percent))
    inner_points = [scale * v for v in Circle.line_vertexes]

    class _Donut(Model):
        line_mode = Circle.line_mode
        line_vertexes = Circle.line_vertexes

        fill_mode = DrawCommand.triangle_strip
        fill_vertexes = []
        for i in range(circle_step):
            fill_vertexes.append(Circle.line_vertexes[i])
            fill_vertexes.append(inner_points[i])
        fill_vertexes += fill_vertexes[:2]

    return _Donut


@functools.cache
def Fan(degree):
    arc_percent = degree / 360
    start_angle = math.pi * (.5 - arc_percent)
    arc_step = math.ceil(arc_percent * circle_step)
    arc_rad = math.radians(degree)
    vertices = []
    for i in range(arc_step + 1):
        angle = start_angle + arc_rad * i / arc_step
        vertices.append(glm.vec4(math.cos(angle), 0, math.sin(angle), 1))

    class _Fan(Model):
        line_mode = DrawCommand.line_strip
        line_vertexes = [Circle.center] + vertices + [Circle.center]

        fill_mode = DrawCommand.triangle_list
        fill_vertexes = []
        for i in range(arc_step):
            fill_vertexes.append(vertices[i])
            fill_vertexes.append(vertices[i + 1])
            fill_vertexes.append(Circle.center)

    return _Fan


@functools.cache
def DonutFan(inner_percent, degree, side_line=True, inner_line=True):
    arc_percent = degree / 360
    start_angle = math.pi * (.5 - arc_percent)
    arc_step = math.ceil(arc_percent * circle_step)
    arc_rad = math.radians(degree)
    out_vertices = []
    in_vertices = []
    for i in range(arc_step + 1):
        angle = start_angle + arc_rad * i / arc_step
        cosa = math.cos(angle)
        sina = math.sin(angle)
        out_vertices.append(glm.vec4(cosa, 0, sina, 1))
        in_vertices.append(glm.vec4(cosa * inner_percent, 0, sina * inner_percent, 1))

    class _DonutFan(Model):
        line_mode = DrawCommand.line_strip
        line_vertexes = out_vertices
        if side_line:
            line_vertexes = [in_vertices[0]] + line_vertexes + [in_vertices[-1]]
            if inner_line:
                line_vertexes += list(reversed(in_vertices))

        fill_mode = DrawCommand.triangle_strip
        fill_vertexes = []
        for out_v, in_v in zip(out_vertices, in_vertices):
            fill_vertexes.append(out_v)
            fill_vertexes.append(in_v)

    return _DonutFan


class Arrow(Model):
    fill_mode = DrawCommand.triangle_list
    fill_vertexes = [
        glm.vec4(0, 0, 0, 1),
        glm.vec4(-1, 0, -1, 1),
        glm.vec4(-1, 0, 0, 1),

        glm.vec4(0, 0, 0, 1),
        glm.vec4(-1, 0, 0, 1),
        glm.vec4(0, 0, 1, 1),

        glm.vec4(0, 0, 0, 1),
        glm.vec4(0, 0, 1, 1),
        glm.vec4(1, 0, 0, 1),

        glm.vec4(0, 0, 0, 1),
        glm.vec4(1, 0, 0, 1),
        glm.vec4(1, 0, -1, 1),
    ]

    line_mode = DrawCommand.line_strip
    line_vertexes = [
        glm.vec4(0, 0, 0, 1),
        glm.vec4(-1, 0, -1, 1),
        glm.vec4(-1, 0, 0, 1),
        glm.vec4(0, 0, 1, 1),
        glm.vec4(1, 0, 0, 1),
        glm.vec4(1, 0, -1, 1),
        glm.vec4(0, 0, 0, 1),
    ]


class Line(Model):
    fill_mode = DrawCommand.triangle_strip
    fill_vertexes = [
        glm.vec4(.05, 0, 1, 1),
        glm.vec4(-.05, 0, 1, 1),
        glm.vec4(.05, 0, 0, 1),
        glm.vec4(-.05, 0, 0, 1),
    ]

class Point(Model):
    fill_vertexes = [
        glm.vec4(.1, 0, -.1, 1),  # bottom right
        glm.vec4(.1, 0, .1, 1),  # center right
        glm.vec4(-.1, 0, -.1, 1),  # bottom center
        glm.vec4(-.1, 0, .1, 1),  # center
    ]
    line_mode = DrawCommand.line_list
    line_vertexes = [
        glm.vec4(0, 0, .1, 1),
        glm.vec4(0, 0, -.1, 1),
        glm.vec4(.1, 0, 0, 1),
        glm.vec4(-.1, 0, 0, 1),
    ]


out_alpha = 1
in_alpha = 0.4
in_percent = 0.5


def _vec4_percent(v, percent):
    return glm.vec4(v.xyz * percent, v.w)


class RectFV2(Model):
    class RectFTR(RectF):
        fill_vertexes = [
            glm.vec4(.5, 0, 1, 1),  # top right
            glm.vec4(.5, 0, .5, 1),  # center right
            glm.vec4(0, 0, 1, 1),  # top center
            glm.vec4(0, 0, .5, 1),  # center
        ]
        line_vertexes = [fill_vertexes[1], fill_vertexes[0], fill_vertexes[2]]

    class RectFTL(RectF):
        fill_vertexes = [
            glm.vec4(-.5, 0, 1, 1),  # top left
            glm.vec4(-.5, 0, .5, 1),  # center left
            glm.vec4(0, 0, 1, 1),  # top center
            glm.vec4(0, 0, .5, 1),  # center
        ]
        line_vertexes = [fill_vertexes[1], fill_vertexes[0], fill_vertexes[2]]

    class RectFBR(RectF):
        fill_vertexes = [
            glm.vec4(.5, 0, 0, 1),  # bottom right
            glm.vec4(.5, 0, .5, 1),  # center right
            glm.vec4(0, 0, 0, 1),  # bottom center
            glm.vec4(0, 0, .5, 1),  # center
        ]
        line_vertexes = [fill_vertexes[1], fill_vertexes[0], fill_vertexes[2]]

    class RectFBL(RectF):
        fill_vertexes = [
            glm.vec4(-.5, 0, 0, 1),  # bottom left
            glm.vec4(-.5, 0, .5, 1),  # center left
            glm.vec4(0, 0, 0, 1),  # bottom center
            glm.vec4(0, 0, .5, 1),  # center
        ]
        line_vertexes = [fill_vertexes[1], fill_vertexes[0], fill_vertexes[2]]

    @staticmethod
    def _make_color(fill_color):
        if isinstance(fill_color, glm.u8vec4):
            a = fill_color.a
            out_color = glm.u8vec4(fill_color.xyz, int(out_alpha * a))
            in_color = glm.u8vec4(fill_color.xyz, int(in_alpha * a))
            fill_color = [out_color, out_color, out_color, in_color]
        return fill_color

    @classmethod
    def draw(cls, server_3d: Server, transform: glm.mat4, fill_color=None, line_color=None, priority=0, p_material=None, uvs=None, flags=0):
        fill_color = cls._make_color(fill_color)
        cls.RectFTR.draw(server_3d, transform, fill_color, line_color, priority, p_material, uvs, flags)
        cls.RectFTL.draw(server_3d, transform, fill_color, line_color, priority, p_material, uvs, flags)
        cls.RectFBR.draw(server_3d, transform, fill_color, line_color, priority, p_material, uvs, flags)
        cls.RectFBL.draw(server_3d, transform, fill_color, line_color, priority, p_material, uvs, flags)

    @classmethod
    def compile(cls, transform: glm.mat4, fill_color: glm.u8vec4 | list[glm.u8vec4] = None, line_color: glm.u8vec4 = None, flags=0):
        fill_color = cls._make_color(fill_color)
        yield from cls.RectFTR.compile(transform, fill_color, line_color, flags)
        yield from cls.RectFTL.compile(transform, fill_color, line_color, flags)
        yield from cls.RectFBR.compile(transform, fill_color, line_color, flags)
        yield from cls.RectFBL.compile(transform, fill_color, line_color, flags)


class RectFBV2(RectFV2):
    class RectFTR(RectF):
        fill_vertexes = [
            glm.vec4(.5, 0, 1, 1),  # top right
            glm.vec4(.5, 0, 0, 1),  # center right
            glm.vec4(0, 0, 1, 1),  # top center
            glm.vec4(0, 0, 0, 1),  # center
        ]
        line_vertexes = [fill_vertexes[1], fill_vertexes[0], fill_vertexes[2]]

    class RectFTL(RectF):
        fill_vertexes = [
            glm.vec4(-.5, 0, 1, 1),  # top left
            glm.vec4(-.5, 0, 0, 1),  # center left
            glm.vec4(0, 0, 1, 1),  # top center
            glm.vec4(0, 0, 0, 1),  # center
        ]
        line_vertexes = [fill_vertexes[1], fill_vertexes[0], fill_vertexes[2]]

    class RectFBR(RectF):
        fill_vertexes = [
            glm.vec4(.5, 0, -1, 1),  # bottom right
            glm.vec4(.5, 0, 0, 1),  # center right
            glm.vec4(0, 0, -1, 1),  # bottom center
            glm.vec4(0, 0, 0, 1),  # center
        ]
        line_vertexes = [fill_vertexes[1], fill_vertexes[0], fill_vertexes[2]]

    class RectFBL(RectF):
        fill_vertexes = [
            glm.vec4(-.5, 0, -1, 1),  # bottom left
            glm.vec4(-.5, 0, 0, 1),  # center left
            glm.vec4(0, 0, -1, 1),  # bottom center
            glm.vec4(0, 0, 0, 1),  # center
        ]
        line_vertexes = [fill_vertexes[1], fill_vertexes[0], fill_vertexes[2]]


class CircleV2(Model):
    fill_mode = DrawCommand.triangle_list
    fill_vertexes = [_vec4_percent(v, in_percent) for v in Circle.fill_vertexes]

    outer = Donut(in_percent)

    @staticmethod
    def _make_color(fill_color):
        if isinstance(fill_color, glm.u8vec4):
            a = fill_color.a
            out_color = glm.u8vec4(fill_color.xyz, int(out_alpha * a))
            in_color = glm.u8vec4(fill_color.xyz, int(in_alpha * a))
            return in_color, [out_color, in_color]
        else:
            return fill_color, fill_color

    @classmethod
    def draw(cls, server_3d: Server, transform: glm.mat4, fill_color=None, line_color=None, priority=0, p_material=None, uvs=None, flags=0):
        in_colors, out_colors = cls._make_color(fill_color)
        server_3d.draw_command(cls.fill_mode, apply_transform(transform, cls.fill_vertexes), in_colors, priority, p_material, uvs)
        cls.outer.draw(server_3d, transform, out_colors, line_color, priority, p_material, uvs, flags)

    @classmethod
    def compile(cls, transform: glm.mat4, fill_color: glm.u8vec4 | list[glm.u8vec4] = None, line_color: glm.u8vec4 = None, flags=0):
        in_colors, out_colors = cls._make_color(fill_color)
        yield from super().compile(transform, in_colors, flags=flags)
        yield from cls.outer.compile(transform, out_colors, line_color, flags=flags)


@functools.cache
def DonutV2(inner_percent):
    OutDonut = Donut(inner_percent + (1 - inner_percent) * 0.5)

    scale = glm.scale(glm.vec3(inner_percent, 1, inner_percent))
    inner_points = [scale * v for v in Circle.line_vertexes]

    class InDonut(Model):
        line_mode = Circle.line_mode
        line_vertexes = [_vec4_percent(v, inner_percent) for v in Circle.line_vertexes]

        fill_mode = DrawCommand.triangle_strip
        fill_vertexes = []
        for i in range(circle_step):
            fill_vertexes.append(OutDonut.fill_vertexes[i * 2 + 1])
            fill_vertexes.append(inner_points[i])
        fill_vertexes += fill_vertexes[:2]

    class _DonutV2(Model):
        @staticmethod
        def _make_color(fill_color):
            if isinstance(fill_color, glm.u8vec4):
                a = fill_color.a
                out_color = glm.u8vec4(fill_color.xyz, int(out_alpha * a))
                in_color = glm.u8vec4(fill_color.xyz, int(in_alpha * a))
                return [in_color, out_color], [out_color, in_color]
            else:
                return fill_color, fill_color

        @classmethod
        def draw(cls, server_3d: Server, transform: glm.mat4, fill_color=None, line_color=None, priority=0, p_material=None, uvs=None, flags=0):
            in_colors, out_colors = cls._make_color(fill_color)
            OutDonut.draw(server_3d, transform, out_colors, line_color, priority, p_material, uvs, flags=flags)
            InDonut.draw(server_3d, transform, in_colors, line_color, priority, p_material, uvs, flags=flags)

        @classmethod
        def compile(cls, transform: glm.mat4, fill_color: glm.u8vec4 | list[glm.u8vec4] = None, line_color: glm.u8vec4 = None, flags=0):
            in_colors, out_colors = cls._make_color(fill_color)
            yield from OutDonut.compile(transform, out_colors, line_color, flags=flags)
            yield from InDonut.compile(transform, in_colors, line_color, flags=flags)

    return _DonutV2


@functools.cache
def FanV2(degree):
    OutFan = DonutFan(in_percent, degree, inner_line=False)
    _InFan = Fan(degree)
    InFan = type('InFan', (_InFan,), {
        'line_vertexes': [_vec4_percent(_InFan.line_vertexes[1], in_percent), _InFan.line_vertexes[0], _vec4_percent(_InFan.line_vertexes[-2], in_percent)],
        'fill_vertexes': [_vec4_percent(v, in_percent) for v in _InFan.fill_vertexes],
    })

    arc_step = math.ceil(degree / 360 * circle_step)
    half_arc_step = (arc_step + 1) // 2
    mid_arc_delta = (out_alpha - in_alpha) / half_arc_step
    half_mid_arc_alpha = [in_alpha + mid_arc_delta * i for i in range(half_arc_step)]
    mid_arc_alpha = list(reversed((half_mid_arc_alpha[1:] if arc_step % 2 else half_mid_arc_alpha))) + half_mid_arc_alpha

    class _FanV2(Model):
        @staticmethod
        def _make_color(fill_color):
            if not isinstance(fill_color, glm.u8vec4):
                return fill_color, fill_color
            a = fill_color.a
            out_color = glm.u8vec4(fill_color.xyz, int(out_alpha * a))
            mid_colors = [glm.u8vec4(fill_color.xyz, int(a * _a)) for _a in mid_arc_alpha]
            out_colors = []
            in_colors = []
            for i in range(arc_step):
                out_colors.append(out_color)
                out_colors.append(mid_colors[i])
                in_colors.append(mid_colors[i])
                in_colors.append(mid_colors[(i + 1) % arc_step])
                in_colors.append(out_color)
            return in_colors, out_colors

        @classmethod
        def draw(cls, server_3d: Server, transform: glm.mat4, fill_color=None, line_color=None, priority=0, p_material=None, uvs=None, flags=0):
            in_colors, out_colors = cls._make_color(fill_color)
            OutFan.draw(server_3d, transform, out_colors, line_color, priority, p_material, uvs, flags=flags)
            InFan.draw(server_3d, transform, in_colors, line_color, priority, p_material, uvs, flags=flags)

        @classmethod
        def compile(cls, transform: glm.mat4, fill_color: glm.u8vec4 | list[glm.u8vec4] = None, line_color: glm.u8vec4 = None, flags=0):
            in_colors, out_colors = cls._make_color(fill_color)
            yield from OutFan.compile(transform, out_colors, line_color, flags=flags)
            yield from InFan.compile(transform, in_colors, line_color, flags=flags)

    return _FanV2


@functools.cache
def DonutFanV2(inner_percent, degree):
    arc_step = math.ceil(degree / 360 * circle_step)
    mid = inner_percent + (1 - inner_percent) * 0.5

    OutDonutFan = DonutFan(mid, degree, inner_line=False)
    _InDonutFan = DonutFan(inner_percent / mid, degree)
    InDonutFan = type('InDonutFan', (_InDonutFan,), {
        'line_vertexes': [_vec4_percent(v, mid) for v in _InDonutFan.line_vertexes[arc_step + 1:]] + [_vec4_percent(_InDonutFan.line_vertexes[1], mid)],
        'fill_vertexes': [_vec4_percent(v, mid) for v in _InDonutFan.fill_vertexes],
    })
    half_arc_step = (arc_step + 1) // 2
    mid_arc_delta = (out_alpha - in_alpha) / half_arc_step
    half_mid_arc_alpha = [in_alpha + mid_arc_delta * i for i in range(half_arc_step)]
    mid_arc_alpha = (half_mid_arc_alpha[1::-1] if arc_step % 2 else half_mid_arc_alpha[::-1]) + half_mid_arc_alpha

    class _DonutFanV2(Model):
        @staticmethod
        def _make_color(fill_color):
            if not isinstance(fill_color, (glm.u8vec3, glm.u8vec4)):
                return fill_color, fill_color
            a = fill_color.a
            out_color = glm.u8vec4(fill_color.xyz, int(out_alpha * a))
            mid_colors = [glm.u8vec4(fill_color.xyz, int(a * _a)) for _a in mid_arc_alpha]
            out_colors = []
            in_colors = []
            for c in mid_colors:
                out_colors.append(out_color)
                out_colors.append(c)
                in_colors.append(c)
                in_colors.append(out_color)
            return in_colors, out_colors

        @classmethod
        def draw(cls, server_3d: Server, transform: glm.mat4, fill_color=None, line_color=None, priority=0, p_material=None, uvs=None, flags=0):
            in_colors, out_colors = cls._make_color(fill_color)
            OutDonutFan.draw(server_3d, transform, out_colors, line_color, priority, p_material, uvs, flags=flags)
            InDonutFan.draw(server_3d, transform, in_colors, line_color, priority, p_material, uvs, flags=flags)

        @classmethod
        def compile(cls, transform: glm.mat4, fill_color: glm.u8vec4 | list[glm.u8vec4] = None, line_color: glm.u8vec4 = None, flags=0):
            in_colors, out_colors = cls._make_color(fill_color)
            yield from OutDonutFan.compile(transform, out_colors, line_color, flags=flags)
            yield from InDonutFan.compile(transform, in_colors, line_color, flags=flags)

    return _DonutFanV2


y_axis = glm.vec3(0, 1, 0)

# endregion

# region avfx_draw
res_manager = c_size_t.from_address(sp("48 ? ? * * * * f0 0f c1 8a"))
# res_manager = sp("48 ? ? * * * * f0 0f c1 8a")
vfx_dec_ref = CFUNCTYPE(
    c_uint8,  # success or not
    c_void_p,  # res_mgr
    c_void_p,  # res_handle
    c_void_p,
    c_int,
)(sp("e8 * * * * 48 ? ? ? ? ? ? 48 ? ? ? ? ? ? 48 ? ? ? 48 ? ? ? ? ? ? 48 ? ? 74"))
vfx_dec_ref = lambda *args: None
vfx_res_load = CFUNCTYPE(
    c_uint8,  # is_success
    c_void_p,  # p_vfx
    c_void_p,  # p_data
    c_uint,  # size
    c_void_p,  # parent
)(sp("e8 * * * * 48 ? ? ? ? 84 ? 40 ? ? ? b9"))
vfx_res_setup_complete = CFUNCTYPE(
    c_void_p,  # res_handle
    c_void_p,  # res_handle
)(sa("40 ? 48 ? ? ? 48 ? ? 33 ? 8b ? f0 0f c0 83"))
get_res_sync = CFUNCTYPE(
    c_void_p,
    c_void_p,  # p_res_mgr
    c_void_p,  # cate
    c_void_p,  # type
    c_void_p,  # id
    c_char_p,  # path
    c_void_p,  # extend
    c_uint8,  # unk
)(sp("e8 * * * * 48 ? ? ? ? ? ? 48 89 87 ? ? ? ? 48 ? ? ? ? 48 89 44 24"))
init_vfx_param = CFUNCTYPE(c_void_p, c_char_p)(sp("e8 * * * * f3 ? ? ? ? ? ? ? 48 ? ? ? ? ? ? 48 ? ? ? 48 ? ? ? ? c7 44 24"))
set_vfx_p1 = CFUNCTYPE(c_void_p, c_void_p, c_char)(sp("e8 * * * * b2 ? 48 ? ? e8 ? ? ? ? f6 05"))
set_vfx_p2 = CFUNCTYPE(c_void_p, c_void_p, c_char)(sp("e8 * * * * f6 05 ? ? ? ? ? 74 ? 80 3d ? ? ? ? ? 73"))
create_omen = CFUNCTYPE(
    c_int64,  # created_omen
    c_uint,  # omen_id
    c_void_p,  # pos
    c_int64,  # chara
    c_float,  # speed
    c_float,  # facing
    c_float,  # scale_x
    c_float,  # scale_y
    c_float,  # scale_z
    c_ubyte,  # is_enemy
    c_ubyte  # priority
)(sp("E8 * * * * 8D 4F ? 48 63 D1 48 89 84 D3 ? ? ? ?"))
create_vfx = CFUNCTYPE(
    c_int64,  # created_omen
    c_char_p,  # res_path
    c_void_p,  # param
    c_int8,  # attr # 2
    c_int8,  # priority # 0
    c_float,  # pos_x
    c_float,  # pos_y
    c_float,  # pos_z
    c_float,  # scale_x
    c_float,  # scale_y
    c_float,  # scale_z
    c_float,  # facing
    c_float,  # speed 1
    c_int32  # offscreen slot -1
)(sp("e8 * * * * 48 ? ? 48 ? ? 74 ? b2 ? 48 ? ? e8 ? ? ? ? b2 ? 48 ? ? e8 ? ? ? ? f6 05"))
remove_omen = CFUNCTYPE(c_void_p, c_void_p, c_void_p)(sp("e8 * * * * 4c ? ? 48 ? ? ? ? 48 ? ? ? ? ? ? e8 ? ? ? ? eb"))
trigger_omen = CFUNCTYPE(c_void_p, c_void_p, c_void_p)(sp("E8 * * * * 48 8B 4F ? 33 D2 E8 ? ? ? ? 48 89 77 ?"))
set_omen_matrix = CFUNCTYPE(c_void_p, c_void_p, c_void_p)(sp("e8 * * * * 48 ? ? e8 ? ? ? ? 85 ? 74 ? 84"))
set_omen_color = CFUNCTYPE(c_void_p, c_void_p, c_float, c_float, c_float, c_float)(sa("48 ? ? ? ? ? ? 48 ? ? 74 ? 48 ? ? ? f3 ? ? ? ? ? f3 0f 11 89"))

p = pathlib.Path(__file__).parent
pre_make_dir = p / 'pre_make'

PACK_TYPE_TO_KEY_MAP = {
    b"common": 0x00, b"bgcommon": 0x01, b"bg": 0x02, b"cut": 0x03, b"chara": 0x04,
    b"shader": 0x05, b"ui": 0x06, b"sound": 0x07, b"vfx": 0x08, b"exd": 0x0a,
    b"game_script": 0x0b, b"music": 0x0c, b"_sqpack_test": 0x12, b"_debug": 0x13,
}


def parse_path(path: str | bytes):
    if isinstance(path, str): path = path.encode('utf-8')
    c = PACK_TYPE_TO_KEY_MAP.get(path.split(b'/', 1)[0])
    *_, t = path.rsplit(b'.', 1)
    assert c is not None, f"category id is not found at {path.decode('utf-8')} ({path.split(b'/', 1)[0]})"
    assert 4 >= len(t) > 0, f"file type is not found at {path.decode('utf-8')}"
    return c, int.from_bytes(t, 'big', signed=False), zlib.crc32(path)


fan_regex = re.compile(r'fan_(\d+).avfx$'.encode())
donut_regex = re.compile(r'donut_(\d+)(?:_(\d+))?.avfx$'.encode())


def make_donut(temp, ignore_percent, fan_rad=None):
    ring_fan = struct.pack('f', (1 - math.cos(fan_rad / 2)) / 2 if fan_rad is not None else 1)
    _x = .5 * (1 - ignore_percent) / (1 + ignore_percent)
    x = struct.pack('f', _x)
    revised = struct.pack('f', 1 / (.5 + _x))
    _data = bytearray(temp).copy()
    _data[0x0184:0x0188] = revised
    _data[0x019c:0x01a0] = revised
    _data[0x179c:0x17a0] = ring_fan
    _data[0x17c8:0x17cc] = x
    _data[0x2244:0x2248] = ring_fan
    _data[0x2270:0x2274] = x
    _data[0x2cec:0x2cf0] = ring_fan
    _data[0x2d18:0x2d1c] = x
    return _data


def make_fan(temp: bytes, radian):
    ring_fan = struct.pack('f', (1 - math.cos(radian / 2)) / 2)
    scroll1 = struct.pack('f', 0.45333326 - 3.18309884 * radian)
    scroll2 = struct.pack('f', 5.40770276 + 14.22240645 * radian)
    _data = bytearray(temp).copy()
    _data[0x17bc:0x17c0] = ring_fan
    _data[0x1a90:0x1a94] = scroll1
    _data[0x1c74:0x1c78] = scroll1
    _data[0x2574:0x2578] = ring_fan
    _data[0x2848:0x284c] = scroll2
    _data[0x2a2c:0x2a30] = scroll2
    _data[0x332c:0x3330] = ring_fan
    return _data


class AvfxManager:
    instance: 'AvfxManager' = None

    def __init__(self):
        assert AvfxManager.instance is None, 'AvfxManager is singleton'
        AvfxManager.instance = self
        self.vfx_map = {}
        self.tmp_circle = (p / 'template' / 'tmp_circle.avfx').read_bytes()
        self.tmp_donut = (p / 'template' / 'tmp_donut.avfx').read_bytes()
        self.tmp_fan = (p / 'template' / 'tmp_fan.avfx').read_bytes()
        self.tmp_org_donut = (p / 'template' / 'tmp_org_donut.avfx').read_bytes()
        self.tmp_org_fan = (p / 'template' / 'tmp_org_fan.avfx').read_bytes()
        self.tmp_rect = (p / 'template' / 'tmp_rect.avfx').read_bytes()
        self.tmp_rect2 = (p / 'template' / 'tmp_rect2.avfx').read_bytes()

    def add_res(self, path: str | bytes, data: bytes | bytearray, force=False):
        if isinstance(path, str): path = path.encode('utf-8')
        assert path.endswith(b'.avfx'), 'now only support .avfx file'
        if not force and path in self.vfx_map: return
        cate_, type_, id_ = map(c_uint32, parse_path(path))
        res = get_res_sync(res_manager.value, addressof(cate_), addressof(type_), addressof(id_), path, 0, 0)
        if not force and c_uint8.from_address(res + 0XA9).value == 7:
            self.vfx_map[path] = res
            return
        c_uint8.from_address(res + 0XA8).value = 2
        c_uint8.from_address(res + 0XA9).value = 7
        p_vfx = c_void_p.from_address(res + 0xC0)
        if isinstance(data, bytes):
            buffer = (c_uint8 * len(data)).from_buffer_copy(data)
        else:
            buffer = (c_uint8 * len(data)).from_buffer(data)
        if not vfx_res_load(p_vfx, buffer, len(data), res):
            vfx_dec_ref(res_manager.value, res, 0, 0)
            raise Exception('init data fail')
        vfx_res_setup_complete(res)
        self.remove_res(path)
        self.vfx_map[path] = res

    def remove_res(self, path: str | bytes):
        if isinstance(path, str): path = path.encode('utf-8')
        if old_vfx := self.vfx_map.pop(path, None):
            vfx_dec_ref(res_manager.value, old_vfx, 0, 0)

    def unload(self):
        while self.vfx_map:
            self.remove_res(next(iter(self.vfx_map.keys())))

    def check_is_custom(self, path: bytes):
        path = path.rstrip(b'\0')
        if path in self.vfx_map:
            return
        if path.startswith(b'vfx/omen/eff/ffd/'):
            fn = path[17:]
            if fn == b'circle.avfx':
                return self.add_res(path, self.tmp_circle)
            if fn == b'rect.avfx':
                return self.add_res(path, self.tmp_rect)
            if fn == b'rect2.avfx':
                return self.add_res(path, self.tmp_rect2)
            if m := fan_regex.match(fn):
                deg = int(m.group(1))
                return self.add_res(path, make_fan(self.tmp_fan, math.radians(deg)))
            if m := donut_regex.match(fn):
                ignore_percent = int(m.group(1)) / 0xffff
                fan_rad = math.radians(int(m.group(2))) if m.group(2) else None
                return self.add_res(path, make_donut(self.tmp_donut, ignore_percent, fan_rad))
        elif path.startswith(b'vfx/omen/eff/org/'):
            fn = path[17:]
            if m := fan_regex.match(fn):
                deg = int(m.group(1))
                return self.add_res(path, make_fan(self.tmp_org_fan, math.radians(deg)))
            if m := donut_regex.match(fn):
                ignore_percent = int(m.group(1)) / 0xffff
                fan_rad = math.radians(int(m.group(2))) if m.group(2) else None
                return self.add_res(path, make_donut(self.tmp_org_donut, ignore_percent, fan_rad))
        elif path.startswith(b'vfx/omen/eff/pmk/'):
            return self.add_res(path, (pre_make_dir / path[17:].decode()).read_bytes())


AVFX_WANT_KILL = 0
AVFX_CONTINUE = 1
AVFX_WANT_UPDATE = 2


class AvfxOmen:
    omen_key: bytes | int = None
    scale: glm.vec3 = None
    facing: float = None
    trans: glm.vec3 = None
    color: glm.vec4 = None
    set_to: 'AvfxOmen' = None
    state = 0
    last_update = 0
    handle = 0

    def __del__(self):
        self.destroy()

    def set(self, omen_key, trans: bytes, scale: bytes, facing: float, color: bytes):
        real = AvfxOmen()
        real.omen_key = omen_key
        real.trans = glm.vec3.from_bytes(trans)
        real.scale = glm.vec3.from_bytes(scale)
        real.facing = facing
        real.color = glm.vec4.from_bytes(color)
        self.set_to = real
        self.state = AVFX_WANT_UPDATE
        self.last_update = time.time()

    def update(self):
        if self.last_update + 3 < time.time():
            self.state = AVFX_WANT_KILL
        if self.state == AVFX_WANT_UPDATE:
            if self.set_to is None:
                self.state = AVFX_WANT_KILL
            else:
                if self.omen_key != self.set_to.omen_key:
                    self.scale = self.set_to.scale
                    self.facing = self.set_to.facing
                    self.trans = self.set_to.trans
                    self.color = self.set_to.color
                    self.omen_key = self.set_to.omen_key
                    self.create()
                else:
                    any_mat_change = False
                    if self.scale != self.set_to.scale:
                        self.scale = self.set_to.scale
                        any_mat_change = True
                    if self.facing != self.set_to.facing:
                        self.facing = self.set_to.facing
                        any_mat_change = True
                    if self.trans != self.set_to.trans:
                        self.trans = self.set_to.trans
                        any_mat_change = True
                    if any_mat_change:
                        self.update_pos()
                    if self.color != self.set_to.color:
                        self.color = self.set_to.color
                        self.update_color()
                self.state = AVFX_CONTINUE
                self.set_to = None
        if self.state == AVFX_WANT_KILL:
            self.destroy()

    def create(self):
        self.destroy()
        if not self.omen_key: return
        if isinstance(self.omen_key, int):
            self.handle = create_omen(self.omen_key, glm.value_ptr(self.trans), 0, 1, self.facing, *self.scale, 1, 0)
        else:
            AvfxManager.instance.check_is_custom(self.omen_key)
            param = (c_char * 0x1a0)()
            init_vfx_param(param)
            self.handle = create_vfx(self.omen_key, param, 2, 0, *self.trans, *self.scale, self.facing, 1, -1)
            if self.handle:
                set_vfx_p1(self.handle, 1)
                set_vfx_p2(self.handle, 1)
        self.update_color()

    def update_pos(self):
        if self.handle:
            mat = (glm.translate(self.trans) *
                   glm.rotate(self.facing, glm.vec3(0, 1, 0)) *
                   glm.scale(self.scale))
            set_omen_matrix(self.handle, glm.value_ptr(mat))

    def update_color(self):
        if self.handle and self.color:
            set_omen_color(self.handle, *self.color)

    def destroy(self):
        if self.handle:
            try:
                remove_omen(self.handle, 1)
            except OSError:
                logger.error(f"omen destroy error: {self.handle:x}", exc_info=True)
            self.handle = 0

    def copy(self):
        new_omen = AvfxOmen()
        new_omen.omen_key = self.omen_key
        new_omen.scale = self.scale
        new_omen.facing = self.facing
        new_omen.trans = self.trans
        new_omen.color = self.color
        new_omen.state = AVFX_WANT_UPDATE
        return new_omen


class AvfxServer:
    pass


# endregion

def handle_exception(e):
    logger.error(f"draw server error: {e}", exc_info=e)


class DrawConfig(Structure):
    _fields_ = [
        ('version', c_uint32),
        ('flag', c_uint32),
    ]


class DrawServer:
    is_init = False
    server_3d: Server = None
    server_2d: Server = None

    def __init__(self):
        self.begin_hook = create_hook(
            sp("e8 * * * * 48 ? ? ? ? ff 15 ? ? ? ? e8"),
            address_type, [address_type]
        )(self.on_begin).install_and_enable()
        self.end_hook = create_hook(
            sa("48 89 5c 24 ? 57 48 ? ? ? 48 ? ? e8 ? ? ? ? 48 ? ? e8 ? ? ? ? 48 ? ? ? 48 ? ? ? e8"),
            address_type, [address_type]
        )(self.on_end).install_and_enable()

        self.put_buffer = None
        self.avfx_buffer = None
        self.compiled_commands = []
        self.last_update = 0
        self.config = DrawConfig()

        self.avfx_counter = 0
        self.avfx_omens = {}
        self.avfx_mgr = AvfxManager()

        # self.avfx_omens[self.create_omen(
        #     # b'i1',
        #     # b'vfx/omen/eff/ffd/circle.avfx',
        #     # b'vfx/omen/eff/ffd/fan_270.avfx',
        #     # b'vfx/omen/eff/ffd/donut_%d.avfx' % int(.75 * 0xffff),
        #     glm.vec3(-10, 0, -10).to_bytes(), glm.vec3(5).to_bytes(), 0, glm.vec4(.9, .1, .7, 1).to_bytes()
        # )].last_update = 1e+99

    def __del__(self):
        self.unload()

    def unload(self):
        for omen in self.avfx_omens.values():
            omen.destroy()
        self.avfx_mgr.unload()
        self.begin_hook.uninstall()
        self.end_hook.uninstall()
        if self.server_3d:
            self.server_3d.close()
            delattr(self, "server_3d")
        if self.server_2d:
            self.server_2d.close()
            delattr(self, "server_2d")

    def init(self):
        # should run in main thread
        if self.is_init: return
        try:
            self.server_3d = Server(True)
            self.server_2d = Server(False)
            self.is_init = True
            logger.info('server init')
        except Exception as e:
            handle_exception(e)
            self.unload()

    def on_begin(self, hook, *args):
        try:
            if self.server_3d:
                self.server_3d.begin_frame()
            if self.server_2d:
                self.server_2d.begin_frame()
        except Exception as e:
            handle_exception(e)
        return hook.original(*args)

    def on_end(self, hook, *args):
        try:
            self.flush()
        except Exception as e:
            handle_exception(e)
        try:
            if self.server_3d:
                self.server_3d.end_frame()
                self.server_3d.render_frame()
            if self.server_2d:
                self.server_2d.end_frame()
                self.server_2d.render_frame()
            self.init()
        except Exception as e:
            handle_exception(e)
        return hook.original(*args)

    # impl for rpc

    def create_omen(self, omen_key: bytes, trans: bytes, scale: bytes, facing: float, color: bytes):
        holder = AvfxOmen()
        if omen_key.startswith(b'i'): omen_key = int(omen_key[1:])
        holder.set(omen_key, trans, scale, facing, color)
        self.avfx_counter += 1
        k = self.avfx_counter
        self.avfx_omens[k] = holder
        return k

    def swap(self, data: bytes, avfx_data: bytes = None):
        self.put_buffer = data
        self.avfx_buffer = avfx_data

    def flush(self):
        if self.server_2d:
            self.server_2d.draw_triangle_strip([
                glm.vec3(0, 0, .5),
                glm.vec3(50, 0, .5),
                glm.vec3(0, 50, .5),
            ], glm.u8vec4(255, 0, 0, 150))
        if not self.server_3d: return
        if self.put_buffer is not None:
            self.compiled_commands.clear()
            self.put_buffer, data = None, self.put_buffer

            if self.config.version == 2:
                donut_type = DonutV2
                circle_type = CircleV2
                rectf_type = RectFV2
                rectfb_type = RectFBV2
                fan_type = FanV2
            else:
                donut_type = Donut
                circle_type = Circle
                rectf_type = RectF
                rectfb_type = RectFB
                fan_type = Fan

            for shape, _transform, _fill_color, _line_color in struct.iter_unpack("<I64s4s4s", data):
                shape_type = shape >> 16
                match shape_type:
                    case 1:  # circle/donut
                        shape_value = shape & 0xFFFF
                        model = donut_type(shape_value / 0xffff) if shape_value else circle_type
                    case 2:  # plane
                        model = rectfb_type if shape & 0xFFFF else rectf_type
                    case 5:  # sector
                        model = fan_type(shape & 0xFFFF)
                    case 6:  # triangle
                        model = Triangle
                    case 8:  # line
                        model = Line
                    case 9:  # point
                        # model = Point
                        return # not implemented
                    case 0x101:
                        model = Arrow
                    case _:
                        continue  # unknown shape
                for cmd, vtx_cnt, vtx_data in model.compile(
                        glm.mat4.from_bytes(_transform),
                        glm.u8vec4.from_bytes(_fill_color) if _fill_color[3] != 0 else None,
                        glm.u8vec4.from_bytes(_line_color) if _line_color[3] != 0 else None,
                        self.config.flag,
                ):
                    self.compiled_commands.append((cmd, vtx_cnt, vtx_data))
            self.last_update = time.time()
        elif self.compiled_commands and time.time() - self.last_update > 5:
            self.compiled_commands.clear()

        for cmd, vtx_cnt, vtx_data in self.compiled_commands:
            memmove(self.server_3d.create_command(cmd, vtx_cnt), vtx_data, len(vtx_data))

        omen: AvfxOmen
        if self.avfx_buffer is not None:
            off = 0
            max_off = len(self.avfx_buffer)
            current = time.time()
            while off < max_off:
                omen_id, command_id = struct.unpack_from("<II", self.avfx_buffer, off)
                off += 8
                if command_id == AVFX_CONTINUE:
                    if not (omen := self.avfx_omens.get(omen_id)): continue
                    omen.last_update = current
                elif command_id == AVFX_WANT_UPDATE:
                    omen_key, trans, scale, facing, color = struct.unpack_from("<64s12s12sf16s", self.avfx_buffer, off)
                    off += 108
                    if not (omen := self.avfx_omens.get(omen_id)): continue
                    omen_key = omen_key[:omen_key.find(b'\0')]
                    if omen_key.startswith(b'i'):
                        omen_key = int(omen_key[1:])
                    omen.set(omen_key, trans, scale, facing, color)
                elif command_id == AVFX_WANT_KILL:
                    if not (omen := self.avfx_omens.pop(omen_id, None)): continue
                    omen.state = AVFX_WANT_KILL
        for k in list(self.avfx_omens.keys()):
            omen = self.avfx_omens[k]
            omen.update()
            if omen.state == AVFX_WANT_KILL:
                self.avfx_omens.pop(k, None)
                omen.destroy()
