import time
import moderngl
import numba.typed.typeddict
import numpy as np
from PIL import Image
from numba import njit
import ffmpegcv

def load_pcf_v1():
    data = open('Uni2-Fixed16.psf', 'rb').read()
    assert data.startswith(b'\x36\x04')
    assert 0 <= data[2] < 4
    n, has_unicode = 256 * (1 + (data[2] & 1)), bool(data[2] & 2)
    font_height = int(data[3])
    data = data[4:]
    # print(f'{n=} {has_unicode=} {font_height=}')
    # print(f'{len(data)-font_height*n=}')
    chars = np.frombuffer(data[: font_height * n], dtype='u1').reshape(n, font_height)
    data = data[font_height * n:]
    points = np.frombuffer(data, dtype='<u2')
    char_id = np.cumsum(points == 2**16-1)
    assert char_id[-1] == n
    char_table = dict(zip(map(chr, points), char_id))
    del char_table[chr(2**16-1)]
    return chars, char_table


def main(chars, char_table):
    ctx = moderngl.create_context(standalone=True)
    n = 800
    fbo = ctx.framebuffer([ctx.renderbuffer((n, n), components=4, samples=16)])
    output = ctx.framebuffer([ctx.renderbuffer((n, n), components=4)])

    program = ctx.program(
        vertex_shader=open('shaders/text.vsh', 'rt').read(),
        geometry_shader=open('shaders/text_geometry.glsl', 'rt').read(),
        fragment_shader=open('shaders/text.fsh', 'rt').read(),
    )

    # print(program._attribute_locations)
    # print(program._attribute_types)
    # print(program._members)

    vbo = np.mgrid[-0.9:0.9:16j, -0.9:0.9:32j].reshape(2, -1).T
    vbo = np.flip(vbo, 1)
    assert vbo.shape == (32 * 16, 2)
    assert -1.0 <= vbo.min() and vbo.max() <= 1.0
    vbo_c = np.zeros((vbo.shape[0], 3))
    rng = np.random.RandomState(42)
    vbo_c[
        np.arange(vbo.shape[0]),
        rng.randint(0, 3, size=vbo.shape[0])
    ] = 1.0
    vbo = ctx.buffer(vbo.astype('float32').tobytes())
    vbo_c = ctx.buffer(vbo_c.astype('float32').tobytes())
    vbo_i = ctx.buffer(np.arange(32*16).astype('uint32').tobytes())

    vao = ctx.vertex_array(program, [
        (vbo, '2f', 'in_position'),
        (vbo_c, '3f', 'in_color'),
        (vbo_i, '1u', 'in_idx'),
    ])

    fbo.use()
    ctx.enable(ctx.BLEND)
    ctx.clear()
    program['chars'].write(chars.astype('u1').tobytes())
    fbo.clear(1., 1., 1., 1.)
    vao.render(mode=ctx.POINTS)
    ctx.copy_framebuffer(output, fbo)
    img = Image.frombuffer('RGBA', output.size, output.read(components=4)) #.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    img.save('output.png')


@njit(cache=True)
def render_text(data, n, m, text, text_pos, char_table):
    data[:] = char_table[' ']
    screen_row, screen_col = 0, 0
    while text_pos < len(text) and screen_row < n:
        c = text[text_pos]
        text_pos += 1
        if c == '\n':
            screen_row, screen_col = screen_row + 1, 0
        else:
            data[screen_row, screen_col] = char_table[c]
            screen_col += 1
            if screen_col == m:
                screen_row, screen_col = screen_row + 1, 0
    return text_pos


def main_speed(text, chars, char_table):
    ctx = moderngl.create_context(standalone=True)
    m, n = 80, 24
    w, h = 9, 16
    fbo = ctx.framebuffer([ctx.renderbuffer((m*w, n*h), components=4)])
    output = ctx.framebuffer([ctx.renderbuffer((m*w, n*h), components=4)])
    program = ctx.program(
        vertex_shader=open('shaders/text.vsh', 'rt').read(),
        geometry_shader=open('shaders/text_geometry.glsl', 'rt').read(),
        fragment_shader=open('shaders/text.fsh', 'rt').read(),
    )
    print(chars.shape)
    program['chars'].write(chars.astype('u1').tobytes())

    x, y = np.arange(m), np.arange(n)
    xy = np.stack(np.meshgrid(x * w + (w / 2), y * h + (h / 2), indexing='xy')).reshape(2, -1).T
    # print(xy)
    xy = xy * np.array((2.0 / m / w, 2.0 / n / h)) - 1.0

    data = np.full((n, m), char_table[' '], dtype='uint32')
    vbo = ctx.buffer(data.astype('uint32').tobytes())

    colors = np.zeros((n, m, 3), dtype='float32')
    # colors[:, :, [1]] = 1.0 # green
    colors[:, :, [0, 1]] = 1.0 # yellow

    vbo_pos = ctx.buffer(xy.astype('float32').tobytes())
    vbo_scale = ctx.buffer(np.tile(
        np.array(((w - 1.0) / (m * w), float(h) / (n * h)), dtype='float32'), (n, m)
    ).astype('float32').tobytes())
    vbo_color = ctx.buffer(colors.tobytes())

    vao = ctx.vertex_array(program, [
        (vbo_pos, '2f', 'in_position'),
        (vbo_scale, '2f', 'in_scale'),
        (vbo_color, '3f', 'in_color'),
        (vbo, '1u', 'in_idx'),
    ])
    fbo.use()
    ctx.enable(ctx.BLEND)

    text_pos = np.int64(0)

    char_table_typed = numba.typed.typeddict.Dict.empty(numba.types.UnicodeCharSeq(1), numba.i4)
    for k, v in char_table.items():
        char_table_typed[k] = v
    render_text(data, n, m, text, text_pos, char_table_typed)

    # frames_bin = open('frames.bin', 'wb', buffering=2**20)
    frames = ffmpegcv.VideoWriter('frames.mp4', 'h264_nvenc', fps=60, pix_fmt='rgb24', bitrate=10**8)
    frame_buffer = bytearray(output.read(components=3))
    st_t = time.time()
    n_page = 0
    while text_pos < len(text):
        text_pos = render_text(data, n, m, text, text_pos, char_table_typed)
        n_page += 1

        vbo.write(data.astype('uint32').tobytes())
        ctx.clear(0,0,0,1)
        vao.render(mode=ctx.POINTS)
        ctx.copy_framebuffer(output, fbo)
        output.read_into(frame_buffer, components=3)
        frames.write(np.frombuffer(frame_buffer, dtype='uint8').reshape((n*h, m*w, 3)))
        # frames_bin.write(frame_buffer)

    print(f'{time.time() - st_t:.3} took. {n_page} pages with {len(text)/n_page:.2f} average chars by page and {len(text)/n_page/n:.2f} by line')
    frames.close()
    # frames_bin.close()

    img = Image.frombuffer('RGBA', output.size, output.read(components=4))
    img.save('render.png')


if __name__ == '__main__':
    chars, char_table = load_pcf_v1()
    # main(chars, char_table)
    text = open('WarAndPeace.txt', 'rt', encoding='utf-8-sig').read()
    print(len(text))
    main_speed(text, chars, char_table)
