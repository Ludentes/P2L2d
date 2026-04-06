"""Extract per-drawable UV bounding boxes from a Live2D moc3 file.

Usage:
    python -m pipeline.extract_uvs <moc3_path> [<texture_size>]

Output: JSON to stdout with list of drawables, each having:
    id, tex, order, x, y, w, h (pixel coords), verts (vertex count)
"""
from __future__ import annotations

import ctypes
import json
import sys
from pathlib import Path


_LIB_PATH = Path(__file__).parent.parent / ".venv/lib/python3.12/site-packages/live2d/v3/live2d.so"

def extract(moc3_path: Path, texture_size: int = 2048) -> list[dict]:
    import mmap

    lib = ctypes.CDLL(str(_LIB_PATH))

    # Load moc3 into mmap — gives page-aligned (≥4096-byte) buffer, satisfying Cubism's 64-byte requirement
    raw = moc3_path.read_bytes()
    size = len(raw)
    alloc_size = (size + 4095) & ~4095
    mm = mmap.mmap(-1, alloc_size)
    mm.write(raw)
    mm.seek(0)
    aligned = (ctypes.c_char * size).from_buffer(mm)

    lib.csmReviveMocInPlace.restype = ctypes.c_void_p
    moc = lib.csmReviveMocInPlace(aligned, ctypes.c_uint(size))
    if not moc:
        raise RuntimeError("csmReviveMocInPlace returned NULL")

    lib.csmGetSizeofModel.restype = ctypes.c_uint
    model_size = lib.csmGetSizeofModel(ctypes.c_void_p(moc))
    model_buf = (ctypes.c_uint8 * model_size)()
    lib.csmInitializeModelInPlace.restype = ctypes.c_void_p
    model = lib.csmInitializeModelInPlace(ctypes.c_void_p(moc), model_buf, ctypes.c_uint(model_size))
    if not model:
        raise RuntimeError("csmInitializeModelInPlace returned NULL")

    m = ctypes.c_void_p(model)
    lib.csmGetDrawableCount.restype = ctypes.c_int
    count = lib.csmGetDrawableCount(m)

    # IDs
    lib.csmGetDrawableIds.restype = ctypes.POINTER(ctypes.c_char_p)
    ids_ptr = lib.csmGetDrawableIds(m)
    ids = [ids_ptr[i].decode() for i in range(count)]

    # Texture indices
    lib.csmGetDrawableTextureIndices.restype = ctypes.POINTER(ctypes.c_int)
    tex_ptr = lib.csmGetDrawableTextureIndices(m)
    tex_indices = [tex_ptr[i] for i in range(count)]

    # Draw orders
    lib.csmGetDrawableDrawOrders.restype = ctypes.POINTER(ctypes.c_int)
    order_ptr = lib.csmGetDrawableDrawOrders(m)
    orders = [order_ptr[i] for i in range(count)]

    # Vertex counts
    lib.csmGetDrawableVertexCounts.restype = ctypes.POINTER(ctypes.c_int)
    vc_ptr = lib.csmGetDrawableVertexCounts(m)
    vcounts = [vc_ptr[i] for i in range(count)]

    # UV pointers (array of pointers, one per drawable)
    class _Vec2(ctypes.Structure):
        _fields_ = [("x", ctypes.c_float), ("y", ctypes.c_float)]

    lib.csmGetDrawableVertexUvs.restype = ctypes.POINTER(ctypes.POINTER(_Vec2))
    uv_ptrs = lib.csmGetDrawableVertexUvs(m)

    results = []
    for i in range(count):
        n = vcounts[i]
        if n == 0:
            continue
        uvs = uv_ptrs[i]
        us = [uvs[j].x for j in range(n)]
        vs = [uvs[j].y for j in range(n)]
        min_u, max_u = min(us), max(us)
        min_v, max_v = min(vs), max(vs)
        x = round(min_u * texture_size)
        y = round(min_v * texture_size)
        w = max(1, round((max_u - min_u) * texture_size))
        h = max(1, round((max_v - min_v) * texture_size))
        results.append({
            "id": ids[i],
            "tex": tex_indices[i],
            "order": orders[i],
            "x": x, "y": y, "w": w, "h": h,
            "verts": n,
        })

    return results


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.extract_uvs <moc3_path> [texture_size]", file=sys.stderr)
        sys.exit(1)
    moc3 = Path(sys.argv[1])
    tex_size = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
    drawables = extract(moc3, tex_size)
    print(json.dumps(drawables, indent=2))


if __name__ == "__main__":
    main()
