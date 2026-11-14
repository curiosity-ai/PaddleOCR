"""Build a checkpoint whose targeted weights hold quantize->dequantize values (exact emulation).

Usage: make_variant.py <scheme>:<components> <out_dir>
  scheme      f32 (control: bf16->float32, no quantization), int8row, int8g<N>, int4g<N>
              (symmetric, round to nearest; per output row, or per group of N along the reduction)
  components  dec (decoder layers + lm_head), vis (vision tower layers + projector), all, head, layers

The targeted tensors are written as float32 holding the dequantized values; everything else is
copied as-is. The pipeline then runs unchanged on the result, so what it measures is the accuracy
of the scheme and nothing about the speed of any kernel.
"""
import json, os, re, struct, sys
import numpy as np

SRC = os.path.expanduser("~/.cache/paddleocr-sharp/PaddleOCR-VL-1.6")
DEC_LAYERS = re.compile(r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$")
VIS_LAYERS = re.compile(r"^visual\.vision_model\.encoder\.layers\.\d+\.(self_attn\.(q|k|v|out)_proj|mlp\.fc[12])\.weight$")
PROJ = re.compile(r"^mlp_AR\.linear_[12]\.weight$")

def selected(name, comps):
    dec = bool(DEC_LAYERS.match(name)); head = name == "lm_head.weight"
    vis = bool(VIS_LAYERS.match(name)); proj = bool(PROJ.match(name))
    return ((comps in ("dec", "all")) and (dec or head)) or ((comps in ("vis", "all")) and (vis or proj)) \
        or (comps == "head" and head) or (comps == "layers" and dec)

def bf16_to_f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)

def quantize(w, scheme):
    if scheme == "f32":
        return w
    rows, cols = w.shape
    if scheme == "int8row":
        group, qmax = cols, 127
    else:
        m = re.fullmatch(r"int(8|4)g(\d+)", scheme)
        bits, group = int(m.group(1)), int(m.group(2)); qmax = 127 if bits == 8 else 7
        assert cols % group == 0, (cols, group)
    g = w.reshape(rows, cols // group, group)
    scale = np.abs(g).max(axis=2, keepdims=True) / qmax
    scale[scale == 0] = 1.0
    q = np.clip(np.rint(g / scale), -qmax, qmax)
    return (q * scale).reshape(rows, cols).astype(np.float32)

def main():
    spec, out_dir = sys.argv[1], sys.argv[2]
    scheme, comps = spec.split(":")
    os.makedirs(out_dir, exist_ok=True)
    for name in os.listdir(SRC):
        if name != "model.safetensors":
            dst = os.path.join(out_dir, name)
            if not os.path.exists(dst):
                os.symlink(os.path.join(SRC, name), dst)

    path = os.path.join(SRC, "model.safetensors")
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    base = 8 + n
    data = np.memmap(path, dtype=np.uint8, mode="r")

    meta = header.pop("__metadata__", None)
    items = sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])
    new_header, offset, chosen = {}, 0, 0
    plan = []
    for name, v in items:
        a, b = v["data_offsets"]
        pick = selected(name, comps) and v["dtype"] == "BF16" and len(v["shape"]) == 2
        size = (b - a) * (2 if pick else 1)
        new_header[name] = {"dtype": "F32" if pick else v["dtype"], "shape": v["shape"], "data_offsets": [offset, offset + size]}
        plan.append((name, a, b, pick))
        offset += size
        chosen += pick
    if meta is not None:
        new_header["__metadata__"] = meta
    blob = json.dumps(new_header, separators=(",", ":")).encode()
    blob += b" " * ((-len(blob)) % 8)

    out_path = os.path.join(out_dir, "model.safetensors")
    err_num = err_den = 0.0
    with open(out_path, "wb") as out:
        out.write(struct.pack("<Q", len(blob))); out.write(blob)
        for name, a, b, pick in plan:
            raw = data[base + a: base + b]
            if not pick:
                out.write(raw.tobytes()); continue
            shape = new_header[name]["shape"]
            w = bf16_to_f32(np.frombuffer(raw, dtype=np.uint16)).reshape(shape)
            d = quantize(w, scheme)
            err_num += float(((d - w) ** 2).sum()); err_den += float((w ** 2).sum())
            out.write(np.ascontiguousarray(d, dtype=np.float32).tobytes())
    print(f"{spec}: {chosen} tensors, weight RMS error vs bf16 {np.sqrt(err_num / max(err_den, 1e-30)):.3e} relative")

main()
