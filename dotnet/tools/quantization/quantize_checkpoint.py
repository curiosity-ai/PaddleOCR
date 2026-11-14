"""Write a PaddleOCR-VL checkpoint whose weights are stored as int8 with group scales.

Usage: quantize_checkpoint.py <out_dir> [--components dec|all] [--source <checkpoint_dir>]

  dec   the decoder's seven projections per layer and lm_head (the bytes a decode step streams)
  all   those, plus the vision tower's projections and the projector

For every quantised tensor `name` (bfloat16, [rows, cols]) the file holds `name` as I8 [rows, cols]
and `name.scale` as F32 [rows, cols / 128]: symmetric, scale = max|w| / 127 over each group of 128
consecutive columns, values rounded to nearest even and clamped to [-127, 127]. Everything else —
norms, biases, embeddings, position embeddings — is copied unchanged. This is the same arithmetic as
the library's on-the-fly path (`--quantization int8`), in float32 throughout, so the two agree
byte for byte. The config, tokenizer and other small files are copied beside it.

The result is a plain safetensors file; the library reads an int8 tensor as stored, whatever
`--quantization` says.
"""
import argparse, json, os, re, shutil, struct, sys
import numpy as np

GROUP = 128
DEC = re.compile(r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$")
VIS = re.compile(r"^visual\.vision_model\.encoder\.layers\.\d+\.(self_attn\.(q|k|v|out)_proj|mlp\.fc[12])\.weight$|^mlp_AR\.linear_[12]\.weight$")

def wanted(name, comps):
    return bool(DEC.match(name)) or name == "lm_head.weight" or (comps == "all" and bool(VIS.match(name)))

def quantize(w):
    rows, cols = w.shape
    group = GROUP if cols % GROUP == 0 else cols
    g = w.reshape(rows, cols // group, group)
    scale = np.abs(g).max(axis=2, keepdims=True) / np.float32(127)
    scale[scale == 0] = np.float32(1)
    q = np.clip(np.rint(g / scale), -127, 127).astype(np.int8)
    return q.reshape(rows, cols), scale.reshape(rows, cols // group).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--components", choices=("dec", "all"), default="dec")
    ap.add_argument("--source", default=os.path.expanduser("~/.cache/paddleocr-sharp/PaddleOCR-VL-1.6"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for name in os.listdir(a.source):
        if name != "model.safetensors":
            shutil.copy2(os.path.join(a.source, name), os.path.join(a.out_dir, name))

    path = os.path.join(a.source, "model.safetensors")
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    base = 8 + n
    data = np.memmap(path, dtype=np.uint8, mode="r")
    meta = header.pop("__metadata__", {}) or {}
    items = sorted(header.items(), key=lambda kv: kv[1]["data_offsets"][0])

    plan, new_header, offset = [], {}, 0
    for name, v in items:
        lo, hi = v["data_offsets"]
        if wanted(name, a.components) and v["dtype"] == "BF16" and len(v["shape"]) == 2:
            rows, cols = v["shape"]; group = GROUP if cols % GROUP == 0 else cols
            new_header[name] = {"dtype": "I8", "shape": [rows, cols], "data_offsets": [offset, offset + rows * cols]}
            offset += rows * cols
            sc = rows * (cols // group) * 4
            new_header[name + ".scale"] = {"dtype": "F32", "shape": [rows, cols // group], "data_offsets": [offset, offset + sc]}
            offset += sc
            plan.append((name, lo, hi, True))
        else:
            new_header[name] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [offset, offset + hi - lo]}
            offset += hi - lo
            plan.append((name, lo, hi, False))
    meta = dict(meta); meta["quantization"] = f"int8-g{GROUP}-{a.components}"
    new_header["__metadata__"] = meta
    blob = json.dumps(new_header, separators=(",", ":")).encode()
    blob += b" " * ((-len(blob)) % 8)

    out_path = os.path.join(a.out_dir, "model.safetensors")
    with open(out_path, "wb") as out:
        out.write(struct.pack("<Q", len(blob))); out.write(blob)
        for name, lo, hi, pick in plan:
            raw = data[base + lo: base + hi]
            if not pick:
                out.write(raw.tobytes()); continue
            w = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(new_header[name]["shape"])
            q, s = quantize(w)
            out.write(q.tobytes()); out.write(s.tobytes())
    print(f"wrote {out_path}: {os.path.getsize(out_path) / 2**20:.0f} MiB (source {os.path.getsize(path) / 2**20:.0f} MiB), components={a.components}")

main()
