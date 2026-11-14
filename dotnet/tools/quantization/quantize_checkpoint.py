"""Write a PaddleOCR-VL checkpoint whose weights are stored as int8.

Usage: quantize_checkpoint.py <out_dir> [--components dec|all] [--group 128|row] [--source <checkpoint_dir>]

With no options this writes the checkpoint `--quantization int8-compute` runs from, the one published
at https://models.curiosity.ai/paddleocr-vl/int8-compute/model.safetensors: every projection of the
tower, the projector and the decoder, one scale per output row. The other combinations are what the
experiments in CLAUDE.md compared; the library only ships and downloads the default.

  dec   the decoder's seven projections per layer and lm_head (the bytes a decode step streams)
  all   those, plus the vision tower's projections and the projector

For every quantised tensor `name` (bfloat16, [rows, cols]) the file holds `name` as I8 [rows, cols]
and `name.scale` as F32 [rows, cols / 128]: symmetric, scale = max|w| / 127 over each group of 128
consecutive columns, values rounded to nearest even and clamped to [-127, 127]. Everything else —
norms, biases, embeddings, position embeddings — is copied unchanged. This is the same arithmetic as
the library's on-the-fly path (`--quantization int8-compute`), in float32 throughout, so the two agree
byte for byte. The config, tokenizer and other small files are copied beside it.

--group row stores one scale per output row instead of one per 128 columns. It is what an
integer-dot-product kernel wants — one int32 sum per output, scaled once at the end — and costs a
little weight accuracy against the grouped form.

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

def group_for(cols, mode):
    return cols if mode == "row" or cols % GROUP != 0 else GROUP

def quantize(w, mode):
    rows, cols = w.shape
    group = group_for(cols, mode)
    g = w.reshape(rows, cols // group, group)
    scale = np.abs(g).max(axis=2, keepdims=True) / np.float32(127)
    scale[scale == 0] = np.float32(1)
    q = np.clip(np.rint(g / scale), -127, 127).astype(np.int8)
    return q.reshape(rows, cols), scale.reshape(rows, cols // group).astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--components", choices=("dec", "all"), default="all")
    ap.add_argument("--group", choices=("128", "row"), default="row")
    ap.add_argument("--source", default=os.path.expanduser("~/.cache/paddleocr-sharp/PaddleOCR-VL-1.6"))
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    # A download cache can wrap the checkpoint in a folder (for instance `main/`). Use the folder that
    # actually holds model.safetensors: the source itself, or the one subfolder that does.
    source = a.source
    if not os.path.isfile(os.path.join(source, "model.safetensors")):
        nested = [d for d in sorted(os.listdir(source))
                  if os.path.isfile(os.path.join(source, d, "model.safetensors"))]
        if len(nested) != 1:
            sys.exit(f"no model.safetensors in {source}" + (f" or in exactly one subfolder (found {nested})" if nested else ""))
        source = os.path.join(source, nested[0])
        print(f"using the checkpoint in {source}")

    # Copy the small files (config, tokenizer, ...) beside the new weights. Files only: a cache may
    # keep folders in here, and they are not part of the checkpoint.
    for name in os.listdir(source):
        src = os.path.join(source, name)
        if name != "model.safetensors" and os.path.isfile(src):
            shutil.copy2(src, os.path.join(a.out_dir, name))

    path = os.path.join(source, "model.safetensors")
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
            rows, cols = v["shape"]; group = group_for(cols, a.group)
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
    meta = dict(meta); meta["quantization"] = f"int8-{'row' if a.group == 'row' else 'g' + str(GROUP)}-{a.components}"
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
            q, s = quantize(w, a.group)
            out.write(q.tobytes()); out.write(s.tobytes())
    print(f"wrote {out_path}: {os.path.getsize(out_path) / 2**20:.0f} MiB (source {os.path.getsize(path) / 2**20:.0f} MiB), components={a.components}")

main()
