#!/usr/bin/env python3
import hashlib
import os
import pathlib
import shutil
import subprocess
import sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    if len(sys.argv) != 4:
        raise SystemExit('usage: package_chunks.py INPUT OUTPUT_DIR CHUNK_MIB')
    src = pathlib.Path(sys.argv[1]).resolve()
    out = pathlib.Path(sys.argv[2]).resolve()
    chunk = int(sys.argv[3]) * 1024 * 1024
    if not src.is_file():
        raise SystemExit(f'missing input: {src}')
    out.mkdir(parents=True, exist_ok=True)
    digest = sha256(src)
    size = src.stat().st_size
    parts = []
    with src.open('rb') as fin:
        index = 0
        while True:
            data = fin.read(chunk)
            if not data:
                break
            part = out / f'{src.name}.part-{index:04d}'
            part.write_bytes(data)
            parts.append({'name': part.name, 'size': len(data), 'sha256': sha256(part)})
            index += 1
    manifest = out / 'manifest.txt'
    lines = [f'file={src.name}', f'size={size}', f'sha256={digest}', f'parts={len(parts)}']
    lines += [f"{p['name']}\t{p['size']}\t{p['sha256']}" for p in parts]
    manifest.write_text('\n'.join(lines) + '\n')
    print(manifest)


if __name__ == '__main__':
    main()
