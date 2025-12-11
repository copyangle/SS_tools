import os
import hashlib
import struct
from typing import Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

WATERMARK_SKIP_W_RATIO = 0.40
WATERMARK_SKIP_H_RATIO = 0.08
DUCK_CHANNELS = 3

try:
    import folder_paths  # type: ignore
except Exception:
    folder_paths = None

def _bytes_to_binary_image(data: bytes, width: int = 512) -> Image.Image:
    """
    将任意二进制数据转为一张无损 PNG：
    - 以 RGB 像素承载，每像素 3 字节
    - 自动补零到整行
    - 不涉及压缩解压原始数据，避免音视频信息丢失
    """
    # 计算所需高度
    pixels = (len(data) + 2) // 3
    height = int(np.ceil(pixels / width))
    total_bytes = width * height * 3
    padded = data + b"\x00" * (total_bytes - len(data))
    arr = np.frombuffer(padded, dtype=np.uint8).reshape((height, width, 3))
    img = Image.fromarray(arr, mode="RGB")
    return img


def _generate_key_stream(password: str, salt: bytes, length: int) -> bytes:
    key_material = (password + salt.hex()).encode("utf-8")
    out = bytearray()
    counter = 0
    while len(out) < length:
        combined = key_material + str(counter).encode("utf-8")
        out.extend(hashlib.sha256(combined).digest())
        counter += 1
    return bytes(out[:length])

def _encrypt_with_password(data: bytes, password: str):
    if not password:
        return data, b"", b"", False
    salt = os.urandom(16)
    key_stream = _generate_key_stream(password, salt, len(data))
    cipher = bytes(a ^ b for a, b in zip(data, key_stream))
    pwd_hash = hashlib.sha256((password + salt.hex()).encode("utf-8")).digest()
    return cipher, salt, pwd_hash, True

def _build_file_header(raw: bytes, password: str, ext: str = "png") -> bytes:
    cipher, salt, pwd_hash, has_pwd = _encrypt_with_password(raw, password)
    payload = cipher if has_pwd else raw
    ext_bytes = ext.encode("utf-8")
    header = bytearray()
    header.append(1 if has_pwd else 0)
    if has_pwd:
        header.extend(pwd_hash)
        header.extend(salt)
    header.append(len(ext_bytes))
    header.extend(ext_bytes)
    header.extend(struct.pack(">I", len(payload)))
    header.extend(payload)
    return bytes(header)

def _build_duck_image(size: int = 640, title: str = "") -> Image.Image:
    bg = Image.new("RGBA", (size, size), (153, 204, 255, 255))
    draw = ImageDraw.Draw(bg)
    body_color = (255, 223, 94)
    beak_color = (255, 153, 51)
    eye_color = (0, 0, 0)
    wing_color = (255, 200, 70)
    draw.ellipse([size * 0.2, size * 0.35, size * 0.8, size * 0.85], fill=body_color + (255,), outline=(255, 190, 60), width=4)
    draw.ellipse([size * 0.35, size * 0.15, size * 0.65, size * 0.45], fill=body_color + (255,), outline=(255, 190, 60), width=4)
    draw.ellipse([size * 0.4, size * 0.55, size * 0.75, size * 0.75], fill=wing_color + (255,), outline=(255, 190, 60), width=3)
    draw.polygon([(size * 0.65, size * 0.32),(size * 0.78, size * 0.36),(size * 0.68, size * 0.40),(size * 0.60, size * 0.38)], fill=beak_color + (255,), outline=(200, 120, 30))
    draw.ellipse([size * 0.56, size * 0.24, size * 0.60, size * 0.28], fill=eye_color + (255,))
    draw.ellipse([size * 0.47, size * 0.24, size * 0.51, size * 0.28], fill=eye_color + (255,))
    draw.arc([size * 0.1, size * 0.75, size * 0.9, size * 0.9], start=10, end=170, fill=(255, 255, 255, 255), width=3)
    draw.arc([size * 0.15, size * 0.78, size * 0.85, size * 0.93], start=10, end=170, fill=(240, 240, 240, 255), width=2)
    fs_title = max(12, int(size * 0.06))
    fs_ver_base = max(10, int(size * 0.045))
    fs_ver = max(8, int(round(fs_ver_base * 0.5)))
    base_font = ImageFont.load_default()
    def make_scaled_text(text: str, target_h: int, color: tuple) -> Image.Image:
        tmp = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        tmp_draw = ImageDraw.Draw(tmp)
        bbox = tmp_draw.textbbox((0, 0), text, font=base_font)
        w0 = max(1, bbox[2] - bbox[0])
        h0 = max(1, bbox[3] - bbox[1])
        pad = 2
        img = Image.new("RGBA", (w0 + pad * 2, h0 + pad * 2), (0, 0, 0, 0))
        ImageDraw.Draw(img).text((pad, pad), text, fill=color, font=base_font)
        s = max(1e-6, float(target_h) / float(h0))
        tw = max(1, int(round((w0 + pad * 2) * s)))
        th = max(1, int(round((h0 + pad * 2) * s)))
        return img.resize((tw, th), Image.BICUBIC)
    if title:
        title_img = make_scaled_text(title[:30], fs_title, (0, 0, 0, 255))
        margin = int(size * 0.06)
        tx = margin
        ty = max(int(size * 0.10), margin)
        if tx + title_img.width > size - margin:
            tx = max(margin, size - margin - title_img.width)
        if ty + title_img.height > int(size * 0.35):
            ty = max(margin, int(size * 0.35) - title_img.height)
        bg.paste(title_img, (tx, ty), title_img)
    ver_text = "V1.0"
    ver_img = make_scaled_text(ver_text, fs_ver, (255, 255, 255, 255))
    bottom_margin = int(size * 0.06)
    vx = max(int((size - ver_img.width) // 2), 0)
    vy = size - ver_img.height - bottom_margin
    vy = max(vy, int(size * 0.80))
    vy = min(vy + ver_img.height, size - ver_img.height - int(size * 0.02))
    if vx + ver_img.width > size:
        vx = max(0, size - ver_img.width)
    bg.paste(ver_img, (vx, vy), ver_img)
    return bg

def _required_canvas_size(bit_len: int, lsb_bits: int) -> int:
    side = 640
    while True:
        skip_w = int(side * WATERMARK_SKIP_W_RATIO)
        skip_h = int(side * WATERMARK_SKIP_H_RATIO)
        excluded = skip_w * skip_h
        usable_bits = (side * side - excluded) * DUCK_CHANNELS * lsb_bits
        if usable_bits >= bit_len:
            return side
        side += 64

def _embed_payload_lsb(img: Image.Image, file_header: bytes, lsb_bits: int) -> Image.Image:
    img = img.convert("RGB")
    arr = np.array(img).astype(np.uint8)
    h, w, c = arr.shape
    skip_w = int(w * WATERMARK_SKIP_W_RATIO)
    skip_h = int(h * WATERMARK_SKIP_H_RATIO)
    mask2d = np.ones((h, w), dtype=bool)
    if skip_w > 0 and skip_h > 0:
        mask2d[:skip_h, :skip_w] = False
    mask3d = np.repeat(mask2d[:, :, None], c, axis=2)
    length_prefix = struct.pack(">I", len(file_header))
    payload_with_len = length_prefix + file_header
    bits = np.unpackbits(np.frombuffer(payload_with_len, dtype=np.uint8), bitorder="big")
    bit_len = len(bits)
    capacity_bits = int(mask3d.sum()) * lsb_bits
    if bit_len > capacity_bits:
        raise ValueError("Data too large, capacity exceeded. 数据过大，鸭子图容量不够。请使用更小的文件。")
    flat = arr.reshape(-1)
    mask = (1 << lsb_bits) - 1
    groups = bit_len // lsb_bits + (1 if bit_len % lsb_bits else 0)
    pad = groups * lsb_bits - bit_len
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    bg = bits.reshape(groups, lsb_bits)
    weights = (1 << np.arange(lsb_bits - 1, -1, -1, dtype=np.uint8))
    vals = (bg * weights).sum(axis=1).astype(np.uint8)
    idxs = np.flatnonzero(mask3d.reshape(-1))
    if groups > len(idxs):
        raise ValueError("Data too large, capacity exceeded. 数据过大，鸭子图容量不够。请使用更小的文件。")
    sel = idxs[:groups]
    flat[sel] = (flat[sel] & (~mask)) | vals
    arr = flat.reshape(arr.shape)
    if skip_w > 0 and skip_h > 0:
        src_w = max(0, arr.shape[1] - skip_w)
        if src_w > 0:
            src_block = arr[:skip_h, skip_w:skip_w + min(skip_w, src_w), :]
            if src_block.shape[1] == skip_w:
                dest = src_block
            else:
                reps = int(np.ceil(skip_w / max(1, src_block.shape[1])))
                dest = np.tile(src_block, (1, reps, 1))[:, :skip_w, :]
            arr[:skip_h, :skip_w, :] = dest
    return Image.fromarray(arr, mode="RGB")

def export_duck_payload(
    raw_bytes: bytes,
    password: str,
    ext: str,
    compress: int,
    title: str,
    output_dir: Optional[str] = None,
    output_name: str = "duck_payload.png",
) -> Tuple[str, Image.Image]:
    file_header = _build_file_header(raw_bytes, password, ext=ext)
    lsb_bits = 8 if compress >= 8 else (6 if compress >= 6 else 2)
    required_size = _required_canvas_size((len(file_header) + 4) * 8, lsb_bits)
    duck_img = _build_duck_image(size=required_size, title=title)
    duck_img = _embed_payload_lsb(duck_img, file_header, lsb_bits)
    base_dir = output_dir or (folder_paths.get_output_directory() if folder_paths else os.getcwd())
    os.makedirs(base_dir, exist_ok=True)
    out_path = os.path.join(base_dir, output_name)
    duck_img.save(out_path, format="PNG", optimize=True, compress_level=9)
    return out_path, duck_img
