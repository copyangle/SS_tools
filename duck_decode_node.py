import io
import os
import struct
import subprocess
import numpy as np
from typing import Any, List
from PIL import Image
import torch
from moviepy import VideoFileClip
try:
    import folder_paths  # type: ignore
except Exception:
    folder_paths = None

CATEGORY = "SSTool"
WATERMARK_SKIP_W_RATIO = 0.40
WATERMARK_SKIP_H_RATIO = 0.08

def _extract_payload_with_k(arr: np.ndarray, k: int) -> bytes:
    h, w, c = arr.shape
    skip_w = int(w * WATERMARK_SKIP_W_RATIO)
    skip_h = int(h * WATERMARK_SKIP_H_RATIO)
    mask2d = np.ones((h, w), dtype=bool)
    if skip_w > 0 and skip_h > 0:
        mask2d[:skip_h, :skip_w] = False
    mask3d = np.repeat(mask2d[:, :, None], c, axis=2)
    flat = arr.reshape(-1)
    idxs = np.flatnonzero(mask3d.reshape(-1))
    vals = (flat[idxs] & ((1 << k) - 1)).astype(np.uint8)
    ub = np.unpackbits(vals, bitorder="big").reshape(-1, 8)[:, -k:]
    bits = ub.reshape(-1)
    if len(bits) < 32:
        raise ValueError("Insufficient image data. 图像数据不足")
    len_bits = bits[:32]
    length_bytes = np.packbits(len_bits, bitorder="big").tobytes()
    header_len = struct.unpack(">I", length_bytes)[0]
    total_bits = 32 + header_len * 8
    if header_len <= 0 or total_bits > len(bits):
        raise ValueError("Payload length invalid. 载荷长度异常")
    payload_bits = bits[32:32 + header_len * 8]
    return np.packbits(payload_bits, bitorder="big").tobytes()

def _generate_key_stream(password: str, salt: bytes, length: int) -> bytes:
    import hashlib
    key_material = (password + salt.hex()).encode("utf-8")
    out = bytearray()
    counter = 0
    while len(out) < length:
        out.extend(hashlib.sha256(key_material + str(counter).encode("utf-8")).digest())
        counter += 1
    return bytes(out[:length])

def _parse_header(header: bytes, password: str):
    idx = 0
    if len(header) < 1:
        raise ValueError("Header corrupted. 文件头损坏")
    has_pwd = header[0] == 1
    idx += 1
    pwd_hash = b""
    salt = b""
    if has_pwd:
        if len(header) < idx + 32 + 16:
            raise ValueError("Header corrupted. 文件头损坏")
        pwd_hash = header[idx:idx + 32]; idx += 32
        salt = header[idx:idx + 16]; idx += 16
    if len(header) < idx + 1:
        raise ValueError("Header corrupted. 文件头损坏")
    ext_len = header[idx]; idx += 1
    if len(header) < idx + ext_len + 4:
        raise ValueError("Header corrupted. 文件头损坏")
    ext = header[idx:idx + ext_len].decode("utf-8", errors="ignore"); idx += ext_len
    data_len = struct.unpack(">I", header[idx:idx + 4])[0]; idx += 4
    data = header[idx:]
    if len(data) != data_len:
        raise ValueError("Data length mismatch. 数据长度不匹配")
    if not has_pwd:
        return data, ext
    if not password:
        raise ValueError("Password required. 需要密码")
    import hashlib
    check_hash = hashlib.sha256((password + salt.hex()).encode("utf-8")).digest()
    if check_hash != pwd_hash:
        raise ValueError("Wrong password. 密码错误")
    ks = _generate_key_stream(password, salt, len(data))
    plain = bytes(a ^ b for a, b in zip(data, ks))
    return plain, ext

def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    if image.dim() == 4:
        image = image[0]
    arrf = image.detach().cpu().numpy() * 255.0
    arru = np.rint(np.clip(arrf, 0, 255)).astype(np.uint8)
    if arru.ndim == 2:
        arru = np.stack([arru, arru, arru], axis=-1)
        return Image.fromarray(arru, mode="RGB")
    if arru.shape[-1] == 3:
        return Image.fromarray(arru, mode="RGB")
    if arru.shape[-1] == 4:
        return Image.fromarray(arru, mode="RGBA")
    if arru.shape[-1] > 4:
        return Image.fromarray(arru[..., :3], mode="RGB")
    return Image.fromarray(np.repeat(arru[..., :1], 3, axis=-1), mode="RGB")


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]

def binpng_bytes_to_mp4_bytes(p: str) -> bytes:
    img = Image.open(p).convert("RGB")
    arr = np.array(img).astype(np.uint8)
    flat = arr.reshape(-1, 3).reshape(-1)
    return flat.tobytes().rstrip(b"\x00")

class DuckDecodeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),

            },
            "optional": {
                "password": ("STRING", {"default": "", "multiline": False}),
                "Notes": ("STRING", {
                    "multiline": True,  # 核心：开启多行模式
                    "default": "使用方法：https://github.com/copyangle/SS_tools\n支持图片/视频隐写保护",
                    # 多行默认内容
                    "placeholder": "使用方法：https://github.com/copyangle/SS_tools",  # 输入提示（可选）
                    "dynamicPrompts": False,  # 关闭动态提示（按需开启）
                    "rows": 2,  # 可选：指定输入框默认行数（视觉效果）
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING", "INT")
    RETURN_NAMES = ("images", "audio", "file_path", "fps")
    FUNCTION = "decode"
    CATEGORY = CATEGORY

    def decode(self, image: torch.Tensor, password: str = "",Notes: str = ""):
        pil = _tensor_to_pil(image)
        arr = np.array(pil.convert("RGB")).astype(np.uint8)
        header = None
        raw = None
        ext = None
        last_err = None
        for k in (2, 6, 8):
            try:
                header = _extract_payload_with_k(arr, k)
                raw, ext = _parse_header(header, password)
                break
            except Exception as e:
                last_err = e
                continue
        if raw is None:
            raise last_err or RuntimeError("解析失败")

        base_dir = folder_paths.get_output_directory() if folder_paths else os.getcwd()
        os.makedirs(base_dir, exist_ok=True)
        name = "duck_recovered"
        out_path = os.path.join(base_dir, name)

        final_path = ""
        final_ext = ext
        if ext.endswith(".binpng"):
            tmp_png = out_path + ".binpng"
            with open(tmp_png, "wb") as f:
                f.write(raw)
            mp4_bytes = binpng_bytes_to_mp4_bytes(tmp_png)
            os.unlink(tmp_png)
            final_path = out_path + ".mp4"
            with open(final_path, "wb") as f:
                f.write(mp4_bytes)
            final_ext = "mp4"
        else:
            final_path = out_path + ("." + ext if not ext.startswith(".") else ext)
            with open(final_path, "wb") as f:
                f.write(raw)

        img_tensor = None
        audio_out = None
        fps_out = 0
        if final_ext.lower() == "png":
            img_tensor = _pil_to_tensor(Image.open(final_path).convert("RGB"))
        elif final_ext.lower() == "mp4":
            clip = VideoFileClip(final_path)
            fps_out = int(round(clip.fps)) if clip.fps else 0
            times: List[float] = []
            try:
                cmd = [
                    "ffprobe", "-v", "error",
                    "-select_streams", "v:0",
                    "-show_frames",
                    "-show_entries", "frame=pkt_pts_time,key_frame",
                    "-of", "csv=print_section=0",
                    final_path,
                ]
                res = subprocess.run(cmd, capture_output=True, text=True, check=False)
                for line in res.stdout.splitlines():
                    if "key_frame=1" in line:
                        # extract time after pkt_pts_time=
                        t = None
                        for part in line.split(","):
                            if "pkt_pts_time=" in part:
                                try:
                                    t = float(part.split("=")[1])
                                except Exception:
                                    t = None
                        if t is not None:
                            times.append(t)
            except Exception:
                pass
            if not times:
                times = [i / max(1, fps_out) for i in range(int(clip.duration * max(1, fps_out)))]
            frames: List[np.ndarray] = []
            for t in times:
                try:
                    frames.append(clip.get_frame(t))
                except Exception:
                    continue
            if len(frames) == 0:
                img_tensor = torch.zeros((1, 1, 1, 3), dtype=torch.float32)
            else:
                arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(arr)
            if clip.audio is not None:
                try:
                    sr_attr = getattr(clip.audio, "fps", None)
                    sr = int(round(sr_attr)) if sr_attr else 44100
                    audio_np = clip.audio.to_soundarray(fps=sr)
                    # 归一化处理
                    max_val = np.max(np.abs(audio_np))
                    if max_val > 0:
                        audio_np = audio_np / max_val

                    if len(audio_np.shape) == 1:
                        audio_np = np.expand_dims(audio_np, axis=1)
                    wf = torch.from_numpy(audio_np.T.astype(np.float32)).unsqueeze(0)
                    audio_out = {"waveform": wf, "sample_rate": sr}
                except Exception:
                    audio_out = None
            clip.close()
        else:
            img_tensor = torch.zeros((1, 1, 1, 3), dtype=torch.float32)

        return (img_tensor, audio_out, final_path, fps_out)


NODE_CLASS_MAPPINGS = {"DuckDecodeNode": DuckDecodeNode}
NODE_DISPLAY_NAME_MAPPINGS = {"DuckDecodeNode": "SSuper-SecureMediaProtection dec媒体内容保护解码V1.0"}
