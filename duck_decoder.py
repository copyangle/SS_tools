#!/usr/bin/env python3
"""
Duck Image Decoder CLI Tool
Extracts hidden media files from cartoon duck images

Usage:
    python duck_decoder.py --duck <duck_image> [options]

Examples:
    python duck_decoder.py --duck my_duck.png --out recovered.jpg
    python duck_decoder.py --duck my_duck.png --password "secret123" --out recovered.mp4
    python duck_decoder.py --duck encrypted_duck.png --output-dir ./recovered/
"""

import argparse
import os
import sys
import struct
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List

# Import existing functionality
try:
    import numpy as np
    from PIL import Image
    from moviepy import VideoFileClip
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("Please ensure required packages are installed: pip install -r requirements.txt")
    sys.exit(1)

# Constants from existing implementation
WATERMARK_SKIP_W_RATIO = 0.40
WATERMARK_SKIP_H_RATIO = 0.08
DUCK_CHANNELS = 3

def _extract_payload_with_k(arr: np.ndarray, k: int) -> bytes:
    """
    Extract payload from image array using k bits per channel

    Args:
        arr: Image numpy array
        k: Number of bits to extract per channel

    Returns:
        Extracted payload bytes
    """
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
        raise ValueError("Insufficient image data. / 图像数据不足")
    len_bits = bits[:32]
    length_bytes = np.packbits(len_bits, bitorder="big").tobytes()
    header_len = struct.unpack(">I", length_bytes)[0]
    total_bits = 32 + header_len * 8
    if header_len <= 0 or total_bits > len(bits):
        raise ValueError("Payload length invalid. / 载荷长度异常")
    payload_bits = bits[32:32 + header_len * 8]
    return np.packbits(payload_bits, bitorder="big").tobytes()

def _generate_key_stream(password: str, salt: bytes, length: int) -> bytes:
    """
    Generate keystream for password-based encryption

    Args:
        password: Password string
        salt: Salt bytes
        length: Desired keystream length

    Returns:
        Generated keystream
    """
    import hashlib
    key_material = (password + salt.hex()).encode("utf-8")
    out = bytearray()
    counter = 0
    while len(out) < length:
        combined = key_material + str(counter).encode("utf-8")
        out.extend(hashlib.sha256(combined).digest())
        counter += 1
    return bytes(out[:length])

def _parse_header(header: bytes, password: str):
    """
    Parse header and extract/decrypt data

    Args:
        header: Header bytes
        password: Password for decryption

    Returns:
        Tuple of (decrypted_data, file_extension)
    """
    idx = 0
    if len(header) < 1:
        raise ValueError("Header corrupted. / 文件头损坏")
    has_pwd = header[0] == 1
    idx += 1
    pwd_hash = b""
    salt = b""
    if has_pwd:
        if len(header) < idx + 32 + 16:
            raise ValueError("Header corrupted. / 文件头损坏")
        pwd_hash = header[idx:idx + 32]; idx += 32
        salt = header[idx:idx + 16]; idx += 16
    if len(header) < idx + 1:
        raise ValueError("Header corrupted. / 文件头损坏")
    ext_len = header[idx]; idx += 1
    if len(header) < idx + ext_len + 4:
        raise ValueError("Header corrupted. / 文件头损坏")
    ext = header[idx:idx + ext_len].decode("utf-8", errors="ignore"); idx += ext_len
    data_len = struct.unpack(">I", header[idx:idx + 4])[0]; idx += 4
    data = header[idx:]
    if len(data) != data_len:
        raise ValueError("Data length mismatch. / 数据长度不匹配")
    if not has_pwd:
        return data, ext
    if not password:
        raise ValueError("Password required. / 需要密码")
    import hashlib
    check_hash = hashlib.sha256((password + salt.hex()).encode("utf-8")).digest()
    if check_hash != pwd_hash:
        raise ValueError("Wrong password. / 密码错误")
    ks = _generate_key_stream(password, salt, len(data))
    plain = bytes(a ^ b for a, b in zip(data, ks))
    return plain, ext

def validate_duck_image(image_path: str) -> bool:
    """
    Validate that input is a proper duck image

    Args:
        image_path: Path to the image file

    Returns:
        True if valid duck image
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Duck image not found: {image_path} / 鸭子图片未找到: {image_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {image_path} / 路径不是文件: {image_path}")

    if path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
        raise ValueError(f"Duck image must be PNG/JPG format: {image_path} / 鸭子图片必须是PNG/JPG格式: {image_path}")

    try:
        with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Basic size validation - duck images are typically square
            if img.size[0] != img.size[1]:
                print(f"Warning: Image is not square ({img.size[0]}x{img.size[1]}). May not be a valid duck image.")
            return True
    except Exception as e:
        raise ValueError(f"Invalid image file: {e} / 无效的图片文件: {e}")

def extract_with_fallback(image_path: str, password: str = "") -> Tuple[bytes, str]:
    """
    Try all compression levels (2, 6, 8) until successful extraction

    Args:
        image_path: Path to duck image
        password: Password for decryption (if required)

    Returns:
        Tuple of (extracted_data, file_extension)
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            arr = np.array(img).astype(np.uint8)

        header = None
        raw = None
        ext = None
        last_err = None

        # Try all compression levels
        for k in (2, 6, 8):
            try:
                header = _extract_payload_with_k(arr, k)
                raw, ext = _parse_header(header, password)
                return raw, ext
            except Exception as e:
                last_err = e
                continue

        if raw is None:
            raise last_err or RuntimeError("Failed to extract data from duck image / 无法从鸭子图片中提取数据")

    except Exception as e:
        raise ValueError(f"Extraction failed: {e} / 提取失败: {e}")

def binpng_bytes_to_video_data(png_bytes: bytes) -> bytes:
    """
    Convert binary PNG data back to video bytes

    Args:
        png_bytes: Binary PNG data

    Returns:
        Raw video bytes
    """
    try:
        # Load binary PNG
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(png_bytes)
            tmp_path = tmp.name

        try:
            img = Image.open(tmp_path).convert("RGB")
            arr = np.array(img).astype(np.uint8)
            flat = arr.reshape(-1, 3).reshape(-1)
            return flat.tobytes().rstrip(b"\x00")
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        raise ValueError(f"Failed to convert binary PNG to video data: {e} / 转换二进制PNG到视频数据失败: {e}")

def save_video_from_bytes(video_bytes: bytes, output_path: str, fps: int = 16):
    """
    Save video from binary bytes with specified FPS

    Args:
        video_bytes: Raw video bytes
        output_path: Output video file path
        fps: Frame rate for video reconstruction (default: 16)
    """
    try:
        # First try to save as raw video bytes
        with open(output_path, 'wb') as f:
            f.write(video_bytes)

        # Verify if the file is a valid video and get actual FPS if possible
        import subprocess
        actual_fps = fps
        try:
            # Try to get video info with ffprobe if available
            result = subprocess.run([
                'ffprobe', '-v', 'error', '-show_entries', 'stream=r_frame_rate',
                '-of', 'csv=p=0', output_path
            ], capture_output=True, text=True, timeout=10)

            if result.returncode == 0 and result.stdout.strip():
                try:
                    # Parse frame rate (e.g., "25/1" -> 25.0)
                    fps_str = result.stdout.strip().split(',')[0]
                    if '/' in fps_str:
                        num, den = fps_str.split('/')
                        actual_fps = float(num) / float(den) if float(den) != 0 else fps
                    else:
                        actual_fps = float(fps_str)
                    print(f"Video detected with FPS: {actual_fps}")
                    print(f"检测到视频帧率: {actual_fps}")
                except (ValueError, ZeroDivisionError):
                    actual_fps = fps
                    print(f"Could not parse FPS, using default: {fps}")
                    print(f"无法解析帧率，使用默认值: {fps}")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            # ffprobe not available, use default fps
            print(f"Video file saved (ffprobe not available for FPS detection)")
            print(f"视频文件已保存（无法用ffprobe检测帧率）")

        print(f"Using FPS: {fps} (default: 16)")
        print(f"使用帧率: {fps} (默认: 16)")

    except Exception as e:
        raise ValueError(f"Failed to save video: {e} / 保存视频失败: {e}")

def save_extracted_data(raw_data: bytes, file_ext: str, output_path: str, fps: int = 16):
    """
    Save extracted data to appropriate file format

    Args:
        raw_data: Extracted raw data
        file_ext: File extension
        output_path: Output file path
        fps: Frame rate for video reconstruction (default: 16)
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        if file_ext.endswith(".binpng"):
            # Handle video files - convert binary PNG back to raw video bytes
            print("Converting binary PNG to video data...")
            video_bytes = binpng_bytes_to_video_data(raw_data)
            save_video_from_bytes(video_bytes, output_path, fps)
        else:
            # Handle regular files
            with open(output_path, 'wb') as f:
                f.write(raw_data)

        print(f"Successfully saved extracted data to: {output_path}")
        print(f"成功保存提取的数据到: {output_path}")

    except Exception as e:
        raise ValueError(f"Failed to save extracted data: {e} / 保存提取数据失败: {e}")

def generate_output_path(input_duck_path: str, file_ext: str, output_spec: str = None, output_dir: str = None) -> str:
    """
    Generate appropriate output file path

    Args:
        input_duck_path: Path to input duck image
        file_ext: Extracted file extension
        output_spec: User-specified output path or name
        output_dir: User-specified output directory

    Returns:
        Generated output path
    """
    # Handle .binpng extensions by using the base extension (e.g., "mp4.binpng" -> "mp4")
    clean_ext = file_ext.replace('.binpng', '') if file_ext.endswith('.binpng') else file_ext

    if output_spec:
        # If output_spec is a directory, use it with auto-generated filename
        if os.path.isdir(output_spec):
            output_dir = output_spec
            base_name = Path(input_duck_path).stem + "_recovered"
            output_path = os.path.join(output_dir, f"{base_name}.{clean_ext}")
        else:
            # Use as full path
            output_path = output_spec
            # Ensure extension matches extracted file
            if not output_path.lower().endswith(clean_ext.lower()):
                if '.' in output_path:
                    output_path = output_path.rsplit('.', 1)[0] + '.' + clean_ext
                else:
                    output_path += '.' + clean_ext
    else:
        # Auto-generate path
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = Path(input_duck_path).stem + "_recovered"
            output_path = os.path.join(output_dir, f"{base_name}.{clean_ext}")
        else:
            base_name = Path(input_duck_path).stem + "_recovered"
            output_path = f"{base_name}.{clean_ext}"

    return output_path

def print_progress(message: str, quiet: bool = False):
    """Print progress message if not in quiet mode"""
    if not quiet:
        print(message)

def decode_duck_image(duck_path: str, password: str = "", output: str = None,
                     output_dir: str = None, fps: int = 16, quiet: bool = False) -> str:
    """
    Main decoding function

    Args:
        duck_path: Path to duck image
        password: Password for decryption (if required)
        output: Output file path or directory
        output_dir: Output directory (alternative to --out)
        fps: Frame rate for video reconstruction (default: 16)
        quiet: Suppress verbose output

    Returns:
        Path to the extracted file
    """
    try:
        print_progress(f"Loading duck image: {duck_path}", quiet)

        # Validate duck image
        validate_duck_image(duck_path)

        print_progress("Extracting hidden data...", quiet)

        # Extract data with fallback
        raw_data, file_ext = extract_with_fallback(duck_path, password)

        print_progress(f"Extracted file type: {file_ext}", quiet)
        print_progress(f"Data size: {len(raw_data)} bytes", quiet)

        # Generate output path
        output_path = generate_output_path(duck_path, file_ext, output, output_dir)

        print_progress(f"Saving extracted data to: {output_path}", quiet)

        # Save extracted data
        save_extracted_data(raw_data, file_ext, output_path, fps)

        return output_path

    except Exception as e:
        error_msg = f"Decoding failed: {e}"
        print(error_msg)
        sys.exit(1)

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Extract hidden media files from duck images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --duck my_duck.png
  %(prog)s --duck my_duck.png --password "secret123" --out recovered.jpg
  %(prog)s --duck encrypted_duck.png --output-dir ./recovered/
  %(prog)s --duck my_duck.png --fps 24 --out recovered.mp4

Note:
  This tool will try all compression levels (2, 6, 8) automatically to find the correct data.
  If the duck image is password-protected, you must provide the correct password.
  Use --fps to specify frame rate for video reconstruction (default: 16).
        """
    )

    parser.add_argument(
        '--duck',
        required=True,
        help='Path to duck image file / 鸭子图片文件路径'
    )

    parser.add_argument(
        '--password',
        default='',
        help='Password for decryption (if required) / 解密密码（如果需要）'
    )

    parser.add_argument(
        '--out',
        help='Output file path or directory (default: auto-generate) / 输出文件路径或目录'
    )

    parser.add_argument(
        '--output-dir',
        help='Output directory (alternative to --out) / 输出目录（--out的替代选项）'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output / 静默模式'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=16,
        help='Frame rate for video reconstruction (default: 16) / 视频重建帧率（默认：16）'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='Duck Decoder CLI v1.0'
    )

    args = parser.parse_args()

    # Validate that --out and --output-dir are not both specified
    if args.out and args.output_dir:
        print("Error: Cannot specify both --out and --output-dir")
        print("错误: 不能同时指定 --out 和 --output-dir")
        sys.exit(1)

    # Run decoding
    decode_duck_image(
        duck_path=args.duck,
        password=args.password,
        output=args.out,
        output_dir=args.output_dir,
        fps=args.fps,
        quiet=args.quiet
    )

if __name__ == '__main__':
    main()