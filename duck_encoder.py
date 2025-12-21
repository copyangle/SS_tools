#!/usr/bin/env python3
"""
Duck Image Encoder CLI Tool
Hides media files (images/videos) within cartoon duck images using LSB steganography

Usage:
    python duck_encoder.py <media_file> [options]

Examples:
    python duck_encoder.py photo.jpg --title "My Photo" --out my_duck.png
    python duck_encoder.py video.mp4 --password "secret123" --compress 6 --fps 24
    python duck_encoder.py large_video.mp4 --compress 8 --title "Large File"
"""

import argparse
import os
import sys
import tempfile
import mimetypes
from pathlib import Path
from typing import Optional, Tuple, Union

# Import existing functionality
try:
    from duck_payload_exporter import export_duck_payload, _bytes_to_binary_image
except ImportError as e:
    print(f"Error: Could not import required modules: {e}")
    print("Please ensure duck_payload_exporter.py is in the same directory.")
    sys.exit(1)

# Supported file formats
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.webm', '.wmv', '.m4v'}

def get_file_info(file_path: str) -> Tuple[bool, str, str]:
    """
    Detect file type and return (is_video, extension, file_type)

    Args:
        file_path: Path to the input file

    Returns:
        Tuple of (is_video, extension, file_type)
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path} / 文件未找到: {file_path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {file_path} / 路径不是文件: {file_path}")

    is_video = ext in VIDEO_EXTENSIONS
    is_image = ext in IMAGE_EXTENSIONS

    if not (is_video or is_image):
        raise ValueError(f"Unsupported file format: {ext}. / 不支持的文件格式: {ext}\n"
                        f"Supported formats: Images {IMAGE_EXTENSIONS}, Videos {VIDEO_EXTENSIONS}")

    file_type = "video" if is_video else "image"
    return is_video, ext[1:], file_type

def process_image_file(file_path: str) -> bytes:
    """
    Process image file and return bytes

    Args:
        file_path: Path to the image file

    Returns:
        Raw image bytes
    """
    from PIL import Image

    try:
        with Image.open(file_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Create temporary file to save as PNG
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                img.save(tmp.name, format='PNG', optimize=True, compress_level=9)
                tmp_path = tmp.name

            try:
                # Read the PNG bytes
                with open(tmp_path, 'rb') as f:
                    return f.read()
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)

    except Exception as e:
        raise ValueError(f"Failed to process image: {e} / 图像处理失败: {e}")

def process_video_file(file_path: str, fps: int = 16) -> bytes:
    """
    Process video file and convert to bytes using moviepy

    Args:
        file_path: Path to the video file
        fps: Target frame rate for processing

    Returns:
        Raw video bytes as binary image
    """
    try:
        from moviepy import VideoFileClip
        from duck_payload_exporter import _bytes_to_binary_image
        import numpy as np
    except ImportError as e:
        raise ImportError(f"Required moviepy module not found: {e} / 未找到必需的moviepy模块: {e}")

    try:
        clip = VideoFileClip(file_path)

        # Extract frames at specified fps
        duration = clip.duration
        total_frames = int(duration * fps)

        frames = []
        for i in range(total_frames):
            t = i / fps
            if t > duration:
                break
            frame = clip.get_frame(t)
            # Ensure frame values are in proper range
            frame = np.clip(frame, 0.0, 1.0)
            # Convert frame to RGB and normalize
            frame_rgb = (frame * 255).astype(np.uint8)
            frames.append(frame_rgb)

        clip.close()

        if not frames:
            raise ValueError("No frames extracted from video / 无法从视频中提取帧")

        # Stack frames and convert to binary image (frames are already uint8)
        frames_array = np.stack(frames, axis=0)

        # Reshape to fit binary image format (height, width, 3)
        # Use a fixed width of 512 pixels
        binary_width = 512
        binary_height = (frames_array.size // (binary_width * 3)) + 1

        # Convert all frame data to flat bytes
        flat_bytes = frames_array.tobytes()

        # Create binary image representation
        binary_img = _bytes_to_binary_image(flat_bytes, width=binary_width)

        # Save binary image to temporary file and read bytes
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            binary_img.save(tmp.name, format='PNG')
            tmp_path = tmp.name

        try:
            with open(tmp_path, 'rb') as f:
                return f.read()
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        raise ValueError(f"Failed to process video: {e} / 视频处理失败: {e}")

def process_media_file(file_path: str, fps: int = 16) -> Tuple[bytes, str]:
    """
    Process input media and return (raw_bytes, file_extension)

    Args:
        file_path: Path to the media file
        fps: Frame rate for video processing

    Returns:
        Tuple of (processed_bytes, file_extension)
    """
    is_video, ext, file_type = get_file_info(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path} / 文件未找到: {file_path}")

    file_size = os.path.getsize(file_path)
    if file_size == 0:
        raise ValueError(f"File is empty: {file_path} / 文件为空: {file_path}")

    if not is_video:
        # Process image file
        raw_bytes = process_image_file(file_path)
        # Use "binpng" as extension for binary images
        return raw_bytes, f"{ext}.binpng"
    else:
        # Process video file
        raw_bytes = process_video_file(file_path, fps)
        return raw_bytes, f"{ext}.binpng"

def validate_compression_level(compress: int) -> int:
    """
    Validate and return compression level

    Args:
        compress: Compression level input

    Returns:
        Validated compression level (2, 6, or 8)
    """
    if compress not in [2, 6, 8]:
        raise ValueError(f"Invalid compression level: {compress}. / 无效的压缩级别: {compress}\n"
                        f"Valid levels: 2 (highest capacity), 6 (medium), 8 (best quality)")
    return compress

def validate_output_path(output_path: str) -> str:
    """
    Validate and prepare output path

    Args:
        output_path: Output file path

    Returns:
        Validated output path
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)

    # Ensure file has .png extension
    if not output_path.lower().endswith('.png'):
        output_path += '.png'

    return output_path

def print_progress(message: str, quiet: bool = False):
    """Print progress message if not in quiet mode"""
    if not quiet:
        print(message)

def encode_media(media_file: str, title: str = "", password: str = "",
                compress: int = 2, output: str = "duck_payload.png",
                fps: int = 16, quiet: bool = False) -> str:
    """
    Main encoding function using existing export_duck_payload

    Args:
        media_file: Path to input media file
        title: Custom title for duck image
        password: Password for encryption
        compress: Compression level (2/6/8)
        output: Output duck image path
        fps: Frame rate for video processing
        quiet: Suppress verbose output

    Returns:
        Path to the generated duck image
    """
    try:
        print_progress(f"Processing media file: {media_file}", quiet)

        # Validate inputs
        validate_compression_level(compress)
        output_path = validate_output_path(output)

        # Process media file
        print_progress(f"Converting media to binary format...", quiet)
        raw_bytes, file_ext = process_media_file(media_file, fps)

        print_progress(f"Encoding data into duck image...", quiet)
        print_progress(f"File extension: {file_ext}, Compression level: {compress}", quiet)

        # Generate duck image with embedded payload
        output_dir = os.path.dirname(output_path)
        output_name = os.path.basename(output_path)

        result_path, duck_img = export_duck_payload(
            raw_bytes=raw_bytes,
            password=password,
            ext=file_ext,
            compress=compress,
            title=title,
            output_dir=output_dir,
            output_name=output_name
        )

        print_progress(f"Successfully created duck image: {result_path}", quiet)
        print_progress(f"Duck image size: {duck_img.size[0]}x{duck_img.size[1]}", quiet)

        return result_path

    except Exception as e:
        error_msg = f"Encoding failed: {e}"
        print(error_msg)
        sys.exit(1)

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Encode media files into duck images using steganography",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s photo.jpg --title "My Photo" --out my_duck.png
  %(prog)s video.mp4 --password "secret123" --compress 6 --fps 24
  %(prog)s large_video.mp4 --compress 8 --title "Large File"

Supported formats:
  Images: png, jpg, jpeg, bmp, webp
  Videos: mp4, avi, mov, mkv, flv, webm, wmv, m4v
        """
    )

    parser.add_argument(
        'media_file',
        help='Path to input media file (image or video) / 输入媒体文件路径'
    )

    parser.add_argument(
        '--title',
        default='',
        help='Custom title for duck image (default: empty) / 鸭子图片的自定义标题'
    )

    parser.add_argument(
        '--password',
        default='',
        help='Password for encryption (default: no encryption) / 加密密码'
    )

    parser.add_argument(
        '--compress',
        type=int,
        choices=[2, 6, 8],
        default=2,
        help='Compression level - 2 (highest capacity), 6 (medium), 8 (best quality) / 压缩级别'
    )

    parser.add_argument(
        '--out',
        default='duck_payload.png',
        help='Output duck image path (default: duck_payload.png) / 输出鸭子图片路径'
    )

    parser.add_argument(
        '--fps',
        type=int,
        default=16,
        help='Video frame rate for processing (default: 16) / 视频处理的帧率'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress verbose output / 静默模式'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='Duck Encoder CLI v1.0'
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.media_file):
        print(f"Error: Input file not found: {args.media_file}")
        print(f"错误: 输入文件未找到: {args.media_file}")
        sys.exit(1)

    # Run encoding
    encode_media(
        media_file=args.media_file,
        title=args.title,
        password=args.password,
        compress=args.compress,
        output=args.out,
        fps=args.fps,
        quiet=args.quiet
    )

if __name__ == '__main__':
    main()