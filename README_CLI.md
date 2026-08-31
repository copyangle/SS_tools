# SS_tools Python CLI Documentation

## Overview

SS_tools provides Python CLI tools for steganography - hiding media files (images/videos) within cartoon duck images using LSB (Least Significant Bit) encoding.

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Make the CLI scripts executable:
```bash
chmod +x duck_encoder.py duck_decoder.py
```

## CLI Tools

### duck_encoder.py

Encodes images and videos into duck images using steganography.

#### Usage
```bash
python duck_encoder.py <media_file> [options]
```

#### Arguments
- `media_file` (required): Path to input media file (image or video)

#### Options
- `--title <text>`: Custom title for duck image (default: empty)
- `--password <text>`: Password for encryption (default: no encryption)
- `--compress <2|6|8>`: Compression level (default: 2)
  - `2`: Highest capacity (most data hidden)
  - `6`: Medium capacity/quality balance
  - `8`: Best quality (least data hidden)
- `--out <path>`: Output duck image path (default: duck_payload.png)
- `--fps <number>`: Video frame rate for processing (default: 16)
- `--quiet`: Suppress verbose output
- `--version`: Show version information

#### Examples

**Basic image encoding:**
```bash
python duck_encoder.py photo.jpg --title "My Photo" --out my_duck.png
```

**Video encoding with password:**
```bash
python duck_encoder.py video.mp4 --password "secret123" --compress 6 --fps 24
```

**High-quality encoding:**
```bash
python duck_encoder.py large_video.mp4 --compress 8 --title "Large File"
```

### duck_decoder.py

Extracts hidden media files from duck images.

#### Usage
```bash
python duck_decoder.py --duck <duck_image> [options]
```

#### Required Arguments
- `--duck <path>`: Path to duck image file

#### Options
- `--password <text>`: Password for decryption (if required)
- `--out <path>`: Output file path or directory (default: auto-generate)
- `--output-dir <path>`: Output directory (alternative to --out)
- `--quiet`: Suppress verbose output
- `--version`: Show version information

#### Examples

**Basic decoding:**
```bash
python duck_decoder.py --duck my_duck.png
```

**Password-protected decoding:**
```bash
python duck_decoder.py --duck my_duck.png --password "secret123" --out recovered.mp4
```

**Specify output directory:**
```bash
python duck_decoder.py --duck encrypted_duck.png --output-dir ./recovered/
```

## Supported File Formats

### Images
- PNG
- JPG/JPEG
- BMP
- WebP

### Videos
- MP4
- AVI
- MOV
- MKV
- FLV
- WebM
- WMV
- M4V

## Features

### Password Protection
- Optional password-based encryption
- Uses XOR encryption with SHA-256 key derivation
- 16-byte salt prevents rainbow table attacks

### Compression Levels
- **Level 2**: Maximum data capacity (2 bits per RGB channel)
- **Level 6**: Balanced capacity/quality (6 bits per RGB channel)
- **Level 8**: Best visual quality (8 bits per RGB channel)

### Duck Image Generation
- Automatically generates cartoon duck images
- Dynamic canvas sizing based on payload
- Preserves watermarked areas
- Adds custom titles and version information

### Multi-format Support
- Handles both static images and videos
- Maintains audio tracks in videos
- Automatic file type detection
- Lossless binary conversion for media preservation

## Error Handling

The CLI tools provide bilingual error messages (English/Chinese) for:
- File not found errors
- Unsupported format errors
- Password authentication failures
- Capacity limit exceeded
- File permission issues
- Corrupted duck images

## Technical Details

### LSB Steganography
- Data is hidden in the Least Significant Bits of RGB pixels
- Skips watermark regions to preserve visual integrity
- Configurable bit depth for capacity/quality trade-offs

### Security Features
- Salt-based password hashing
- SHA256 integrity verification
- Stream cipher encryption
- Header-based metadata storage

### Video Processing
- Converts videos to binary image format
- Preserves audio through MoviePy integration
- Configurable frame rates for processing
- Automatic format reconstruction

## Troubleshooting

### Common Issues

1. **Import errors**: Install requirements with `pip install -r requirements.txt`
2. **Permission denied**: Ensure scripts are executable with `chmod +x *.py`
3. **File not found**: Check file paths and permissions
4. **Wrong password**: Ensure correct password case sensitivity
5. **Large files**: Use higher compression levels or smaller input files

### Performance Tips

- Use compression level 2 for maximum capacity
- Use compression level 8 for best visual quality
- Consider file size limitations for very large videos
- Use appropriate video frame rates (16-30 fps)

## Integration Examples

### Batch Processing
```bash
# Encode multiple images
for file in *.jpg; do
    python duck_encoder.py "$file" --out "encoded_${file%.jpg}.png"
done

# Decode multiple duck images
for file in encoded_*.png; do
    python duck_decoder.py --duck "$file" --output-dir "./decoded/"
done
```

### Script Integration
```bash
#!/bin/bash
# Example script for automated encoding

INPUT_DIR="./input"
OUTPUT_DIR="./output"
PASSWORD="my_secret_password"

mkdir -p "$OUTPUT_DIR"

for file in "$INPUT_DIR"/*; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        python duck_encoder.py "$file" --password "$PASSWORD" --out "$OUTPUT_DIR/duck_${filename%.*}.png"
    fi
done
```

## Compatibility

- Python 3.6+
- Cross-platform (Windows, macOS, Linux)
- Requires MoviePy for video processing
- Uses Pillow for image manipulation
- NumPy for efficient array operations

## License and Usage

This tool is provided for educational and creative purposes. Please ensure you have the right to encode any media files and respect copyright laws.

## Support

For issues, questions, or feature requests, please refer to the project repository:
https://github.com/copyangle/SS_tools