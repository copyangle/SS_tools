<div align="center">
  <a href="README.md">🇨🇳 Chinese</a> | 
  <a href="README.en.md">🇬🇧 English</a>
</div>

# SSTool Super-SecureMediaProtection Media Content Protection V1.2

Disclaimer:
This project is originally designed to offer an interesting file packaging solution, with core functions identical to mainstream compression tools such as ZIP and RAR. All steganography and encryption algorithms are open-sourced.
This is a completely non-profit project, intended solely for personal learning, academic research, technical demonstration and personal data privacy protection.
It is prohibited to use this technology for any activities that violate local laws and regulations, or to infringe upon others' data, copyright and other legitimate rights and interests.
The developer shall not be held liable for any losses, liabilities or legal risks arising from the use of this project, which shall be borne entirely by the user.
Your use or deployment of this project constitutes acceptance of the above terms.

![Media Protection Tool Demo](https://github.com/copyangle/SS_tools/blob/main/test.png "Duck Image Media Protection Tool")

## Main Functions:
- Media content protection: Hide images/videos in cartoon duck images, with optional password protection
- Media content extraction: Extract original image/video/string data from duck images
- Provides a ComfyUI workflow for the aforementioned functions
- Provides local EXE files for the aforementioned functions
- Provide local tools for macOS
- Duck Image Local UI Decoder Lite V1.0: 鸭鸭图本地UI解码工具精简版V1.0.rar
- Duck Browser Extension：http://duckp.airush.top/

## Related Links:
- [Introduction to Windows Local EXE Tool](https://www.youtube.com/watch?v=Cr9ulXU7z08)
- [Usage of QR‑Code Encoding Node](https://www.youtube.com/watch?v=3rgMM1Hq0RM)
- [Introduction to Browser‑Side Decoding Plugin](https://www.youtube.com/watch?v=r7ivgZ5I7DE)

## Example:
windows tool
![Media Protection Tool Demo](https://github.com/copyangle/SS_tools/blob/main/Workflow%20Example/gui_exe.png "Duck Image Media Protection Tool")

Workflow for hiding and protecting images and videos
![Media Protection Tool Demo](https://github.com/copyangle/SS_tools/blob/main/Workflow%20Example/encode_img.png "Duck Image Media Protection Tool")
![Media Protection Tool Demo](https://github.com/copyangle/SS_tools/blob/main/Workflow%20Example/encode_video.png "Duck Image Media Protection Tool")

Workflow for extracting images and videos
![Media Protection Tool Demo](https://github.com/copyangle/SS_tools/blob/main/Workflow%20Example/decode_img.png "Duck Image Media Protection Tool")
![Media Protection Tool Demo](https://github.com/copyangle/SS_tools/blob/main/Workflow%20Example/decode_video.png "Duck Image Media Protection Tool")

Workflow for String
![Media Protection Tool Demo](https://github.com/copyangle/SS_tools/blob/main/Workflow%20Example/encode_decode_string.png "Duck Image Media Protection Tool")

## Local Node Deployment Method
- Method 1:
  - Requires installation of ffmpeg, available from https://ffmpeg.org/download.html and added to the environment variables
  - cd `ComfyUI/custom_nodes/`
  - git clone git@github.com:copyangle/SS_tools.git
  - cd SS_tools
  - pip install -r requirements.txt
- ps:moviepy>=2.0，numpy<=1.26.4

**Component Overview**
- ConfyUI nodes:
  - `duck_encode_node` (Hide images/videos in cartoon duck images)
  - `duck_decode_node` (Extract original images/videos from duck images)
- Executable tools:
  - `duck_encoder.exe` (Generate duck images locally, supporting images/videos)
  - `duck_decoder.exe` (Decode payload from duck images, supporting passwords)

**duck_encode_node**
- Function: Hide image or video data into cartoon duck images, with optional password and title
- Inputs:
  - `images` (optional `IMAGE`): Single-frame or multi-frame images
  - `audio` (optional `AUDIO`): Audio input, optional when inputting multiple frames
  - `password` (`STRING`): Leave blank for no encryption; fill in to enable password protection
  - `title` (`STRING`): Draw a title on the duck image
  - `fps` (`INT`): Frame rate when synthesizing video (default 16)
  - `compress` (`INT`): LSB bit width (2/6/8) affects capacity and image quality `duck_payload_exporter.py:187`
- Outputs:
  - `duck_image` (`IMAGE`): Duck image containing steganographic data


**duck_decode_node**
- Function: Extract original image or video data from duck images
- Inputs:
  - `image` (`IMAGE`): Duck image
  - `password` (`STRING`, optional): Required if encrypted
- Outputs:
  - `images` (`IMAGE`): Restored image sequence or single frame
  - `audio` (`AUDIO`): Audio can be recovered when the payload is a video
  - `file_path` (`STRING`): Path of the restored file on the disk
  - `fps` (`INT`): Frame rate when the payload is a video

## Local Protection/Extraction Tools

**duck_encoder.exe**
- Function: Encode media files into duck images locally
- Basic usage:
  - View help: `duck_encoder.exe --help`
  - Encode image: `duck_encoder.exe media_file.png --title Title --password Password --compress 2 --out duck_payload.png`
  - Encode video: `duck_encoder.exe media_file.mp4 --title Title --password Password --compress 2 --out duck_payload.png`
- Parameters:
  - `media`: Image (png/jpg/jpeg/bmp/webp) or video (mp4/avi/mov)
  - `--title`: Draw a title on the duck image
  - `--password`: Enable password protection (stream XOR)
  - `--compress`: Three levels (2/6/8); larger bit width means higher capacity but more impact on image quality
  - `--out`: Output file name, default
- Explanation:
  - Videos will be converted to "binary images" first for steganography to avoid loss of audio and other information

**duck_decoder.exe**
- Function: Decode original payload (image/video/binary) from duck images
- Basic usage:
  - Without password: `duck_decoder.exe --duck duck_payload.png --out recovered.bin`
  - With password: `duck_decoder.exe --duck duck_payload.png --out recovered.mp4 --password YourPassword`
- Parameters:
  - `--duck`: Path of the input duck image
  - `--out`: Path of the output file (the suffix will be automatically matched according to the payload type)
  - `--password`: Required if encrypted


**Development Reference**
- Node registration: `__init__.py:1`

- Output the original image/video to the path specified by `--out` (if left blank, it will be automatically named according to the extension).

## Notes
- Do not re-save the duck image with image editing software to avoid truncation of tail data.
- If a password is set during generation, the same password must be provided for decoding, otherwise a verification failure will be prompted.
- Mobile web decoding is only available on the following four browsers:Apple's native browser, WeChat in-app browser (iOS), Google Chrome (all systems), and Firefox (all systems).