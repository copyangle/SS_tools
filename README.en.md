<div align="center">
  <a href="README.md">🇨🇳 Chinese</a> | 
  <a href="README.en.md">🇬🇧 English</a>
</div>

# SSTool Super-SecureMediaProtection Media Content Protection

Disclaimer:
The original intention of this project is to provide an interesting file packaging method, and no commercial use has been considered.
This project is entirely a public welfare project and promises not to charge in any form or charge in a disguised form.
The developer shall not be liable for any losses, liabilities, or legal risks arising from the use of this project; all the above shall be borne by the user. Your use or deployment of this project shall be deemed as your consent.

![Media Protection Tool Demo](https://github.com/copyangle/SS_tools/blob/main/test.png "Duck Image Media Protection Tool")

## Main Functions:
- Media content protection: Hide images/videos in cartoon duck images, with optional password protection
- Media content extraction: Extract original image/video data from duck images
- Provides a ComfyUI workflow for the aforementioned functions
- Provides local EXE files for the aforementioned functions
- Provide local tools for macOS

## Related Links:
- Video Tutorial: https://www.bilibili.com/video/BV1w6qYBREzV
- Web Version User Guide: https://www.bilibili.com/video/BV1K2icBqES3/
- Ready-made Workflow: https://www.runninghub.ai/post/2000762471365439490/?inviteCode=rh-v1393
- Unrestricted Wan2.2Remix Image-to-Video: Full Demo of Duck Image: https://www.runninghub.ai/ai-detail/2000755012961792002/?inviteCode=rh-v1393
- Unrestricted Scail Motion Transfer: https://www.runninghub.ai/post/2001603322555559938/?inviteCode=rh-v1393
- Unrestricted Wan2.2Remix High-Quality Text-to-Video: https://www.runninghub.cn/post/1988716572640423937/?inviteCode=rh-v1393
- Special Thanks to T8 for Supporting This Project: https://www.bilibili.com/video/BV1vqqeB1ENo/
- Discussion QQ Group: 1067393850
- Little Yellow Duck Decoder - Free Web Version：http://duck.airush.top:81/

## Example:
Workflow for hiding and protecting images and videos
![Media Protection Tool Demo](https://github.com/copyangle/SS_tools/blob/main/Workflow%20Example/encode_img.png "Duck Image Media Protection Tool")
![Media Protection Tool Demo](https://github.com/copyangle/SS_tools/blob/main/Workflow%20Example/encode_video.png "Duck Image Media Protection Tool")

Workflow for extracting images and videos
![Media Protection Tool Demo](https://github.com/copyangle/SS_tools/blob/main/Workflow%20Example/decode_img.png "Duck Image Media Protection Tool")
![Media Protection Tool Demo](https://github.com/copyangle/SS_tools/blob/main/Workflow%20Example/decode_video.png "Duck Image Media Protection Tool")


## Local Node Deployment Method
- Method 1:
  - cd `ComfyUI/custom_nodes/`
  - git clone git@github.com:copyangle/SS_tools.git
  - cd SS_tools
  - pip install -r requirements.txt

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