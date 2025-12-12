import hashlib
import io
import os
import os
import struct
from typing import Tuple, List, Any
from moviepy import ImageSequenceClip, AudioFileClip, concatenate_audioclips
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import tempfile
try:
    import folder_paths  # type: ignore
except Exception:  # pragma: no cover
    folder_paths = None
from .duck_payload_exporter import export_duck_payload, _bytes_to_binary_image


# 分类名称要求
CATEGORY = "SSTool"
LSB_BITS_PER_CHANNEL = 2
DUCK_CHANNELS = 3
WATERMARK_SKIP_W_RATIO = 0.40
WATERMARK_SKIP_H_RATIO = 0.08




def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """将 ComfyUI 的 IMAGE Tensor 转为 PIL.Image。"""
    if image.dim() == 4:
        image = image[0]
    image = image.detach().cpu().numpy()
    image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(image)


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    """将 PIL.Image 转为 ComfyUI 需要的 IMAGE Tensor。"""
    arr = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None, ...]




# ===================== 核心：复用VideoHelperSuite的音频解析逻辑 =====================
def export_lazy_audio_to_file(audio_obj) -> str:

    path_attr = getattr(audio_obj, "file", None)
    if isinstance(path_attr, str) and os.path.exists(path_attr):
        return path_attr


    temp_audio_path = tempfile.mkstemp(suffix=".wav")

    try:

        if isinstance(audio_obj, torch.Tensor):
            import soundfile as sf
            audio_np = audio_obj.detach().cpu().numpy()
            if audio_np.ndim == 1:
                audio_np = audio_np[:, None]
            if audio_np.ndim == 2 and audio_np.shape[0] < audio_np.shape[1]:
                audio_np = audio_np.T
            sf.write(temp_audio_path, audio_np, samplerate=44100)
            print(f"✅ 导出音频张量为WAV：{temp_audio_path}")
            print(f"✅ Export audio tensor to WAV: {temp_audio_path}")
            return temp_audio_path

        elif isinstance(audio_obj, np.ndarray):
            import soundfile as sf
            arr = audio_obj
            if arr.ndim == 1:
                arr = arr[:, None]
            if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
                arr = arr.T
            sf.write(temp_audio_path, arr, samplerate=44100)
            print(f"✅ 导出numpy音频为WAV：{temp_audio_path}")
            print(f"✅ Export numpy audio to WAV: {temp_audio_path}")
            return temp_audio_path

        elif isinstance(audio_obj, dict):
            import soundfile as sf
            data = audio_obj.get("samples") or audio_obj.get("audio")
            sr = audio_obj.get("sample_rate") or audio_obj.get("samplerate") or 44100
            if data is not None:
                arr = np.array(data)
                if arr.ndim == 1:
                    arr = arr[:, None]
                if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
                    arr = arr.T
                sf.write(temp_audio_path, arr, samplerate=int(sr))
                print(f"✅ 导出dict音频为WAV：{temp_audio_path}")
                print(f"✅ Export dict audio to WAV: {temp_audio_path}")
                return temp_audio_path

        elif isinstance(audio_obj, tuple) and len(audio_obj) >= 1:
            import soundfile as sf
            arr = np.array(audio_obj[0])
            sr = audio_obj[1] if len(audio_obj) > 1 else 44100
            if arr.ndim == 1:
                arr = arr[:, None]
            if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
                arr = arr.T
            sf.write(temp_audio_path, arr, samplerate=int(sr))
            print(f"✅ 导出tuple音频为WAV：{temp_audio_path}")
            print(f"✅ Export tuple audio to WAV: {temp_audio_path}")
            return temp_audio_path

        elif isinstance(audio_obj, str):
            if os.path.exists(audio_obj):
                import shutil
                shutil.copy2(audio_obj, temp_audio_path)
                print(f"✅ 复制音频文件到临时路径：{temp_audio_path}")
                print(f"✅ Copy audio file to temp path: {temp_audio_path}")
                return temp_audio_path
            raise FileNotFoundError(f"音频路径不存在：{audio_obj}")

        raise TypeError(f"不支持的音频类型：{type(audio_obj)}")

    except Exception as e:
        print(f"❌ Export audio failed: {str(e)}")
        print(f"❌ 导出音频失败：{str(e)}")
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
        return None

class DuckHideNode:
    """生成鸭子图并将真实图片/视频数据隐藏其中，可选密码保护。"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "password": ("STRING", {"default": "", "multiline": False}),
                "title": ("STRING", {"default": "", "multiline": False}),
                "fps": ("INT", {"default": 16, "min": 1, "max": 60, "step": 1}),
                "compress": ([2, 6, 8], {"default": 2}),
                "Notes": ("STRING", {
                    "multiline": True,        # 核心：开启多行模式
                    "default": "使用方法：https://github.com/copyangle/SS_tools\n1. 支持图片/视频隐写保护\n2. compress: 2/6/8 选择压缩方式，8为最小体积",  # 多行默认内容
                    "placeholder": "使用方法：https://github.com/copyangle/SS_tools",  # 输入提示（可选）
                    "dynamicPrompts": False,  # 关闭动态提示（按需开启）
                    "rows": 3,                # 可选：指定输入框默认行数（视觉效果）
                }),
            },
            "optional": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("duck_image",)
    FUNCTION = "hide"
    CATEGORY = CATEGORY

    def _convert_comfy_image_to_cv2(self, comfy_image):
        img = np.rint(comfy_image.detach().cpu().numpy() * 255.0).astype(np.uint8)
        if img.ndim == 4:
            img = img.squeeze(0)
        if img.shape[-1] == 4:
            img = img[..., :3]
        return img

    def _parse_comfy_audio(self, audio: Any) -> str:
        """
        统一解析音频：复用保存音频节点的export逻辑
        """
        if audio is None or audio == "":
            return None
        
        # 核心：调用导出函数，将任意音频对象转为临时文件路径
        audio_path = export_lazy_audio_to_file(audio)
        return audio_path

    def _images_to_video(self, images: List[Image.Image], fps: int,audio: Any) -> np.ndarray:
        # """将多张图片合成视频"""
        frame_list = []
        for img in images:
            rgb_img = self._convert_comfy_image_to_cv2(img)
            frame_list.append(rgb_img)

        clip = ImageSequenceClip(frame_list, fps=fps)
        audio_clip = None
        if audio is not None and audio != "":
            print("检测到音频输入，嵌入视频中")
            print("Audio detected, embedding into video")
            # 处理ComfyUI的AUDIO输入（兼容路径/二进制数据）
            audio_path = self._parse_comfy_audio(audio)
            print("audio_path：",audio_path)
            print("audio_path:", audio_path)
            if audio_path is not None:
                audio_clip = AudioFileClip(audio_path)
                if audio_clip.duration > clip.duration:
                    audio_clip = audio_clip.subclip(0, clip.duration)
                else:
                    repeats = int(clip.duration // audio_clip.duration)
                    remainder = clip.duration - repeats * audio_clip.duration
                    parts = []
                    if repeats <= 0:
                        parts.append(audio_clip.subclip(0, min(audio_clip.duration, clip.duration)))
                    else:
                        for _ in range(repeats):
                            parts.append(audio_clip)
                        if remainder > 0:
                            parts.append(audio_clip.subclip(0, remainder))
                    audio_clip = concatenate_audioclips(parts)
                clip = clip.with_audio(audio_clip)
        try:
        # 先写入临时内存文件，再读取二进制
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
                temp_video_path = tf.name
                clip.write_videofile(
                    temp_video_path ,
                    codec="libx264",
                    audio_codec="aac",
                    fps=fps,
                    ffmpeg_params=[
                        "-pix_fmt","yuv420p",
                        "-crf","16",
                        "-preset","medium",
                        "-profile:v","high",
                        "-movflags","+faststart"
                    ]
                )
            # 读取临时文件为二进制数据
            with open(temp_video_path, "rb") as f:
                video_bytes = f.read()
        finally:
            # 强制释放资源（避免句柄占用）
            clip.close()
            if audio_clip:
                audio_clip.close()
            # print("temp_video_path：",temp_video_path)
            # 删除临时文件（清理磁盘）
            if temp_video_path and os.path.exists(temp_video_path):
                os.unlink(temp_video_path)

        return video_bytes

    def hide(self, fps: int , password: str, title: str , compress: int, images=None, audio=None, Notes: str = ""):
        return self._hide(fps, password, title, compress, images, audio, Notes)

    def _hide(self, fps: int , password: str, title: str , compress: int, images=None, audio=None, Notes: str = "", video_path=""):
        # image 
        if images is None and not video_path:
            raise ValueError("Images or video_path required. 需要提供 images 或 video_path 。")

        if isinstance(images, (torch.Tensor, np.ndarray)):
            # 获取维度：3维=单帧，4维=多帧序列
            dims = len(images.shape)
            if dims == 4:
                # Load Video输出的4维帧序列 (N, H, W, C)
                frame_count = images.shape[0]
                # 拆分4维张量为单帧列表
                frame_list = [images[i] for i in range(frame_count)]
            elif dims == 3:
                # 单张图片 (H, W, C)
                frame_count = 1
                frame_list = [images]
            else:
                raise ValueError(f"Unsupported input dims: {dims} (only 3/4D). 不支持的输入维度：{dims}（仅支持3/4维）")
        elif isinstance(images, list):
            # 手动传入的图片列表
            frame_count = len(images)
            frame_list = images
        else:
            # 未知类型，按单帧处理
            frame_count = 1
            frame_list = [images]


        if video_path:
            # 直接使用视频文件
            with open(video_path, "rb") as f:
                vid_bytes = f.read()
                # 转为二进制图片，再走图片逻辑
                bin_img = _bytes_to_binary_image(vid_bytes, width=512)
                with io.BytesIO() as buf:
                    bin_img.save(buf, format="PNG")
                    raw_bytes = buf.getvalue()
                orig_ext = os.path.splitext(video_path)[1].lower().lstrip('.')
                ext = f"{orig_ext}.binpng"

        elif frame_count > 1:
            print("检测到视频输入，合成视频中")
            print("Detected video input, composing video")
            print("图片张数：",frame_count)
            print("Number of images:", frame_count)
            #合成视频
            vid_bytes = self._images_to_video(frame_list, fps,audio)

            # 转为二进制图片，再走图片逻辑
            bin_img = _bytes_to_binary_image(vid_bytes, width=512)
            with io.BytesIO() as buf:
                bin_img.save(buf, format="PNG")
                raw_bytes = buf.getvalue()
            orig_ext = "mp4"
            ext = f"{orig_ext}.binpng"
        else:
            pil = _tensor_to_pil(frame_list[0])
            with io.BytesIO() as buf:
                pil.save(buf, format="PNG")
                raw_bytes = buf.getvalue()
            ext = "png"

        out_path, duck_img = export_duck_payload(
            raw_bytes=raw_bytes,
            password=password,
            ext=ext,
            compress=compress,
            title=title,
            output_dir=(folder_paths.get_output_directory() if folder_paths else os.getcwd()),
            output_name="duck_payload.png",
        )

        duck_tensor = _pil_to_tensor(duck_img)
        return (duck_tensor,)


NODE_CLASS_MAPPINGS = {"DuckHideNode": DuckHideNode}
NODE_DISPLAY_NAME_MAPPINGS = {"DuckHideNode": "SSuper-SecureMediaProtection媒体内容保护V1.0"}

