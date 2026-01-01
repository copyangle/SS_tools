<!-- 顶部添加语言切换区 -->
<div align="center">
  <a href="README.md">🇨🇳 中文</a> | 
  <a href="README.en.md">🇬🇧 English</a>
</div>

# SSTool Super-SecureMediaProtection媒体内容保护 

声明：
本项目初衷是提供一种有趣的文件打包方式，未考虑任何商业用途。
本项目完全是公益项目，承诺绝不以任何形式收费或者变相收费。
开发者本人不承担任何因使用本项目而导致的损失、责任以及法律风险，以上均由使用者自行承担。
如果您使用或者部署此项目即视为同意。

This project is a completely non-profit initiative. 
The developer shall not be liable for any losses or liabilities arising from the use of this project, and all relevant legal risks shall be borne by the users themselves. 
Your use or deployment of this project shall constitute your acceptance of the above terms.

![媒体保护工具演示](https://github.com/copyangle/SS_tools/blob/main/test.png "鸭子图媒体保护工具")

## 主要功能和特点：
- 媒体内容保护：将图片/视频隐藏在卡通鸭子图中，可选密码保护
- 媒体内容提取：从鸭子图中提取原始图片/视频数据
- 提供以上功能的ConfyUI工作流节点
- 提供以上功能的本地exe文件，支持本地打包、提取
- 打包、提取速度极快
- 压缩模式3档可选
- 提供macOS本地工具

## 相关链接：
- 视频教程: https://www.bilibili.com/video/BV1w6qYBREzV
- 现成工作流：https://www.runninghub.cn/post/2000762471365439490/?inviteCode=rh-v1393
- 无限制Wan2.2Remix图生视频鸭子图完整演示:https://www.runninghub.cn/ai-detail/2000755012961792002/?inviteCode=rh-v1393
- 无限制Scail动作迁移 https://www.runninghub.cn/post/2001603322555559938/?inviteCode=rh-v1393
- 无限制Wan2.2Remix高质量文生视频 https://www.runninghub.cn/post/1988716572640423937/?inviteCode=rh-v1393
- 感谢T8对本项目的支持：https://www.bilibili.com/video/BV1vqqeB1ENo/
- 交流Q群：1067393850
- AIGC交流微信群：
  ![AIGC交流微信群](https://github.com/copyangle/SS_tools/blob/main/AIGCgroup.jpg "AIGC交流微信群")

## Example:
隐藏保护图片和视频工作流
![媒体保护工具演示](https://github.com/copyangle/SS_tools/blob/main/Workflow%20Example/encode_img.png "鸭子图媒体保护工具")
![媒体保护工具演示](https://github.com/copyangle/SS_tools/blob/main/Workflow%20Example/encode_video.png "鸭子图媒体保护工具")

提取图片和视频工作流
![媒体保护工具演示](https://github.com/copyangle/SS_tools/blob/main/Workflow%20Example/decode_img.png "鸭子图媒体保护工具")
![媒体保护工具演示](https://github.com/copyangle/SS_tools/blob/main/Workflow%20Example/decode_video.png "鸭子图媒体保护工具")


## 本地节点部署方法
- 部署：
  - cd `ComfyUI/custom_nodes/`
  - git clone git@github.com:copyangle/SS_tools.git
  - cd SS_tools
  - pip install -r requirements.txt

**组件概览**
- confyUI节点：
  - `duck_encode_node`（将图片/视频隐藏在卡通鸭子图中）
  - `duck_decode_node`（从鸭子图提取原始图片/视频）
- 可执行工具：
  - `duck_encoder.exe`（本地生成鸭子图，支持图片/视频）
  - `duck_decoder.exe`（从鸭子图解码载荷，支持密码）

**duck_encode_node**
- 作用：将图片或视频数据隐藏到卡通鸭子图中，可选密码与标题 
- 输入：
  - `images`（可选 `IMAGE`）：单帧或多帧图片
  - `audio`（可选 `AUDIO`）：音频输入，输入多帧时可选
  - `password`（`STRING`）：留空不加密；填写开启密码保护
  - `title`（`STRING`）：在鸭子图上绘制标题
  - `fps`（`INT`）：合成视频时的帧率（默认 16）
  - `compress`（`INT`）：LSB 位宽（2/6/8）影响容量与画质 `duck_payload_exporter.py:187`
- 输出：
  - `duck_image`（`IMAGE`）：包含隐写数据的鸭子图


**duck_decode_node**
- 作用：从鸭子图中提取原始图片或视频数据 
- 输入：
  - `image`（`IMAGE`）：鸭子图
  - `password`（`STRING`，可选）：若加密则需填写正确密码
- 输出：
  - `images`（`IMAGE`）：还原出的图片序列或单帧
  - `audio`（`AUDIO`）：当载荷为视频时可恢复音频
  - `file_path`（`STRING`）：磁盘上的还原文件路径
  - `fps`（`INT`）：当载荷为视频时的帧率

## 本地保护/提取工具

**duck_encoder.exe**
- 作用：在本地将媒体文件编码为鸭子图 
- 基本用法：
  - 查看帮助：`duck_encoder.exe --help`
  - 编码图片：`duck_encoder.exe 媒体文件.png --title 标题 --password 密码 --compress 2 --out duck_payload.png`
  - 编码视频：`duck_encoder.exe 媒体文件.mp4 --title 标题 --password 密码 --compress 2 --out duck_payload.png`
- 参数：
  - `media`：图片（png/jpg/jpeg/bmp/webp）或视频（mp4/avi/mov）
  - `--title`：在鸭子图上绘制标题 
  - `--password`：开启密码保护（流式异或）
  - `--compress`：2/6/8 三档，位宽越大容量越高但更影响图像
  - `--out`：输出文件名，默认 
- 说明：
  - 视频会先转为“二进制图片”再隐写，避免音频等信息丢失 

**duck_decoder.exe**
- 作用：从鸭子图解码出原始载荷（图片/视频/二进制）
- 基本用法：
  - 无密码：`duck_decoder.exe --duck duck_payload.png --out recovered.bin`
  - 有密码：`duck_decoder.exe --duck duck_payload.png --out recovered.mp4 --password 你的密码`
- 参数：
  - `--duck`：输入鸭子图路径
  - `--out`：输出文件路径（后缀会根据载荷类型自动匹配）
  - `--password`：若加密则需要提供


**开发参考**
- 节点注册：`__init__.py:1`

- 输出原图/视频到 `--out` 指定路径（留空则按扩展名自动命名）。

## 注意事项
- 不要用图片编辑软件重新保存鸭子图，避免尾部数据被截断。
- 如果生成时设置了密码，解码必须提供相同密码，否则会提示校验失败。


