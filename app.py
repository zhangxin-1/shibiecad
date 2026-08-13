from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
from typing import Any, List, Literal, Optional, Tuple, Type, TypeVar

import fitz
import streamlit as st
from openai import OpenAI
from anthropic import Anthropic
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from pydantic import BaseModel, Field, ValidationError


T = TypeVar("T", bound=BaseModel)

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.log")
logger = logging.getLogger("cad_recognition")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

DEFAULT_API_KEY = ""
DEFAULT_BASE_URL = "https://api.ppio.com/openai"
DEFAULT_MODEL = "qwen/qwen3.8-max"
BACKEND_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend_config.json")


class Box(BaseModel):
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)
    x2: float = Field(ge=0, le=1)
    y2: float = Field(ge=0, le=1)


class Point(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class Marker(BaseModel):
    number: str
    bbox: Box
    triangle_tip: Optional[Point] = None
    arrow_direction: str
    confidence: float = Field(ge=0, le=1)


class MarkerPageResult(BaseModel):
    markers: List[Marker]


class DimensionItem(BaseModel):
    marker_number: str
    full_annotation: str
    dimension_type: str
    meaning: str
    confidence_level: Literal["high", "medium", "low"]
    confidence_reason: str


SYSTEM_PROMPT = """你是一名专业机械工程图智能解析助手，擅长识别机械制图中的尺寸标识关联关系。

你的任务不是单纯OCR，而是根据上传的工程图/CAD图纸、编号标识以及文字描述，识别每个编号标识实际指向的尺寸或技术要求，并按照真实机械加工、检验和工程图纸语义解释其含义。

====================
一、核心任务
====================

对于图片中每一个实际可见的编号标识：

1. 找到编号标识本身；
2. 所有标识的小三角尖角方向固定为右下方；
3. 以圆圈中心为原点，只在图像第四象限（x增大、y增大，即右下区域）寻找对应的尺寸文字、技术要求或公差框；
4. 必须识别完整内容；
5. 判断该标注属于什么机械制图类型；
6. 用符合真实机械加工/检验场景的语言解释它的工程含义。

====================
二、尺寸文字识别规则
====================

必须尽量保留原图中的完整工程标注，包括但不限于：

φ、Φ、Ø、R、M、±、°、×、Ra、基准 A/B/C、上下偏差、尺寸公差、配合公差、螺纹公差带、几何公差符号、表面粗糙度、倒角、锥度、斜度、数量、节距、深度、沉孔/锪孔等符号。

其中 Φ、Ø、φ 统一规范输出为 φ。

如果原图为 φ94 +0.035/0，应尽可能完整识别为 φ94 +0.035/0，并解释允许实际尺寸范围。

如果图片清晰度不足，无法可靠识别某一位数字或符号，不得猜测，应标记"局部字符无法可靠确认"。

====================
三、必须区分不同工程含义
====================

根据机械制图语义正确分类：

1. 直径：φ94 → 表示该处圆柱形特征的公称直径为94 mm
2. 半径：R5 → 表示该圆弧或圆角的半径为5 mm
3. 普通线性尺寸：350 → 表示所对应两尺寸界线之间的距离为350 mm
4. 倒角：1.5×45° → 表示该处加工1.5 mm × 45°倒角
5. 角度：45° → 表示对应两几何要素之间的夹角为45°
6. 公制螺纹：M50×3-6H → 公称直径50 mm、螺距3 mm、公差带6H
7. 尺寸公差：50±0.02 → 允许实际尺寸范围为49.98～50.02 mm
8. 表面粗糙度：Ra 3.2 → 该加工表面的表面粗糙度要求为 Ra 3.2 μm
9. 几何公差：⊥ 0.05 A → 相对于基准A的垂直度公差为0.05 mm
10. 基准：A → 位于基准框中则解释为基准A

====================
四、编号与尺寸对应规则
====================

编号圆圈 → 读取圈内数字仅用于确定编号 → 找到圆圈中心和右下侧三角尖角 → 从圆圈外缘的三角尖端开始向右下延伸 → 只在圆圈中心的第四象限（x增大且y增大）寻找有效标注 → 判断引线或尺寸线关系 → 确认完整尺寸 → 再进行工程语义解释。

禁止仅根据"距离哪个数字最近"进行匹配。

本批图所有三角尖角都固定朝右下。禁止输出左下、左上、右上、左、右、上或下等其他方向；即使视觉上看似朝左下，也必须按右下方向重新判断。这里的“右下”不是“右侧或下侧”，而是目标位置必须同时满足 x目标>x圆心 且 y目标>y圆心。

同时结合尺寸线、尺寸界线、引出线、箭头、中心线、剖面线、零件轮廓判断真正对应的尺寸。

====================
五、防止错误匹配
====================

以下内容不得作为编号对应尺寸：
圆圈内部的全部像素和文字（尤其圈内编号）、标题栏日期、图号、页码、比例、公司名称、材料栏、与尖角方向相反的无关数字、附近但属于其他尺寸线的数字、无法证明与编号存在指向关系的文字。圈内编号只能作为 marker_number，永远不能作为 full_annotation，也不能参与目标值判断。

====================
六、真实工程场景约束
====================

1. 不虚构图纸中不存在的信息
2. 不擅自判断加工工艺或材料
3. 不擅自判断孔/轴，除非图纸结构明确
4. 不把粗糙度当尺寸
5. 不把形位公差当普通尺寸
6. 不把基准符号当普通字母
7. 不丢失尺寸公差
8. 不丢失直径、半径、螺纹、角度等符号
9. 不根据常识"补全"图片中看不清楚的数字
10. 默认单位mm，表面粗糙度Ra使用μm，角度使用°

====================
七、置信度
====================

高：编号指向明确，尺寸字符清晰，工程含义明确。
中：指向关系明确，但部分文字较小或存在轻微模糊。
低：存在两个以上可能目标，或者关键字符无法可靠确认。

置信度为"低"时，禁止强行给出确定答案，应明确说明不确定部分。

准确性优先于识别数量。宁可输出"无法确认"也绝对不能为了给出答案而猜一个尺寸。

最重要的判定原则是：三角尖角方向 > 引线方向 > 空间距离。目标不是寻找编号附近的文字，而是理解编号引线的实际指向关系。"""


LOCATE_PROMPT = """你是机械 CAD 工程图视觉定位器。输入图片是完整图纸页面。编号标识的样式已在下文中描述。
先只做定位，不识别编号对应的尺寸。找出页面中所有与参考样式一致的橙红色/红色圆圈编号（通常为1~17），并逐个定位圆圈边缘的小三角形尖角。

要求：
1. bbox 必须紧贴完整标识（包含圆圈与三角形），坐标为相对完整页面宽高的 0~1 数值。
2. triangle_tip 是三角形最尖端在完整页面中的归一化坐标；看不清则为 null。
3. arrow_direction 必须固定输出"右下"。坐标定义：图片左上角为原点，x向右增大，y向下增大；右下即圆圈中心的第四象限（x增大、y增大）。禁止输出"左下"或其他方向。
4. 不要把普通尺寸圆、剖视圆或粗糙度符号误认成该标识。
5. 若没有可靠结果，markers 返回空数组。不要臆造。
6. 仔细区分圈内编号，确保不混淆相近数字（如6和9、1和7）；不要把圈内数字当作尺寸。
7. 按编号逐个检查，不能因为找到部分编号就停止；同一编号重复出现时保留定位更可靠者。
"""


TARGET_PROMPT = """你是机械工程图尺寸标识关联分析专家。输入图片是围绕编号 {number} 裁剪出的高清局部图。编号标识样式已在提示词中描述。本次只能分析编号 {number}，不要同时推理其他编号。

请严格按照以下流程识别该编号标识指向的工程尺寸或技术要求：

1. 找到编号 {number} 的圆圈标识
2. 找到圆圈中心以及圆圈右下侧的小三角尖角；方向固定为右下，禁止判断为左下或其他方向
3. 以圆圈中心为原点，只允许从圆圈外缘的右下三角尖端出发，沿 x增大且y增大的方向向右下延伸；只在第四象限寻找对应的尺寸、形位公差或技术要求
4. 完整识别标注内容（保留φ、R、M、±、°、×、Ra等所有工程符号）
5. 判断标注的机械制图类型
6. 用机械加工/检验场景语言解释工程含义

关键规则：
- 判定优先级严格为：三角尖角方向 > 引线方向 > 空间距离
- 三角尖角方向固定为右下；这不是“优先”条件，而是硬性过滤条件。候选目标必须同时满足 x目标>x圆心 且 y目标>y圆心；只满足其中一个条件的目标一律排除
- 延伸起点必须是圆圈外缘的右下三角尖端，不得从圆圈内部、编号数字、圆心或圆圈左侧开始搜索
- 圆圈内部仅用于读取 marker_number={number}。必须忽略圈内所有内容，严禁把圈内数字 {number} 或其任何笔画识别为尺寸值或标注内容
- 搜索射线只能由标识向右下延伸，绝不能从右下候选反向穿过圆圈后继续到左上、左下或右上寻找文字
- 即使其他象限的文字更近、更清晰或存在引线，也不得选取；第四象限没有可靠目标时必须输出“无法确认”
- 在右下方向存在多个候选时，先计算候选到“三角尖端向右下延伸射线”的垂直距离，只保留射线命中或最贴近射线的候选；再选择沿射线前进距离最近、最先命中的值
- “射线上最近”高于普通二维距离。不得越过射线先命中的 1.5×45°，去选择更远或偏离射线的 M48 等文字
- 若提示中列出了相邻重叠标识已经占用的值，本编号不得重复选择这些值，应从自己射线上的剩余候选中选择
- confidence_reason 中必须将方向描述为"右下"，不得写"左下"、"左上"或"右上"
- 禁止选择距离编号最近、视觉上最大或位于尖角反方向的文字
- 目标可离编号较远；必须沿尖角和引线追踪到实际落点
- 绿色文字和绿色框可能是形位公差、表面粗糙度或技术要求，不得忽略
- 红色尺寸线仅作辅助，编号对应关系仍以三角尖角和引线实际指向为准
- 例如编号附近同时有 M60×2-6H、φ49，但尖角指向倒角时，应识别 1.5×45°，不得按距离选择螺纹或直径
- Φ/Ø 统一输出为 φ
- 必须保留完整公差信息（如 φ94 +0.035/0）
- 不要仅根据距离最近匹配，必须依据三角尖角方向、引线关系、尺寸线关系
- 如果尖角、引线落点或关键字符无法可靠确认，full_annotation 输出"无法确认"，dimension_type 输出"无法确定"，confidence_level 输出"low"，不要猜测或补全
- 排除标题栏、图号、页码、比例等非尺寸信息
- 如果存在多个候选目标，在confidence_reason中说明所有候选项

dimension_type 应优先使用以下类型：
直径尺寸、长度尺寸、半径、螺纹、倒角、角度、形位公差、其他、无法确定

confidence_level 判定：
- high：指向明确，字符清晰，含义明确
- medium：指向明确但文字较小或轻微模糊
- low：存在多个候选目标或关键字符无法确认
"""


FULL_PAGE_PROMPT = """分析这张机械工程图/CAD图纸中所有橙红色圆圈编号（通常为1~17）实际指向的尺寸、形位公差或技术要求。

输入图片是待分析的工程图纸页面。编号标识样式已在提示词中描述。

必须先定位所有编号，再严格按编号顺序逐个分析；完成一个编号后才分析下一个，不要同时推理所有编号。每个编号均执行：
1. 确认橙红色/红色圆圈及圈内编号，圈内数字不是尺寸
2. 找到圆圈右下侧三角尖角，方向固定为右下
3. 从圆圈中心第四象限（x增大、y增大）沿尖角方向和引线追踪最终落点
4. 完整识别落点处标注，保留φ、R、M、±、°、×、小数、上下偏差和基准字母
5. 判断类型；无法确认时明确输出"无法确认"，不得猜测

注意：
- 判定优先级：三角尖角方向 > 引线方向 > 空间距离
- 禁止按最近文字、最大数字或固定方向匹配，禁止选择尖角反方向的尺寸
- 编号可能距离目标较远；绿色文字/绿色框可能是形位公差、粗糙度或技术要求，不得忽略
- 红色尺寸线仅作辅助，以三角尖角和引线落点为准
- Φ/Ø统一输出为φ
- 必须保留完整公差（如 φ94 +0.035/0）
- 不要仅按距离匹配，依据尖角方向和引线关系
- 看不清或无法证明指向关系的标记"无法确认"，置信度为0
- 排除标题栏、图号、页码等非尺寸信息

只返回JSON数组，不要输出Markdown或解释文字。每项格式：
{"编号":"1","指向值":"◎0.05 A","类型":"形位公差","置信度":0.95}
类型使用：直径尺寸、长度尺寸、半径、螺纹、倒角、角度、形位公差、其他。无法确定时输出：{"编号":"X","指向值":"无法确认","类型":"其他","置信度":0}。"""


def load_backend_config() -> dict:
    """读取项目目录中的后端私有配置文件。"""
    if not os.path.exists(BACKEND_CONFIG_PATH):
        return {}
    try:
        with open(BACKEND_CONFIG_PATH, "r", encoding="utf-8") as config_file:
            data = json.load(config_file)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError, AttributeError):
        logger.exception("failed to load backend config path=%s", BACKEND_CONFIG_PATH)
        return {}


def openai_runtime_settings(config: dict) -> Tuple[str, str, str]:
    try:
        cloud_config = dict(st.secrets.get("backend", {}))
    except (FileNotFoundError, KeyError):
        cloud_config = {}
    base_url = (
        os.getenv("OPENAI_BASE_URL")
        or cloud_config.get("base_url")
        or config.get("base_url")
        or DEFAULT_BASE_URL
    )
    base_url = str(base_url).rstrip("/")

    api_key = (
        os.getenv("OPENAI_API_KEY")
        or cloud_config.get("api_key")
        or config.get("api_key")
        or DEFAULT_API_KEY
    )
    model = (
        os.getenv("OPENAI_MODEL")
        or cloud_config.get("model")
        or config.get("model")
        or DEFAULT_MODEL
    )
    return base_url, str(api_key), str(model)


def backend_protocol(base_url: str) -> str:
    return "anthropic" if base_url.rstrip("/").lower().endswith("/anthropic") else "openai"


def image_base64(image: Image.Image, image_format: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def resize_for_api(image: Image.Image, max_long_side: int = 2048) -> Image.Image:
    w, h = image.size
    if max(w, h) <= max_long_side:
        return image
    scale = max_long_side / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return image.resize((new_w, new_h), Image.LANCZOS)


def image_block(image: Image.Image) -> dict:
    image = resize_for_api(image)
    return {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64," + image_base64(image),
            "detail": "high",
        },
    }


def enhance_cad_lines(image: Image.Image, strength: float = 2.4) -> Image.Image:
    """Darken pale CAD strokes while keeping the white paper background white."""
    if strength <= 1.0:
        return image.convert("RGB")
    rgb = image.convert("RGB")
    lut = [max(0, min(255, round(255 - (255 - value) * strength))) for value in range(256)]
    enhanced = rgb.point(lut * 3)
    enhanced = ImageEnhance.Color(enhanced).enhance(1.15)
    return ImageEnhance.Sharpness(enhanced).enhance(1.2)


def render_pdf(pdf_bytes: bytes, dpi: int, color_strength: float = 2.4) -> List[Image.Image]:
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    pages = []
    try:
        for page in document:
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            rendered = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
            pages.append(enhance_cad_lines(rendered, color_strength))
    finally:
        document.close()
    return pages


def extract_text(completion: object) -> str:
    choices = getattr(completion, "choices", [])
    if not choices:
        return ""
    content = getattr(choices[0].message, "content", "")
    return content.strip() if isinstance(content, str) else ""


def _anthropic_messages(messages: list[dict]) -> tuple[Optional[str], list[dict]]:
    system_parts: list[str] = []
    converted: list[dict] = []
    for message in messages:
        role = message.get("role", "user")
        if role == "system":
            content = message.get("content", "")
            system_parts.append(content if isinstance(content, str) else str(content))
            continue
        content = message.get("content", "")
        if isinstance(content, list):
            blocks = []
            for item in content:
                if item.get("type") == "text":
                    blocks.append({"type": "text", "text": item.get("text", "")})
                elif item.get("type") == "image_url":
                    data_url = item.get("image_url", {}).get("url", "")
                    match = re.match(r"data:([^;]+);base64,(.+)", data_url, re.DOTALL)
                    if match:
                        blocks.append({"type": "image", "source": {
                            "type": "base64", "media_type": match.group(1), "data": match.group(2)
                        }})
            content = blocks
        converted.append({"role": role, "content": content})
    return ("\n\n".join(system_parts) or None), converted


def stream_completion(client: Any, kwargs: dict, label: str) -> str:
    """流式接收模型文本；原始返回写日志，页面只显示耗时与 token usage。"""
    parts: List[str] = []
    reasoning_parts: List[str] = []
    usage = None
    finish_reasons: List[str] = []
    started = time.monotonic()
    stream_kwargs = dict(kwargs)
    stream_kwargs["stream"] = True
    stream_kwargs["stream_options"] = {"include_usage": True}

    logger.info(
        "request started label=%s model=%s images=%s",
        label,
        kwargs.get("model"),
        sum(1 for message in kwargs.get("messages", []) for item in (message.get("content", []) if isinstance(message.get("content"), list) else []) if item.get("type") == "image_url"),
    )
    if isinstance(client, Anthropic):
        system, messages = _anthropic_messages(kwargs.get("messages", []))
        anthropic_kwargs = {
            "model": kwargs["model"], "max_tokens": kwargs.get("max_tokens", 8192),
            "temperature": kwargs.get("temperature", 0), "messages": messages,
        }
        if system:
            anthropic_kwargs["system"] = system
        with client.messages.stream(**anthropic_kwargs) as stream:
            for text in stream.text_stream:
                parts.append(text)
            final_message = stream.get_final_message()
            usage = getattr(final_message, "usage", None)
            finish_reason = getattr(final_message, "stop_reason", None)
            if finish_reason:
                finish_reasons.append(finish_reason)
    else:
        with client.chat.completions.create(**stream_kwargs) as stream:
            for chunk in stream:
                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage is not None:
                    usage = chunk_usage
                choices = getattr(chunk, "choices", [])
                if not choices:
                    continue
                delta = getattr(choices[0], "delta", None)
                finish_reason = getattr(choices[0], "finish_reason", None)
                if finish_reason and finish_reason not in finish_reasons:
                    finish_reasons.append(finish_reason)
                text = getattr(delta, "content", None) if delta is not None else None
                reasoning = getattr(delta, "reasoning_content", None) if delta is not None else None
                if isinstance(reasoning, str) and reasoning:
                    reasoning_parts.append(reasoning)
                if isinstance(text, str) and text:
                    parts.append(text)

    elapsed = time.monotonic() - started
    raw_text = "".join(parts).strip()
    prompt_tokens = getattr(usage, "prompt_tokens", getattr(usage, "input_tokens", None))
    completion_tokens = getattr(usage, "completion_tokens", getattr(usage, "output_tokens", None))
    total_tokens = getattr(usage, "total_tokens", None)
    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    logger.info(
        "request completed label=%s model=%s elapsed=%.1fs prompt_tokens=%s completion_tokens=%s total_tokens=%s finish_reason=%s content_chars=%d reasoning_chars=%d",
        label, kwargs.get("model"), elapsed, prompt_tokens, completion_tokens, total_tokens,
        finish_reasons or None, len(raw_text), sum(len(item) for item in reasoning_parts),
    )
    logger.info("model response content begin label=%s\n%s\nmodel response content end label=%s", label, raw_text, label)
    if reasoning_parts:
        reasoning_text = "".join(reasoning_parts)
        logger.info("model reasoning content begin label=%s\n%s\nmodel reasoning content end label=%s", label, reasoning_text, label)
    if total_tokens is None:
        st.caption(f"{label} · 耗时 {elapsed:.1f}s · Token：上游接口未返回 usage")
    else:
        st.caption(
            f"{label} · 耗时 {elapsed:.1f}s · 输入 {prompt_tokens} tokens · "
            f"输出 {completion_tokens} tokens · 合计 {total_tokens} tokens"
        )
    return raw_text


def extract_json_text(text: str) -> str:
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    raise ValueError("模型响应中没有找到 JSON 对象")


def repair_json(text: str) -> str:
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    repaired = text
    repaired = repaired.replace("“", "「")
    repaired = repaired.replace("”", "」")
    try:
        json.loads(repaired)
        return repaired
    except json.JSONDecodeError:
        pass

    result = []
    i = 0
    in_string = False
    escaped = False

    while i < len(repaired):
        ch = repaired[i]

        if escaped:
            result.append(ch)
            escaped = False
            i += 1
            continue

        if ch == '\\':
            result.append(ch)
            escaped = True
            i += 1
            continue

        if ch == '"':
            if not in_string:
                in_string = True
                result.append(ch)
            else:
                rest = repaired[i + 1:].lstrip()
                if not rest or rest[0] in (',', '}', ']', ':'):
                    in_string = False
                    result.append(ch)
                else:
                    result.append('\\"')
            i += 1
            continue

        result.append(ch)
        i += 1

    final = "".join(result)
    try:
        json.loads(final)
        return final
    except json.JSONDecodeError:
        return text


def create_client(api_key: str, base_url: str) -> Any:
    protocol = backend_protocol(base_url)
    logger.info("create client protocol=%s base_url=%s", protocol, base_url.rstrip("/"))
    if protocol == "anthropic":
        return Anthropic(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            timeout=180.0,
            max_retries=2,
        )
    return OpenAI(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        timeout=180.0,
        max_retries=2,
    )


def call_vision_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    images: List[Image.Image],
    result_type: Type[T],
    max_retries: int = 2,
) -> T:
    schema = json.dumps(result_type.model_json_schema(), ensure_ascii=False)
    full_prompt = (
        user_prompt
        + "\n\n严格只输出一个合法的 JSON 对象，不要输出 Markdown 或额外解释。"
        + "\n重要：JSON 字符串值内部不能包含未转义的双引号。如需引用文字请使用「」或''，不要使用英文双引号。"
        + "\n返回 JSON 必须符合以下 JSON Schema：\n"
        + schema
    )
    content = [{"type": "text", "text": full_prompt}]
    content.extend(image_block(img) for img in images)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    kwargs = {
        "model": model,
        "max_tokens": 8192,
        "temperature": 0,
        "messages": messages,
    }
    last_error = None
    for attempt in range(max_retries + 1):
        raw_text = stream_completion(client, kwargs, f"JSON识别（第 {attempt + 1} 次）")
        if not raw_text:
            logger.warning("empty model response model=%s attempt=%d", model, attempt + 1)
            if attempt < max_retries:
                st.warning(f"模型第 {attempt + 1} 次返回为空，正在自动重试…")
                continue
            raise RuntimeError("模型连续返回空内容。网关可能中断了流式响应，或思考内容耗尽了输出预算，请稍后重试。")
        try:
            json_str = extract_json_text(raw_text)
            json_str = repair_json(json_str)
            return result_type.model_validate_json(json_str)
        except (ValidationError, ValueError) as exc:
            last_error = (exc, raw_text)
            if attempt < max_retries:
                kwargs["messages"] = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [
                    {"role": "user", "content": content},
                    {"role": "assistant", "content": raw_text},
                    {"role": "user", "content": [{"type": "text", "text":
                        "你的 JSON 格式有误（字符串内含未转义引号或格式不合法）。请重新输出合法 JSON，字符串内用「」代替双引号。只输出 JSON，不要其他文字。"}]},
                ]
                continue

    exc, raw_text = last_error
    raise RuntimeError("模型返回内容不是预期 JSON：%s\n\n原始响应：\n%s" % (exc, raw_text))


def call_vision_text(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    images: List[Image.Image],
) -> str:
    content = [{"type": "text", "text": user_prompt}]
    content.extend(image_block(img) for img in images)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content})
    kwargs = {
        "model": model,
        "max_tokens": 8192,
        "temperature": 0,
        "messages": messages,
    }
    raw_text = stream_completion(client, kwargs, "整页识别")
    if not raw_text:
        raise RuntimeError("模型没有返回文本内容，请确认网关和模型支持图片输入")
    return raw_text


def locate_markers(
    client: OpenAI,
    model: str,
    page: Image.Image,
) -> MarkerPageResult:
    return call_vision_json(client, model, "", LOCATE_PROMPT, [page], MarkerPageResult)


def normalized_box_to_pixels(box: Box, size: Tuple[int, int]) -> Tuple[int, int, int, int]:
    width, height = size
    x1 = int(min(box.x1, box.x2) * width)
    y1 = int(min(box.y1, box.y2) * height)
    x2 = int(max(box.x1, box.x2) * width)
    y2 = int(max(box.y1, box.y2) * height)
    return x1, y1, x2, y2


def crop_around_marker(
    page: Image.Image,
    marker: Marker,
    multiplier: float,
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    width, height = page.size
    x1, y1, x2, y2 = normalized_box_to_pixels(marker.bbox, page.size)
    marker_w = max(x2 - x1, 20)
    marker_h = max(y2 - y1, 20)
    # Keep the marker visible near the crop's upper-left and spend most of the
    # crop area on its required fourth quadrant (right/down). This prevents
    # unrelated values in the other three quadrants from dominating OCR.
    context_w = marker_w * 1.0
    context_h = marker_h * 1.0
    reach_w = marker_w * max(multiplier - 1.0, 1.0)
    reach_h = marker_h * max(multiplier - 1.0, 1.0)
    crop_box = (
        max(0, int(x1 - context_w)),
        max(0, int(y1 - context_h)),
        min(width, int(x2 + reach_w)),
        min(height, int(y2 + reach_h)),
    )
    return page.crop(crop_box), crop_box


def identify_target(
    client: OpenAI,
    model: str,
    crop: Image.Image,
    number: str,
    occupied_annotations: Optional[List[str]] = None,
) -> DimensionItem:
    prompt = TARGET_PROMPT.format(number=number)
    if occupied_annotations:
        prompt += (
            "\n\n相邻或重叠标识已确认并占用以下值："
            + "、".join(occupied_annotations)
            + "。除非图中明确存在另一处完全相同的独立标注，否则本编号不得重复选择这些值；"
              "请在本编号右下射线上选择最近的剩余候选。"
        )
    return call_vision_json(client, model, SYSTEM_PROMPT, prompt, [crop], DimensionItem)


def marker_boxes_are_near(a: Marker, b: Marker) -> bool:
    """Whether two markers are close enough to compete for the same annotations."""
    acx, acy = (a.bbox.x1 + a.bbox.x2) / 2, (a.bbox.y1 + a.bbox.y2) / 2
    bcx, bcy = (b.bbox.x1 + b.bbox.x2) / 2, (b.bbox.y1 + b.bbox.y2) / 2
    aw, ah = abs(a.bbox.x2 - a.bbox.x1), abs(a.bbox.y2 - a.bbox.y1)
    bw, bh = abs(b.bbox.x2 - b.bbox.x1), abs(b.bbox.y2 - b.bbox.y1)
    return abs(acx - bcx) <= max(aw, bw) * 1.8 and abs(acy - bcy) <= max(ah, bh) * 2.5


def full_page_analysis(
    client: OpenAI,
    model: str,
    page: Image.Image,
) -> str:
    return call_vision_text(client, model, SYSTEM_PROMPT, FULL_PAGE_PROMPT, [page])


def annotate_page(page: Image.Image, markers: List[Marker]) -> Image.Image:
    result = page.copy()
    draw = ImageDraw.Draw(result)
    font = ImageFont.load_default()
    line_width = max(3, page.width // 500)
    for marker in markers:
        box = normalized_box_to_pixels(marker.bbox, page.size)
        draw.rectangle(box, outline="blue", width=line_width)
        label = "#%s %.2f" % (marker.number, marker.confidence)
        draw.text((box[0], max(0, box[1] - 16)), label, fill="blue", font=font)
    return result


def circled_number(number: str) -> str:
    if number.isdigit() and 1 <= int(number) <= 20:
        return chr(0x2460 + int(number) - 1)
    return f"【{number}】"


def format_result_text(items: List[DimensionItem]) -> str:
    return "\n".join(
        f"{circled_number(item.marker_number)}{item.dimension_type}：{item.full_annotation}"
        for item in items
    )


def main() -> None:
    st.set_page_config(page_title="CAD 机械工程图尺寸识别", layout="wide")
    st.title("CAD 机械工程图尺寸识别")
    st.caption("基于视觉模型的两阶段识别：整页定位标识 → 自动裁剪局部 → 判断关联尺寸与工程含义")

    backend_config = load_backend_config()
    default_url, default_token, default_model = openai_runtime_settings(backend_config)

    with st.sidebar:
        st.subheader("模型配置")
        st.caption("API 地址和密钥由后端配置文件提供")
        model = st.text_input("视觉模型", value=default_model)
        st.divider()
        st.subheader("识别参数")
        mode = st.radio("识别模式", ["两阶段精确识别", "整页直接分析"], index=0)
        dpi = st.slider("PDF 渲染 DPI", 200, 600, 350, 50)
        color_strength = st.slider(
            "PDF 线条增色强度",
            1.0,
            4.0,
            2.4,
            0.2,
            help="浅色 CAD 图层看不清时调高；1.0 表示保持 PDF 原色。",
        )
        crop_multiplier = st.slider("标识裁剪范围（标识尺寸倍数）", 6.0, 20.0, 12.0, 1.0)
        min_confidence = st.slider("最低标识置信度", 0.0, 1.0, 0.45, 0.05)

    col1, col2 = st.columns(2)
    with col1:
        pdf_file = st.file_uploader("上传 CAD 图纸（PDF）", type=["pdf"])
    with col2:
        image_file = st.file_uploader("或上传图片", type=["png", "jpg", "jpeg", "bmp", "tiff"])

    has_input = bool(pdf_file or image_file)
    backend_configured = bool(default_token and default_url)
    ready = bool(has_input and backend_configured and model)
    if not backend_configured:
        st.error("后端配置文件缺少 API Base URL 或 API Key，请检查项目目录中的 backend_config.json。")
    if not st.button("开始识别", type="primary", disabled=not ready):
        return

    client = create_client(default_token, default_url)

    if pdf_file:
        with st.spinner("正在渲染 PDF…"):
            pages = render_pdf(pdf_file.getvalue(), dpi, color_strength)
        source_name = pdf_file.name
    else:
        pages = [Image.open(image_file).convert("RGB")]
        source_name = image_file.name

    try:
        if mode == "整页直接分析":
            _run_full_page_mode(client, model, pages, source_name)
        else:
            _run_two_stage_mode(client, model, pages, source_name, min_confidence, crop_multiplier)
    except Exception as exc:
        st.exception(exc)


def _run_full_page_mode(
    client: OpenAI,
    model: str,
    pages: List[Image.Image],
    source_name: str,
) -> None:
    with st.status("正在分析…", expanded=True) as status:
        all_text = []
        for page_index, page in enumerate(pages, start=1):
            st.write("第 %d/%d 页：整页分析中" % (page_index, len(pages)))
            result_text = full_page_analysis(client, model, page)
            all_text.append("=== 第 %d 页 ===\n\n%s" % (page_index, result_text))
        status.update(label="识别完成", state="complete")

    full_output = "\n\n".join(all_text)
    st.markdown("### 识别结果")
    st.markdown(full_output)
    st.download_button(
        "下载识别结果",
        full_output,
        file_name="cad_dimension_results.txt",
        mime="text/plain",
    )


def _run_two_stage_mode(
    client: OpenAI,
    model: str,
    pages: List[Image.Image],
    source_name: str,
    min_confidence: float,
    crop_multiplier: float,
) -> None:
    with st.status("正在识别…", expanded=True) as status:
        all_results = []
        all_items = []

        for page_index, page in enumerate(pages, start=1):
            st.write("第 %d/%d 页：定位标识" % (page_index, len(pages)))
            located = locate_markers(client, model, page)
            markers = [item for item in located.markers if item.confidence >= min_confidence]

            best_by_number: dict[str, Marker] = {}
            for m in markers:
                if m.number not in best_by_number or m.confidence > best_by_number[m.number].confidence:
                    best_by_number[m.number] = m
            markers = list(best_by_number.values())
            markers.sort(key=lambda m: (int(m.number) if m.number.isdigit() else 999, m.number))
            st.write("去重后检测到 %d 个不同编号标识" % len(markers))

            page_results = []
            resolved: list[tuple[Marker, DimensionItem]] = []
            # Resolve lower markers first. For vertically overlapping labels such
            # as 6/7, this lets the lower ray claim φ75 before 6 considers φ40.
            recognition_order = sorted(
                markers,
                key=lambda m: ((m.bbox.y1 + m.bbox.y2) / 2, (m.bbox.x1 + m.bbox.x2) / 2),
                reverse=True,
            )
            for marker_index, marker in enumerate(recognition_order, start=1):
                st.write(
                    "第 %d 页，标识 %s（%d/%d）：裁剪并识别"
                    % (page_index, marker.number, marker_index, len(markers))
                )
                crop, crop_box = crop_around_marker(page, marker, crop_multiplier)
                occupied = [
                    item.full_annotation
                    for other, item in resolved
                    if marker_boxes_are_near(marker, other)
                    and item.full_annotation not in ("", "无法确认")
                ]
                target = identify_target(client, model, crop, marker.number, occupied)
                resolved.append((marker, target))
                page_results.append(
                    {
                        "marker": marker.model_dump(),
                        "crop_box_pixels": list(crop_box),
                        "target": target.model_dump(),
                    }
                )
                all_items.append(target)

                title = "第 %d 页 · 标识 %s · %s" % (
                    page_index,
                    marker.number,
                    target.full_annotation or target.dimension_type,
                )
                with st.expander(title):
                    left, right = st.columns([1, 1])
                    left.image(crop, caption="自动裁剪的局部图")
                    info = (
                        f"**标注：** `{target.full_annotation}`\n\n"
                        f"**类型：** {target.dimension_type}\n\n"
                        f"**含义：** {target.meaning}\n\n"
                        f"**置信度：** {'高' if target.confidence_level == 'high' else '中' if target.confidence_level == 'medium' else '低'}\n\n"
                        f"**依据：** {target.confidence_reason}"
                    )
                    right.markdown(info)

            page_results.sort(key=lambda item: (int(item["marker"]["number"]) if item["marker"]["number"].isdigit() else 999))
            all_items.sort(key=lambda item: (int(item.marker_number) if item.marker_number.isdigit() else 999))
            all_results.append({"page": page_index, "items": page_results})
            st.image(annotate_page(page, markers), caption="第 %d 页标识定位结果" % page_index)

        status.update(label="识别完成", state="complete")

    output = json.dumps({"source": source_name, "pages": all_results}, ensure_ascii=False, indent=2)
    st.markdown("### 识别结果汇总")
    st.text(format_result_text(all_items))

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "下载 JSON 结果",
            output,
            file_name="cad_marker_results.json",
            mime="application/json",
        )
    with col2:
        text_output = format_result_text(all_items)
        st.download_button(
            "下载文本结果",
            text_output,
            file_name="cad_dimension_results.txt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
