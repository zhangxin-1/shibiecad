from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, List, Literal, Optional, Tuple, Type, TypeVar

import fitz
import streamlit as st
from anthropic import Anthropic
from openai import OpenAI
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from pydantic import BaseModel, Field, ValidationError


T = TypeVar("T", bound=BaseModel)
ApiClient = OpenAI | Anthropic

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.log")
logger = logging.getLogger("cad_recognition")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

DEFAULT_MODEL = "qwen/qwen3.8-max"
CONFIG_PATH = Path(__file__).resolve().with_name("backend_config.json")


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


class RecognizedText(BaseModel):
    text: str
    bbox: Box
    text_type: Literal["尺寸", "公差", "技术要求", "标题栏", "普通文字", "其他"]
    confidence: float = Field(ge=0, le=1)


class PageTextResult(BaseModel):
    texts: List[RecognizedText]


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
2. 判断编号圆圈、箭头、三角尖角或引线的实际指向方向；
3. 沿指向方向寻找它对应的尺寸文字、技术要求或公差框；
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

编号圆圈 → 找到圆圈上的三角尖角/箭头 → 判断尖角朝向 → 沿朝向寻找最近的有效标注 → 判断引线或尺寸线关系 → 确认完整尺寸 → 再进行工程语义解释。

禁止仅根据"距离哪个数字最近"进行匹配。

如果三角尖角朝右，则优先分析右侧目标；朝左则优先分析左侧目标；朝上/下则按照真实方向分析。

同时结合尺寸线、尺寸界线、引出线、箭头、中心线、剖面线、零件轮廓判断真正对应的尺寸。

====================
五、防止错误匹配
====================

以下内容不得作为编号对应尺寸：
圆圈内部的编号、标题栏日期、图号、页码、比例、公司名称、材料栏、与尖角方向相反的无关数字、附近但属于其他尺寸线的数字、无法证明与编号存在指向关系的文字。

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


LOCATE_PROMPT = """你是机械 CAD 工程图视觉定位器。输入图是完整图纸页面。
先只做定位，不识别编号对应的尺寸。编号标识的样式是橙红色或红色圆圈，圈内通常为 1~17 的数字，圆圈边缘带有一个小三角形尖角。找出页面中所有符合此描述的编号标识，并逐个定位小三角形尖角。

要求：
1. bbox 必须紧贴完整标识（包含圆圈与三角形），坐标为相对完整页面宽高的 0~1 数值。
2. triangle_tip 是三角形最尖端在完整页面中的归一化坐标；看不清则为 null。
3. 必须根据每个标识实际可见的三角尖角独立判断 arrow_direction，例如左、右、上、下、左上、右上、左下或右下；不得把方向固定为某一方向。
4. 不要把普通尺寸圆、剖视圆或粗糙度符号误认成该标识。
5. 若没有可靠结果，markers 返回空数组。不要臆造。
6. 仔细区分圈内编号，确保不混淆相近数字（如6和9、1和7）；不要把圈内数字当作尺寸。
7. 按编号逐个检查，不能因为找到部分编号就停止；同一编号重复出现时保留定位更可靠者。
"""


PAGE_TEXT_PROMPT = """你是机械 CAD 工程图 OCR 定位器。识别输入整页图纸中所有能够可靠辨认的字符和完整文字组，并返回它们的位置。

要求：
1. 尽量识别所有尺寸、公差、技术要求、标题栏和普通文字，包括 φ、R、M、±、°、×、小数、上下偏差、形位公差符号和基准字母。
2. 按具有完整工程含义的文字组输出，例如 M60×2-6H、1.5×45°、φ94 +0.035/0，不要无必要地拆成单个字符。
3. bbox 必须紧贴该文字组，坐标为相对整页宽高的 0~1 数值。
4. 橙红色圆圈内的编号由另一任务单独返回，本任务不要把它们加入 texts。
5. 看不清的内容不要猜测；只返回能够可靠辨认的字符。
6. 同一位置和内容不要重复返回。"""


TARGET_PROMPT = """你是机械工程图尺寸标识关联分析专家。输入图是围绕编号 {number} 裁剪出的高清局部图。编号标识是橙红色或红色圆圈，圈内是数字，圆圈边缘带小三角形尖角。本次只能分析编号 {number}，不要同时推理其他编号。

请严格按照以下流程识别该编号标识指向的工程尺寸或技术要求：

1. 找到编号 {number} 的圆圈标识
2. 找到圆圈边缘的小三角形尖角，判断尖角实际朝向及引线方向
3. 从三角尖端沿实际指向方向追踪引线和最终落点，寻找对应的尺寸、形位公差或技术要求
4. 完整识别标注内容（保留φ、R、M、±、°、×、Ra等所有工程符号）
5. 判断标注的机械制图类型
6. 用机械加工/检验场景语言解释工程含义

关键规则：
- 判定优先级严格为：三角尖角方向 > 引线方向 > 空间距离
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

输入图是待分析的工程图纸页面。编号标识是橙红色或红色圆圈，圈内通常为 1~17 的数字，圆圈边缘带小三角形尖角。

必须先定位所有编号，再严格按编号顺序逐个分析；完成一个编号后才分析下一个，不要同时推理所有编号。每个编号均执行：
1. 确认橙红色/红色圆圈及圈内编号，圈内数字不是尺寸
2. 找到圆圈边缘三角尖角并独立判断实际朝向
3. 沿尖角方向和引线追踪最终落点
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


SINGLE_PROMPT = """你是一名专业机械工程图智能解析助手，擅长识别机械制图中的尺寸标识关联关系。

你的任务：

根据输入的机械工程图图片，识别图中橙红色圆圈编号标识（1~17），并找到每个编号对应的三角尖角箭头实际指向的尺寸、形位公差或技术要求。

注意：你的目标不是寻找编号附近的文字，而是理解工程图中的“编号引线指向关系”。

【识别流程】

对于每一个橙红色编号：

步骤1：定位橙红色圆圈编号。确认圆圈外观为橙红色/红色、圆圈内部为阿拉伯数字、数字范围通常为1~17。不要把圆圈中的数字作为尺寸，不要把编号附近最近的数字直接作为结果。

步骤2：寻找圆圈边缘的三角形尖角。三角尖角是该编号真正的指向标识。需要判断尖角朝向、引线方向和尖角最终落点位置。判断优先级：三角尖角方向 > 引线方向 > 空间距离。禁止选择距离编号最近的尺寸、视觉上最大的数字或反方向的尺寸。

步骤3：沿箭头方向寻找目标文字。目标可能包括：普通尺寸（φ40、φ75、φ94、350）、半径（R10）、螺纹（M50-6H、M60×2-6H）、倒角（1×45°、1.5×45°）、角度（45°）、形位公差（◎0.05 A、⊥0.05 A、⌖0.05 A）。

必须完整保留 φ、R、M、±公差、°、×、小数和基准字母。

【机械工程图特殊规则】

1. 编号可能距离目标尺寸较远。例如编号3附近可能存在 M60×2-6H、φ49，但是如果三角尖角指向倒角，则目标应为 1.5×45°，不能选择附近的螺纹尺寸。
2. 绿色文字和绿色框通常表示形位公差、表面粗糙度或技术要求，不要忽略绿色标注。
3. 红色尺寸线只是辅助信息，编号对应关系以三角尖角为准。

【输出要求】

只返回JSON格式：
[
  {
    "编号": "1",
    "指向值": "◎0.05 A",
    "类型": "形位公差",
    "置信度": 0.95
  }
]

类型包括：直径尺寸、长度尺寸、半径、螺纹、倒角、角度、形位公差、其他。

【禁止事项】

1. 禁止把编号数字当成尺寸。例如编号5不能直接输出5，应沿指向找到如R10。
2. 禁止根据距离匹配。例如编号3附近看到M60不能直接输出M60，应沿箭头方向找到1.5×45°。
3. 禁止虚构不存在的尺寸。
4. 禁止修改图片中的尺寸。
5. 禁止补全无法确认的信息。

【最终检查】

输出前必须逐项确认：是否找到橙红编号圆圈；是否找到三角尖角；是否沿尖角方向寻找；是否排除了附近干扰尺寸；是否完整输出尺寸符号。

如果无法确定，输出：
{
  "编号": "X",
  "指向值": "无法确认",
  "置信度": 0
}

不要猜测。

现在开始分析输入工程图。"""


def load_backend_config() -> Tuple[str, str, str]:
    """Streamlit Secrets 优先；未配置 Secrets 时读取本地 JSON。"""
    try:
        secrets_backend = st.secrets.get("backend")
    except (FileNotFoundError, KeyError):
        secrets_backend = None

    if secrets_backend:
        data = dict(secrets_backend)
        config_source = "Streamlit Secrets 的 [backend]"
    elif CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"backend_config.json 读取失败或不是合法 JSON：{exc}") from exc
        config_source = "backend_config.json"
    else:
        raise RuntimeError(
            "未找到后端配置。本地请创建 backend_config.json；Streamlit Cloud 请在 "
            "App settings → Secrets 中配置 [backend]。"
        )

    if not isinstance(data, dict):
        raise RuntimeError(f"{config_source} 必须是配置对象。")
    base_url = str(data.get("base_url", "")).strip().rstrip("/")
    api_key = str(data.get("api_key", "")).strip()
    model = str(data.get("model", DEFAULT_MODEL)).strip() or DEFAULT_MODEL
    if not base_url or not api_key:
        raise RuntimeError(f"{config_source} 必须填写非空的 base_url 和 api_key。")
    if api_key in {"请填写 API Key", "你的 API Key"}:
        raise RuntimeError(f"{config_source} 中的 api_key 仍是示例占位文字，请填写真实 API Key。")
    try:
        api_key.encode("ascii")
    except UnicodeEncodeError as exc:
        raise RuntimeError(
            f"{config_source} 中的 api_key 包含中文、全角引号或其他非 ASCII 字符，"
            "请填写真实的原始 API Key。"
        ) from exc
    return base_url, api_key, model


def image_base64(image: Image.Image, image_format: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def resize_for_api(image: Image.Image, max_long_side: int = 4096) -> Image.Image:
    w, h = image.size
    if max(w, h) <= max_long_side:
        return image
    scale = max_long_side / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return image.resize((new_w, new_h), Image.LANCZOS)


def image_block(image: Image.Image) -> dict:
    original_size = image.size
    image = resize_for_api(image)
    logger.info("prepare API image original_size=%sx%s sent_size=%sx%s", *original_size, *image.size)
    return {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64," + image_base64(image),
            "detail": "high",
        },
    }


def enhance_cad_lines(image: Image.Image, gamma: float = 1.8, saturation: float = 1.35) -> Image.Image:
    """压暗接近白色的彩色 CAD 线条，同时保持纯白背景和原有色相。"""
    lut = [round(255 * ((value / 255) ** gamma)) for value in range(256)]
    enhanced = image.point(lut * 3)
    return ImageEnhance.Color(enhanced).enhance(saturation)


def render_pdf(pdf_bytes: bytes, dpi: int, enhance_lines: bool = True) -> List[Image.Image]:
    document = fitz.open(stream=pdf_bytes, filetype="pdf")
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    pages = []
    try:
        for page in document:
            pixmap = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB, alpha=False, annots=True)
            image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
            if enhance_lines:
                image = enhance_cad_lines(image)
            logger.info("rendered PDF page dpi=%s size=%sx%s", dpi, *image.size)
            pages.append(image)
    finally:
        document.close()
    return pages


def extract_text(completion: object) -> str:
    choices = getattr(completion, "choices", [])
    if not choices:
        return ""
    content = getattr(choices[0].message, "content", "")
    return content.strip() if isinstance(content, str) else ""


def _anthropic_content(content: Any) -> Any:
    if not isinstance(content, list):
        return content
    converted = []
    for item in content:
        if item.get("type") != "image_url":
            converted.append(item)
            continue
        url = item.get("image_url", {}).get("url", "")
        match = re.fullmatch(r"data:([^;]+);base64,(.+)", url, flags=re.DOTALL)
        if not match:
            raise ValueError("Anthropic 图片必须是 base64 data URL。")
        converted.append({
            "type": "image",
            "source": {"type": "base64", "media_type": match.group(1), "data": match.group(2)},
        })
    return converted


def _anthropic_request(kwargs: dict) -> dict:
    system_parts = []
    messages = []
    for message in kwargs.get("messages", []):
        if message.get("role") == "system":
            content = message.get("content", "")
            system_parts.append(content if isinstance(content, str) else str(content))
        else:
            messages.append({"role": message["role"], "content": _anthropic_content(message.get("content", ""))})
    request = {
        "model": kwargs["model"],
        "max_tokens": kwargs.get("max_tokens", 8192),
        "temperature": kwargs.get("temperature", 0),
        "messages": messages,
    }
    if system_parts:
        request["system"] = "\n\n".join(system_parts)
    return request


def stream_completion(client: ApiClient, kwargs: dict, label: str) -> str:
    """流式接收模型文本；原始返回写日志，页面只显示耗时与 token usage。"""
    parts: List[str] = []
    reasoning_parts: List[str] = []
    usage = None
    finish_reasons: List[str] = []
    started = time.monotonic()
    logger.info(
        "request started label=%s protocol=%s model=%s images=%s",
        label,
        "anthropic" if isinstance(client, Anthropic) else "openai",
        kwargs.get("model"),
        sum(1 for message in kwargs.get("messages", []) for item in (message.get("content", []) if isinstance(message.get("content"), list) else []) if item.get("type") == "image_url"),
    )
    if isinstance(client, Anthropic):
        with client.messages.stream(**_anthropic_request(kwargs)) as stream:
            for text in stream.text_stream:
                parts.append(text)
            final_message = stream.get_final_message()
        usage = getattr(final_message, "usage", None)
        stop_reason = getattr(final_message, "stop_reason", None)
        if stop_reason:
            finish_reasons.append(stop_reason)
    else:
        stream_kwargs = dict(kwargs)
        stream_kwargs["stream"] = True
        stream_kwargs["stream_options"] = {"include_usage": True}
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
    prompt_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)
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


def create_client(api_key: str, base_url: str) -> ApiClient:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/anthropic"):
        logger.info("create client protocol=anthropic base_url=%s", base_url)
        return Anthropic(api_key=api_key, base_url=base_url, timeout=180.0, max_retries=2)
    logger.info("create client protocol=openai base_url=%s", base_url)
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=180.0,
        max_retries=2,
    )


def call_vision_json(
    client: ApiClient,
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
    client: ApiClient,
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
    client: ApiClient,
    model: str,
    page: Image.Image,
) -> MarkerPageResult:
    return call_vision_json(client, model, "", LOCATE_PROMPT, [page], MarkerPageResult)


def recognize_page_texts(client: ApiClient, model: str, page: Image.Image) -> PageTextResult:
    return call_vision_json(client, model, "", PAGE_TEXT_PROMPT, [page], PageTextResult)


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
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    half_w = marker_w * multiplier / 2
    half_h = marker_h * multiplier / 2
    crop_box = (
        max(0, int(cx - half_w)),
        max(0, int(cy - half_h)),
        min(width, int(cx + half_w)),
        min(height, int(cy + half_h)),
    )
    return page.crop(crop_box), crop_box


def identify_target(
    client: ApiClient,
    model: str,
    crop: Image.Image,
    marker: Marker,
    crop_box: Tuple[int, int, int, int],
    page_size: Tuple[int, int],
) -> DimensionItem:
    page_width, page_height = page_size
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    crop_width = max(crop_x2 - crop_x1, 1)
    crop_height = max(crop_y2 - crop_y1, 1)
    marker_x1, marker_y1, marker_x2, marker_y2 = normalized_box_to_pixels(marker.bbox, page_size)
    local_bbox = {
        "x1": round((marker_x1 - crop_x1) / crop_width, 4),
        "y1": round((marker_y1 - crop_y1) / crop_height, 4),
        "x2": round((marker_x2 - crop_x1) / crop_width, 4),
        "y2": round((marker_y2 - crop_y1) / crop_height, 4),
    }
    local_tip = None
    if marker.triangle_tip is not None:
        tip_x = marker.triangle_tip.x * page_width
        tip_y = marker.triangle_tip.y * page_height
        local_tip = {
            "x": round((tip_x - crop_x1) / crop_width, 4),
            "y": round((tip_y - crop_y1) / crop_height, 4),
        }
    prompt = TARGET_PROMPT.format(number=marker.number) + (
        "\n\n定位阶段提供的当前裁剪图归一化坐标如下："
        f"编号框 bbox={json.dumps(local_bbox, ensure_ascii=False)}，"
        f"三角尖端={json.dumps(local_tip, ensure_ascii=False)}，"
        f"方向={marker.arrow_direction}。"
        "必须以这些坐标锁定当前编号，不得把裁剪图中的其他编号当作当前编号。"
        "优先检查当前编号圆圈紧邻下方或尖角延长线上的标注。"
        "若某候选标注明显更靠近另一个编号圆圈，禁止将它归给当前编号。"
        "特别注意：1.5×45° 等倒角文字必须按其空间位置和引线归属，不能被下方其他编号拿走。"
    )
    return call_vision_json(client, model, SYSTEM_PROMPT, prompt, [crop], DimensionItem)


def full_page_analysis(
    client: ApiClient,
    model: str,
    page: Image.Image,
) -> str:
    return call_vision_text(client, model, "", SINGLE_PROMPT, [page])


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

    try:
        base_url, api_key, default_model = load_backend_config()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    with st.sidebar:
        st.subheader("模型配置")
        model = st.text_input("视觉模型", value=default_model)
        st.divider()
        st.subheader("识别参数")
        dpi = st.slider(
            "PDF 渲染 DPI",
            200,
            600,
            450,
            50,
            help="PDF 会以所选 DPI 渲染；发送模型时最长边最多保留 4096 像素。",
        )
        enhance_lines = st.checkbox(
            "增强浅色 CAD 线条",
            value=True,
            help="压暗浅绿色、浅红色和浅青色线条，保留白色背景，便于模型读取小字。",
        )
        crop_multiplier = st.slider("标识裁剪范围（标识尺寸倍数）", 6.0, 20.0, 12.0, 1.0)
        min_confidence = st.slider("最低标识置信度", 0.0, 1.0, 0.45, 0.05)

    col1, col2 = st.columns(2)
    with col1:
        pdf_file = st.file_uploader("上传 CAD 图纸（PDF）", type=["pdf"])
    with col2:
        image_file = st.file_uploader("或上传图片", type=["png", "jpg", "jpeg", "bmp", "tiff"])

    has_input = bool(pdf_file or image_file)
    ready = bool(has_input and model.strip())
    if not st.button("开始识别", type="primary", disabled=not ready):
        return

    client = create_client(api_key, base_url)
    model = model.strip()

    if pdf_file:
        with st.spinner("正在渲染 PDF…"):
            pages = render_pdf(pdf_file.getvalue(), dpi, enhance_lines)
        source_name = pdf_file.name
    else:
        pages = [Image.open(image_file).convert("RGB")]
        source_name = image_file.name

    try:
        _run_two_stage_mode(client, model, pages, source_name, min_confidence, crop_multiplier)
    except Exception as exc:
        st.exception(exc)


def _run_full_page_mode(
    client: ApiClient,
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
    client: ApiClient,
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

            st.write("第 %d/%d 页：识别全部字符及位置" % (page_index, len(pages)))
            recognized = recognize_page_texts(client, model, page)
            logger.info(
                "page recognition data page=%s markers=%s recognized_texts=%s",
                page_index,
                json.dumps([item.model_dump() for item in markers], ensure_ascii=False),
                json.dumps([item.model_dump() for item in recognized.texts], ensure_ascii=False),
            )

            best_by_number: dict[str, Marker] = {}
            for m in markers:
                if m.number not in best_by_number or m.confidence > best_by_number[m.number].confidence:
                    best_by_number[m.number] = m
            markers = list(best_by_number.values())
            markers.sort(key=lambda m: (int(m.number) if m.number.isdigit() else 999, m.number))
            st.write("去重后检测到 %d 个不同编号标识" % len(markers))

            page_results = []
            for marker_index, marker in enumerate(markers, start=1):
                st.write(
                    "第 %d 页，标识 %s（%d/%d）：裁剪并识别"
                    % (page_index, marker.number, marker_index, len(markers))
                )
                crop, crop_box = crop_around_marker(page, marker, crop_multiplier)
                target = identify_target(client, model, crop, marker, crop_box, page.size)
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

            all_results.append({
                "page": page_index,
                "image_size_pixels": {"width": page.width, "height": page.height},
                "markers": [item.model_dump() for item in markers],
                "recognized_texts": [item.model_dump() for item in recognized.texts],
                "items": page_results,
            })
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
