# CAD 标识指向识别

这是一个适合少量、偶发任务的视觉大模型工具，不需要训练数据。处理流程：

1. 将 PDF 每页渲染为高清图片。
2. 结合参考标识图，在整页中定位圆圈编号标识。
3. 按标识尺寸自动扩大裁剪局部区域。
4. 再次调用视觉模型，判断小三角是否有方向含义，并提取关联尺寸、文字或轮廓。
5. 展示标识框、局部裁剪和结构化 JSON 结果。

## 安装

建议使用 Python 3.11 或 3.12：

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 运行

复制配置模板并填写真实 API Key：

```powershell
Copy-Item backend_config.example.json backend_config.json
```

`backend_config.json` 格式：

```json
{
  "base_url": "https://api.ppio.com/openai",
  "api_key": "你的 API Key",
  "model": "moonshotai/kimi-k3"
}
```

然后启动：

```powershell
python -m streamlit run app.py
```

程序只从项目目录下的 `backend_config.json` 读取 API 地址和密钥，也允许服务器使用 `OPENAI_BASE_URL`、`OPENAI_API_KEY` 环境变量覆盖。配置文件已加入 `.gitignore`，不会被提交。API 地址和密钥不会显示在网页中，网页侧边栏只允许输入模型。

标识样式已经写入提示词，页面不再要求上传标识参考图，每次模型调用只发送一张图纸图片或局部裁剪图片。

默认模型可在页面中修改，也可通过 `OPENAI_MODEL` 环境变量设置：

```powershell
$env:OPENAI_MODEL="moonshotai/kimi-k3"
```

## 参数建议

- PDF 渲染 DPI：先用 350；小字看不清时升到 450~600。
- 裁剪范围：先用 12 倍；目标离标识较远时升到 16~20 倍。
- 最低置信度：初次使用设为 0.45，漏检时适当降低。
- 多页 PDF 会逐页处理；每个标识会产生一次局部识别调用。

## 已知限制

- 视觉模型返回的坐标并非像素级检测器结果，标识框可能有偏差。
- 如果所有小三角都固定在圆圈右下方，它可能只是符号样式，不一定表示方向；第二阶段会单独输出 `is_directional_arrow`。
- 图纸非常密集时，可以提高 DPI 或缩小裁剪范围后重试。
- 识别结果用于辅助审核，关键尺寸仍应由工程人员确认。
