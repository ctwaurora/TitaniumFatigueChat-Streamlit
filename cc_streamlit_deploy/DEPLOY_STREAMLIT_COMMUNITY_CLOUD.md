# TitaniumFatigueChat Streamlit Community Cloud 部署

## 公开部署范围

只使用 `python scripts/export_streamlit_deploy.py` 生成的部署副本。公开 GitHub 允许代码、Prompt、schema、config、benchmark/evaluation 代码，以及 `document_id/title/doi/year` 元数据。

公开仓库严禁包含：PDF、全文派生 chunks、Evidence 全文片段、向量、完整 RAG 索引、运行输出、日志、密钥或本地绝对路径。安全扫描发现上述内容时，导出必须失败。

## Streamlit 配置

- Main file path: `streamlit_app.py`
- Python dependencies: `requirements.txt`
- Streamlit config: `.streamlit/config.toml`

Secrets 最低配置：

```toml
DEEPSEEK_API_KEY = "your-deepseek-api-key"
APP_PASSWORD = "your-private-app-password"
TFC_PRIVATE_RAG_BUNDLE_URL = "https://<private-host>/titanium-fatigue-rag-v1.zip"
TFC_PRIVATE_RAG_BUNDLE_SHA256 = "<64-character-sha256>"
# 私有端点需要认证时才配置：
TFC_PRIVATE_RAG_BEARER_TOKEN = "<private-token>"
```

可选：

```toml
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
```

不得把真实 Secrets 写入仓库。

## 私有 RAG 运行时

运行时只接受 HTTPS 私有工件并强制校验 SHA-256，随后解压到应用 checkout 之外的临时缓存。也可在受控服务器上用 `TFC_PRIVATE_RAG_ROOT` 指向已挂载的私有 bundle；该路径仅用于运行环境，不得写入公开配置。

私有工件缺失、哈希不符或结构无效时，科研回答 fail-closed，且不得调用外部模型生成结论。公开元数据绝不用于重建全文 RAG。

## 发布检查

1. 运行部署导出与安全扫描。
2. 确认部署副本不存在 `data/cloud_bundle`、`.parquet`、`.npy`、`.npz`、`.pdf`。
3. 只提交生成的公开部署副本。
4. 在 Streamlit Secrets 配置私有 RAG 与 DeepSeek。
5. 在线验证登录、私有 RAG 状态、双 flagship 四 Skill 及 outbound evidence manifest。
