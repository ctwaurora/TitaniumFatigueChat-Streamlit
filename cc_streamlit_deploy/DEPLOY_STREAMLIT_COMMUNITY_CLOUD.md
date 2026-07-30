# TitaniumFatigueChat Streamlit Community Cloud 部署说明

## 1. 部署入口

- 仓库入口文件：`streamlit_app.py`
- CLI 入口保留为：`app.py`
- Python 依赖：`requirements.txt`
- Streamlit 配置：`.streamlit/config.toml`

部署时应上传由 `scripts/export_streamlit_deploy.py` 生成的干净副本，不要直接上传包含本地 PDF、索引和运行结果的工作目录。

## 2. 生成部署副本

在主项目根目录执行：

```powershell
python scripts/export_streamlit_deploy.py
```

默认输出到主项目同级目录 `cc_streamlit_deploy`。脚本采用白名单复制，会先安全清理旧的同名部署目录，再扫描禁止文件、疑似密钥、大文件和 Windows 绝对路径。扫描通过后生成 `DEPLOY_MANIFEST.txt`。

## 3. GitHub 上传范围

将 `cc_streamlit_deploy` 中的全部文件上传到用于部署的 GitHub 仓库。不要上传主项目中的 `.git` 历史、本地 PDF、`outputs`、运行索引、缓存、`.env` 或 `.streamlit/secrets.toml`。

部署副本不会自动执行 `git init`，GitHub 仓库创建、remote 配置、提交和推送均需用户手动完成。

## 4. Streamlit Secrets

在 Streamlit Community Cloud 应用的 Secrets 设置中配置：

```toml
DEEPSEEK_API_KEY = "your-deepseek-api-key"
APP_PASSWORD = "your-private-app-password"
```

可选配置：

```toml
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"
```

不要把实际值写入仓库。仓库中的 `.streamlit/secrets.example.toml` 只包含占位符。

## 5. Community Cloud 设置

1. 在 Streamlit Community Cloud 选择 GitHub 仓库与部署分支。
2. Main file path 设置为 `streamlit_app.py`。
3. 在应用 Secrets 中添加 `DEEPSEEK_API_KEY` 和 `APP_PASSWORD`。
4. 部署后先验证登录页，再验证文献浏览和需要 DeepSeek 的功能。

`APP_PASSWORD` 未配置时应用不会自动放行。`DEEPSEEK_API_KEY` 未配置时，文献浏览和本地规则功能仍可使用，需要模型的功能会显示缺少配置或使用既有回退路径。

## 6. 数据与运行限制

部署副本只包含必要的小型只读初始化 CSV。不会包含本地 PDF、RAG 索引、逐页精读结果、后台任务、日志或 `outputs`。这些内容缺失时页面应显示空状态，不会在页面刷新时自动重建完整索引。

如后续需要持久化用户上传或后台任务结果，应使用独立持久化存储。Streamlit Community Cloud 本地文件系统不适合作为长期数据存储。
