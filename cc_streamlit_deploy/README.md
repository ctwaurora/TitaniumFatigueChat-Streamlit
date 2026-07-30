# TitaniumFatigueChat

**面向钛合金疲劳研究的证据约束型 AI Scientist**

> 本项目 **不是** 普通文献总结工具，而是面向钛合金疲劳研究，
> 通过文献证据抽取、缺失证据检测、研究空白验证和可行性判断，
> 生成 **可追溯、可验证、可推翻** 的科学假设生成系统。

---

## Task Alignment

本项目对齐 AI Scientist 类榜题要求：

| 榜题要求 | 本系统对应设计 |
|---------|---------------|
| 使用 AI 大模型（Qwen/千问） | ingest 阶段使用 Qwen 进行文献卡片抽取；discover 阶段使用 Qwen 进行变量-机制关系抽取 |
| 从文献输入到可验证假设输出 | ingest（文献卡片）→ discover（覆盖矩阵+研究空白）→ validate（质量门禁+假设生成+Hypothesis Card）→ export（比赛包） |
| 有文献挖掘与事实提取 | 13+ 结构化字段的文献卡片抽取；变量—性能—机制—证据关系表；9 维度 × 50+ 类别覆盖矩阵 |
| 有逻辑驱动假设生成 | 缺失证据检测驱动假设方向；质量门禁（13 项检查）确保条件完整性；Hypothesis Card 18 字段验证假设可操作性 |
| 有可行性论证 | A/B/C 可行性等级；最低成本验证路径 + 完整验证路径；成功判据 + 推翻条件 |
| 有多轮质量门槛 | 伪空白筛除（空泛表达拒绝）；8 条件检查（至少 2 篇文献支持）；13 项 quality_gate |
| 有基线对比 | 12 项指标 × 3 组（直接 Qwen / 摘要 Qwen / 本系统）系统对比 |
| 有科学假设与研究计划 | Hypothesis Card（每张推荐卡片）；Scientific Hypothesis Plan（05_scientific_hypothesis_plan.md）；明确 Problem Statement、Rationale、Methods、Experiments 等 |

---

## 系统三层架构

```
┌──────────────────────────────────────────────────────┐
│  第三层：可验证假设推荐层                              │
│  假设生成 → 结构审查 → A/B/C 可行性 → Hypothesis Card │
├──────────────────────────────────────────────────────┤
│  第二层：研究空白质量验证层                            │
│  覆盖矩阵 → 缺失证据检测 → 伪空白筛除 → 历史回溯     │
├──────────────────────────────────────────────────────┤
│  第一层：钛合金疲劳知识库层                            │
│  PDF 文献 → 文献卡片 → 变量—性能—机制—证据表          │
└──────────────────────────────────────────────────────┘
```

### 第一层：钛合金疲劳知识库层

- PDF 文献（papers/, early_papers/, followup_papers/）
- 文献卡片（13+ 结构化字段：材料、工艺、组织、载荷、机制、证据等）
- 变量—疲劳性能—机制—证据关系表

### 第二层：研究空白质量验证层

- 覆盖矩阵（9 大维度 × 50+ 类别）
- 缺失证据检测
- 伪空白筛除（拒绝空泛、不可操作、超出范围等候选空白）
- 历史回溯验证

### 第三层：可验证假设推荐层

- 假设生成（缺失证据驱动）
- 结构审查（12 项检查）
- A/B/C 可行性判断
- 最低成本验证路径
- Hypothesis Card（18 字段）+ Scientific Hypothesis Plan（15 节）

---

## 输入

将 PDF 文献放入以下目录：

```
papers/           # 现代文献（核心分析）
early_papers/     # 早期文献/综述（历史回溯用）
followup_papers/  # 跟进文献（验证早期问题是否解决）
```

系统从 PDF 中提取结构化信息并使用 Qwen 进行知识抽取。

---

## 输出

| 文件 | 定位 | 回答的问题 |
|------|------|-----------|
| `outputs/01_evidence_map.md` | 文献证据地图 | 当前文献库中已有证据是什么 |
| `outputs/02_gap_diagnosis.md` | 研究空白诊断 | 哪些证据缺失，哪些空白可能值得做 |
| `outputs/03_hypothesis_summary.md` | 科学假设摘要 | 系统最终生成了什么科学假设 |
| `outputs/04_baseline_comparison.md` | 基线对比 | 为什么不是普通大模型直接生成 |
| `outputs/05_scientific_hypothesis_plan.md` | 研究计划 | 假设如何验证、如何推翻、需要什么数据 |
| `outputs/06_competition_readiness.md` | 完成度自检 | 对榜题要求满足到什么程度 |
| `outputs/auto_pipeline_report.md` | 自动采集报告 | OpenAlex 开放文献采集结果 |
| `data_preview/` | 结构化数据预览 | 文献卡片和变量-机制关系 |

---

## 运行方式

```bash
# 本地已有文献，一键完整演示
python app.py demo

# 自动采集开放获取文献 + 完整 Pipeline
python app.py auto
```

### 全部命令

```bash
# 1. 构建文献库
python app.py ingest

# 2. 发现研究空白
python app.py discover

# 3. 验证假设并生成科学假设
python app.py validate

# 4. 一键运行完整流程 + 导出比赛提交包
python app.py demo

# 5. 自动采集开放文献 + 运行完整 Pipeline
python app.py auto

# 6. 环境检查
python app.py check

# 7. 帮助
python app.py --help
```

### 命令对比

| 命令 | 网络 | Qwen API | 用途 |
|------|------|----------|------|
| `ingest` | 否 | 是 | 构建文献库 |
| `discover` | 否 | 是 | 发现研究空白 |
| `validate` | 否 | 是 | 验证假设生成科学假设 |
| `demo` | 否 | 是 | 本地完整流程 + 导出比赛提交包 |
| `auto` | 是（OpenAlex） | 是 | 自动采集 OA 文献 + 运行完整流程 |
| `check` | 否 | 否 | 环境依赖检查 |

### 设置

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置阿里云 Qwen API Key
echo "sk-你的QwenKey" > qwen_key.txt

# 3. 将 PDF 文献放入 papers/ 目录

# 4. 一键运行完整流程
python app.py demo
```

---

## 为什么普通大模型不够

| 维度 | 直接 Qwen | TitaniumFatigueChat |
|------|-----------|---------------------|
| 文献追溯 | 无具体文献，或编造文献 | 每条证据可追溯到文献卡片 |
| 缺失证据 | 不标注缺失 | 必须标注，空则不能通过质量门禁 |
| 机制链 | 泛泛描述 | 变量—性能—机制链显式建模 |
| 验证路径 | 不提供 | 最低成本 + 完整验证路径 |
| 推翻条件 | 无 | 必填推翻条件 |
| 空话控制 | 常有"进一步研究" | 质量门禁拒绝空话 |
| 可复现性 | 每次回答不同 | 结构化流程，可复现 |

---

## 系统如何避免普通大模型空泛生成

1. **证据约束** — 每个科学假设必须有具体文献支持（至少 2 篇）
2. **缺失证据必填** — 空则不能通过 quality_gate，无法进入 Hypothesis Card
3. **机制链显式建模** — 变量 → 局部效应 → 损伤行为 → 疲劳指标，箭头格式约束
4. **质量门禁** — 13 项检查过滤空泛表达和伪空白
5. **推翻条件必填** — 每个假设必须明确什么结果会推翻它

---

## 引用校验

系统内置 Reference Verifier：
1. 检查每条参考文献是否在 `literature_database.csv` 中真实存在
2. 未通过验证的引用被移入 "unverified references"，不进入正式输出
3. 校验报告保存在 `outputs/reference_verification_report.md`

---

## 项目结构

```
TitaniumFatigueChat/
├─ app.py                     # 主入口（7 个命令）
├─ src/                       # 核心模块
│  ├─ ingestion.py            # 文献库构建
│  ├─ discovery.py            # 研究空白发现
│  ├─ validator.py            # 验证与假设生成
│  ├─ auto_collector.py       # 自动开放文献采集
│  ├─ exporter.py             # 比赛提交包导出
│  ├─ validation.py           # 质量门禁
│  └─ paper_collector.py      # 预设文献收集
├─ skills/                    # 复用技能模块
├─ config/
│  └─ task_profile.yaml       # 榜题对齐配置
├─ data/                      # 数据文件
├─ outputs/                   # 核心输出
├─ competition_package/       # 比赛提交包
├─ legacy_scripts/            # 旧版脚本（保留）
├─ papers/                    # PDF 文献（用户放置）
├─ qwen_key.txt               # API Key
└─ requirements.txt           # 依赖
```

---

## 当前限制

1. **文献库规模有限** — 覆盖矩阵区分度随文献数量增加
2. **科学假设为初步性质** — 当前为 preliminary evidence-backed hypothesis
3. **LLM 质量依赖** — Qwen 输出质量直接影响卡片抽取
4. **验证停留在方案层面** — 不执行实际实验/仿真
5. **覆盖矩阵关键词匹配粗糙** — 可能漏匹配

---

## 当前竞赛提交包

比赛包文件位于 `competition_package/`：

```
competition_package/
├─ 01_evidence_map.md
├─ 02_gap_diagnosis.md
├─ 03_hypothesis_summary.md
├─ 04_baseline_comparison.md
├─ 05_scientific_hypothesis_plan.md
├─ 06_competition_readiness.md
├─ README_FOR_REVIEWERS.md
├─ run_command.txt
└─ data_preview/
```

运行 `python app.py demo` 后自动生成。
