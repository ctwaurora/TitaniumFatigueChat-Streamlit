import os
import re
import requests
from pathlib import Path
from typing import Any, Dict, Optional

from .pdf_skill import extract_text_from_pdf
from .dedup_skill import calculate_file_hash, is_duplicate, get_doi_from_metadata
from .card_skill import extract_literature_card
from .library_skill import save_card
from .rag_skill import chunk_text, save_chunks


def _safe_filename(text: str) -> str:
    """
    把 DOI 或标题转换成安全文件名。
    """
    text = text or "unknown"
    text = re.sub(r'[\\/:*?"<>|]', "_", text)
    text = text.replace(" ", "_")
    return text[:120]


def _get_open_pdf_url(result: Dict[str, Any]) -> str:
    """
    兼容不同搜索接口返回的开放 PDF 字段。
    """
    if not isinstance(result, dict):
        return ""

    # 你原来使用的字段
    if result.get("open_access_pdf_url"):
        return result.get("open_access_pdf_url", "")

    # Semantic Scholar 常见格式
    open_access_pdf = result.get("openAccessPdf")
    if isinstance(open_access_pdf, dict) and open_access_pdf.get("url"):
        return open_access_pdf.get("url", "")

    # 其他可能格式
    if result.get("pdf_url"):
        return result.get("pdf_url", "")

    if result.get("url") and str(result.get("url")).lower().endswith(".pdf"):
        return result.get("url", "")

    return ""


def _build_metadata_card(result: Dict[str, Any], doi: Optional[str]) -> Dict[str, Any]:
    """
    当没有开放 PDF 时，至少保存文献元数据。
    """
    return {
        "title": result.get("title", "Unknown"),
        "authors": result.get("authors", []),
        "publication_year": result.get("year", ""),
        "year": result.get("year", ""),
        "journal": result.get("journal", ""),
        "doi": doi or result.get("doi", ""),
        "abstract": result.get("abstract", ""),
        "keywords": result.get("keywords", []),
        "source_type": "metadata_only"
    }


def download_open_pdf(pdf_url: str, save_path: str) -> bool:
    """
    下载开放获取的 PDF 文件。

    Args:
        pdf_url: PDF 下载 URL
        save_path: 保存路径

    Returns:
        bool: 下载是否成功
    """
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            )
        }

        with requests.Session() as session:
            session.trust_env = False
            response = session.get(
                pdf_url,
                stream=True,
                timeout=30,
                headers=headers,
                allow_redirects=True
            )
            response.raise_for_status()

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            with open(save_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        # 简单检查文件是否为空
        if not os.path.exists(save_path) or os.path.getsize(save_path) == 0:
            print("PDF 下载失败：文件为空")
            return False

        return True

    except Exception as e:
        print(f"PDF 下载失败：{str(e)}")
        return False


def process_search_result(result: Dict[str, Any]) -> bool:
    """
    处理搜索结果：
    1. 检查 DOI 是否重复
    2. 如果有开放 PDF，则下载 PDF
    3. 提取 PDF 文本
    4. 抽取文献卡片
    5. 保存文献卡片
    6. 保存 RAG 文本块

    Args:
        result: 搜索结果元数据

    Returns:
        bool: 处理是否成功
    """
    if not isinstance(result, dict):
        print("搜索结果格式错误，必须是字典类型")
        return False

    # 检查 DOI 是否已经存在
    doi = get_doi_from_metadata(result)

    if doi and is_duplicate(doi=doi):
        print("该文献 DOI 已存在，已自动跳过重复入库")
        return False

    # 检查是否有开放 PDF
    pdf_url = _get_open_pdf_url(result)

    if not pdf_url:
        print("无开放 PDF 资源，仅保存元数据")

        metadata_card = _build_metadata_card(result, doi)

        try:
            save_card(metadata_card, "", "", result)
            print("元数据保存成功")
            return True
        except Exception as e:
            print(f"元数据保存失败：{str(e)}")
            return False

    # 构造临时 PDF 路径
    filename_source = doi or result.get("title", "unknown_paper")
    filename = _safe_filename(filename_source)
    temp_path = str(Path("temp") / f"{filename}.pdf")

    # 下载 PDF
    if not download_open_pdf(pdf_url, temp_path):
        return False

    try:
        # 计算文件哈希
        file_hash = calculate_file_hash(temp_path)

        # 检查文件是否重复
        if is_duplicate(file_path=temp_path):
            print("该 PDF 文件已存在，已自动跳过重复抽取")
            return False

        # 提取 PDF 文本
        pdf_text = extract_text_from_pdf(temp_path)

        if not pdf_text or not pdf_text.strip():
            print("PDF 文本提取失败或内容为空")
            return False

        # 抽取文献卡片
        card = extract_literature_card(pdf_text)

        # 补充搜索元数据，防止模型漏掉 DOI、期刊、年份
        if isinstance(card, dict):
            card.setdefault("doi", doi or result.get("doi", ""))
            card.setdefault("journal", result.get("journal", ""))
            card.setdefault("publication_year", result.get("year", ""))

        # 保存文献卡片
        save_card(card, file_hash, temp_path, result)

        # 生成并保存 RAG 文本块
        chunks = chunk_text(pdf_text, metadata=result)
        save_chunks(chunks, temp_path, file_hash, result)

        print("文献抽取成功")
        return True

    except Exception as e:
        print(f"文献处理失败：{str(e)}")
        return False