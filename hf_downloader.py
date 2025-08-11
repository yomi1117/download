#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Hugging Face 模型/数据集下载器（带实时进度 & 断点续传）
- 支持 repo_type: model / dataset / space
- 支持只下载匹配的文件（allow-patterns），或忽略模式（ignore-patterns）
- 支持 revision（分支/标签/commit）与私有仓库 token
- 自动创建目录；已下载且大小一致会跳过；部分下载会续传
- 进度：按文件显示 + 总进度汇总（tqdm）
"""

import argparse
import fnmatch
import os
import sys
from pathlib import Path
from typing import Iterable, Optional

import requests
from tqdm import tqdm
from huggingface_hub import HfApi, hf_hub_url

from typing import Optional, NamedTuple, List

class FileInfo(NamedTuple):
    rfilename: str
    size: Optional[int]  # 可能为 None

def list_files_compat(api, repo_id: str, repo_type: str, revision: Optional[str]) -> List[FileInfo]:
    # 1) 新版优先：list_files_info
    try:
        files = api.list_files_info(repo_id=repo_id, repo_type=repo_type, revision=revision)
        return [FileInfo(f.rfilename, getattr(f, "size", None)) for f in files]
    except Exception:
        pass

    # 2) 兼容：通过 repo_info().siblings
    try:
        info = api.repo_info(repo_id=repo_id, repo_type=repo_type, revision=revision)
        siblings = getattr(info, "siblings", None) or []
        if siblings:
            return [FileInfo(s.rfilename, getattr(s, "size", None)) for s in siblings]
    except Exception:
        pass

    # 3) 最后回退：只有路径，没有大小
    names = api.list_repo_files(repo_id=repo_id, repo_type=repo_type, revision=revision)
    return [FileInfo(name, None) for name in names]



def _match_patterns(name: str, allow: Optional[list], ignore: Optional[list]) -> bool:
    """返回该文件是否需要下载（先 allow 后 ignore）。"""
    if allow:
        ok = any(fnmatch.fnmatch(name, pat) for pat in allow)
        if not ok:
            return False
    if ignore:
        if any(fnmatch.fnmatch(name, pat) for pat in ignore):
            return False
    return True


def _ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def _human_size(num: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024:
            return f"{num:.1f}{unit}"
        num /= 1024
    return f"{num:.1f}PB"


def download_repo(
    repo_id: str,
    repo_type: str = "model",        # "model" | "dataset" | "space"
    local_dir: str = "./downloads",
    revision: Optional[str] = None,  # e.g. "main" / "refs/pr/1" / commit sha
    token: Optional[str] = None,     # 私有仓库需要
    allow_patterns: Optional[list] = None,
    ignore_patterns: Optional[list] = None,
    timeout: int = 60,
    chunk_size: int = 1024 * 1024,   # 1MB
):
    """
    逐文件流式下载，实时进度 & 断点续传。
    """
    api = HfApi(token=token)
    info = api.repo_info(repo_id=repo_id, repo_type=repo_type, revision=revision)
    # files = api.list_files_info(repo_id=repo_id, repo_type=repo_type, revision=revision)
    files = list_files_compat(api, repo_id=repo_id, repo_type=repo_type, revision=revision)

    # 统计总大小（仅统计匹配规则的文件）
    selected = []
    total_bytes = 0
    for f in files:
        # f.rfilename: 路径；f.size: 可能为 None（小文件非 LFS 时服务端不返回 size）
        if not _match_patterns(f.rfilename, allow_patterns, ignore_patterns):
            continue
        selected.append(f)
        if f.size is not None:
            total_bytes += int(f.size)

    # 如果部分文件没有 size，之后会在下载时动态加入总进度
    base_dir = Path(local_dir) / repo_type / repo_id.replace("/", "__") / (revision or "main")
    base_dir.mkdir(parents=True, exist_ok=True)

    # 总进度条
    total_bar = tqdm(
        total=total_bytes if total_bytes > 0 else None,
        unit="B",
        unit_scale=True,
        desc=f"[TOTAL] {repo_type}:{repo_id}@{revision or 'main'}",
        leave=True,
    )

    session = requests.Session()
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    for f in selected:
        rel_path = f.rfilename
        dest = base_dir / rel_path
        _ensure_parent(dest)

        # 远端 URL
        url = hf_hub_url(repo_id=repo_id, filename=rel_path, repo_type=repo_type, revision=revision)

        # 远端大小（可能 None）
        remote_size = None if f.size is None else int(f.size)

        # 本地已存在大小（用于续传）
        exist_bytes = dest.stat().st_size if dest.exists() else 0

        # 若已完整下载，跳过并更新总进度
        if remote_size is not None and exist_bytes == remote_size:
            total_bar.update(remote_size)
            continue

        # 对于未知大小文件，先发 HEAD 获取 size（若服务端提供）
        if remote_size is None:
            try:
                head = session.head(url, headers=headers, timeout=timeout, allow_redirects=True)
                if head.ok and head.headers.get("Content-Length"):
                    try:
                        remote_size = int(head.headers["Content-Length"])
                    except Exception:
                        remote_size = None

                # 把未知大小加入 total
                if remote_size is not None:
                    total_bar.total = (total_bar.total or 0) + remote_size
                    total_bar.refresh()
            except Exception as e:
                tqdm.write(f"[WARN] Failed to get size for {rel_path}: {e}")
                remote_size = None

        # 断点续传 Header - 修复HTTP 416问题
        range_headers = headers.copy()
        if exist_bytes > 0 and remote_size is not None:
            # 确保请求范围不超过文件实际大小
            if exist_bytes < remote_size:
                range_headers["Range"] = f"bytes={exist_bytes}-{remote_size-1}"
            else:
                # 如果本地文件大小异常，重新下载
                exist_bytes = 0
                if dest.exists():
                    dest.unlink()  # 删除损坏的文件
        elif exist_bytes > 0 and remote_size is None:
            # 如果不知道远程大小，但本地有文件，重新下载
            exist_bytes = 0
            if dest.exists():
                dest.unlink()  # 删除可能损坏的文件

        # 开始流式下载 - 增加重试机制
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                with session.get(url, headers=range_headers, stream=True, timeout=timeout, allow_redirects=True) as r:
                    if r.status_code in (200, 206):
                        # 每个文件的进度条
                        file_total = None
                        if remote_size is not None:
                            file_total = remote_size - exist_bytes
                        file_bar = tqdm(
                            total=file_total,
                            unit="B",
                            unit_scale=True,
                            desc=rel_path,
                            leave=False,
                        )

                        mode = "ab" if exist_bytes > 0 else "wb"
                        with open(dest, mode) as out:
                            for chunk in r.iter_content(chunk_size=chunk_size):
                                if not chunk:
                                    continue
                                out.write(chunk)
                                file_bar.update(len(chunk))
                                total_bar.update(len(chunk))

                        file_bar.close()
                        break  # 成功下载，跳出重试循环
                    elif r.status_code == 416:  # Range Not Satisfiable
                        tqdm.write(f"[WARN] HTTP 416 for {rel_path}, retrying without range header...")
                        # 清除断点续传，重新下载
                        exist_bytes = 0
                        if dest.exists():
                            dest.unlink()
                        range_headers = headers.copy()  # 重置headers
                        retry_count += 1
                        continue
                    else:
                        # 其他HTTP错误
                        tqdm.write(f"[WARN] Failed {rel_path}: HTTP {r.status_code}")
                        retry_count += 1
                        if retry_count >= max_retries:
                            tqdm.write(f"[ERROR] Max retries exceeded for {rel_path}")
                            break
                        continue
                        
            except Exception as e:
                retry_count += 1
                tqdm.write(f"[WARN] Download error for {rel_path} (attempt {retry_count}/{max_retries}): {e}")
                if retry_count >= max_retries:
                    tqdm.write(f"[ERROR] Max retries exceeded for {rel_path}")
                    break
                continue

    total_bar.close()
    print(f"\n✅ Done. Saved to: {base_dir}  (repo files: {len(selected)})")
    return str(base_dir)


def parse_args():
    p = argparse.ArgumentParser(description="Download Hugging Face model/dataset with live progress.")
    p.add_argument("--repo-id", required=True, help="e.g., meta-llama/Llama-3.1-8B or HuggingFaceH4/ultrachat_200k")
    p.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    p.add_argument("--revision", default=None, help="branch/tag/commit (default: main)")
    p.add_argument("--local-dir", default="./downloads")
    p.add_argument("--token", default=os.getenv("HF_TOKEN"), help="HF token (env HF_TOKEN preferred)")
    p.add_argument("--allow", nargs="*", default=None, help='only download files matching patterns, e.g. "*.bin" "*.safetensors" "*.parquet"')
    p.add_argument("--ignore", nargs="*", default=None, help='ignore files matching patterns, e.g. "*/old/*" "*.md"')
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--chunk-size-mb", type=int, default=1, help="stream chunk size in MB")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    download_repo(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        local_dir=args.local_dir,
        revision=args.revision,
        token=args.token,
        allow_patterns=args.allow,
        ignore_patterns=args.ignore,
        timeout=args.timeout,
        chunk_size=args.chunk_size_mb * 1024 * 1024,
    )
