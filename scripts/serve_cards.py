"""带 HTTP Range 支持的本地静态服务器，供解释卡页面播放、拖动附件4视频。

python -m http.server 不支持 Range 请求，浏览器无法从视频中间取字节，
点击词语/证据跳转会失效或从头重放。本脚本只解决这一点，其余行为同 http.server。

在工程根目录运行：
    python -m scripts.serve_cards
浏览器打开：
    http://127.0.0.1:8765/outputs/predictions_q3_mixed_neutral/explanation_cards.html
"""

from __future__ import annotations

import argparse
import os
import re
import threading
import webbrowser
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RANGE_RE = re.compile(r"bytes=(\d*)-(\d*)\s*$")


class _LimitedReader:
    """只读取剩余字节的文件包装，供 copyfile 逐块拷贝。"""

    def __init__(self, handle, remaining: int):
        self.handle = handle
        self.remaining = remaining

    def read(self, size: int = -1) -> bytes:
        if self.remaining <= 0:
            return b""
        if size is None or size < 0:
            size = self.remaining
        chunk = self.handle.read(min(size, self.remaining))
        self.remaining -= len(chunk)
        return chunk

    def close(self) -> None:
        self.handle.close()


class RangeRequestHandler(SimpleHTTPRequestHandler):
    """静态文件服务，支持单区间 Range 请求（206），视频才能拖动定位。"""

    def end_headers(self) -> None:
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        try:
            handle = open(path, "rb")
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "File not found")
            return None
        stat = os.fstat(handle.fileno())
        size = stat.st_size
        start, end = 0, size - 1
        header = self.headers.get("Range")
        if header:
            match = RANGE_RE.match(header.strip())
            if not match or (not match.group(1) and not match.group(2)):
                handle.close()
                self.send_error(HTTPStatus.BAD_REQUEST, "Malformed Range")
                return None
            first, last = match.group(1), match.group(2)
            if first:
                start = int(first)
                end = int(last) if last else size - 1
            else:  # bytes=-suffix：取末尾若干字节（浏览器探测尾部 moov 时使用）
                start = max(0, size - int(last))
            end = min(end, size - 1)
            if start > end or start >= size:
                handle.close()
                self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return None
            self.send_response(HTTPStatus.PARTIAL_CONTENT)
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        else:
            self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", self.guess_type(path))
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(int(stat.st_mtime)))
        self.end_headers()
        handle.seek(start)
        return _LimitedReader(handle, end - start + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-open", action="store_true", help="只启动服务，不自动打开浏览器")
    parser.add_argument("--rebuild", action="store_true", help="启动前从最新预测和证据表重建页面")
    args = parser.parse_args()
    page_file = ROOT / "outputs/predictions_q3_mixed_neutral/explanation_cards.html"
    if args.rebuild or not page_file.is_file():
        from scripts.build_q3_cards import main as build_cards
        build_cards()
    handler = partial(RangeRequestHandler, directory=str(ROOT))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    page = f"http://{args.host}:{args.port}/outputs/predictions_q3_mixed_neutral/explanation_cards.html"
    print(f"serving {ROOT}")
    print(f"open {page}")
    if not args.no_open:
        threading.Timer(0.6, webbrowser.open, args=(page,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
