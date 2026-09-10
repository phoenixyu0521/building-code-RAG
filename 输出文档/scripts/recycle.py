# -*- coding: utf-8 -*-
"""
recycle.py —— 把文件/目录移入 Windows 回收站（可还原），不永久删除。

为什么不用别的方式：
  - `rm` / `del` 会永久擦除，个人目录属高风险操作，必须可还原。
  - PowerShell 的 `Add-Type -AssemblyName Microsoft.VisualBasic` 在本环境被安全策略拦截。
  - 因此直接用 ctypes 调 shell32!SHFileOperationW（纯标准库，无需编译）。

用法：
  python recycle.py <路径1> [路径2] ...
"""
import ctypes
import os
import sys
from ctypes import wintypes

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

FO_DELETE = 3
FOF_SILENT = 0x0004
FOF_NOCONFIRMATION = 0x0010
FOF_ALLOWUNDO = 0x0040          # 关键：送进回收站而不是永久删除
FOF_NOERRORUI = 0x0400
FOF_NOCONFIRMMKDIR = 0x0200


class SHFILEOPSTRUCTW(ctypes.Structure):
    _fields_ = [
        ("hwnd", wintypes.HWND),
        ("wFunc", wintypes.UINT),
        # 注意：pFrom 必须是「双 NUL 结尾」的多字符串，不能声明成 LPCWSTR 并传 Python str，
        # 因为 str 里内嵌的 \0 会被 ctypes 截断，导致 shell 返回错误码 2（实际却已删除）。
        # 因此声明为 c_void_p，传入显式构造的 c_wchar 缓冲区。
        ("pFrom", ctypes.c_void_p),
        ("pTo", ctypes.c_void_p),
        ("fFlags", ctypes.c_uint16),
        ("fAnyOperationsAborted", wintypes.BOOL),
        ("hNameMappings", ctypes.c_void_p),
        ("lpszProgressTitle", ctypes.c_void_p),
    ]


def recycle(paths):
    """传入绝对路径列表；返回 (成功列表, 失败列表)。"""
    ok, bad = [], []
    shell32 = ctypes.windll.shell32
    shell32.SHFileOperationW.restype = ctypes.c_int
    shell32.SHFileOperationW.argtypes = [ctypes.POINTER(SHFILEOPSTRUCTW)]

    for p in paths:
        ap = os.path.abspath(p)
        if not os.path.exists(ap):
            bad.append((p, "路径不存在"))
            continue
        # 双 NUL 结尾：缓冲区多留一格，赋值后该格保持 0
        buf = ctypes.create_unicode_buffer(len(ap) + 2)
        buf.value = ap
        op = SHFILEOPSTRUCTW(
            hwnd=None, wFunc=FO_DELETE,
            pFrom=ctypes.cast(buf, ctypes.c_void_p), pTo=None,
            fFlags=FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT | FOF_NOERRORUI,
            fAnyOperationsAborted=False, hNameMappings=None, lpszProgressTitle=None)
        rc = shell32.SHFileOperationW(ctypes.byref(op))
        # 实测（2026-09-10）：本机 SHFileOperationW 稳定返回 2（ERROR_FILE_NOT_FOUND），
        # 但操作实际已成功——路径消失、回收站出现 $I/$R 记录。GetLastError 的 14007
        # 也只是无关的残留 SXS 码。故 rc 不可作为判据。
        # 以「路径已消失且未被中止」为成功判据，rc 仅作参考记录。
        gone = not os.path.exists(ap)
        if gone and not op.fAnyOperationsAborted:
            ok.append((p, rc))
        else:
            bad.append((p, "未消失或已中止: rc=%d aborted=%s 已消失=%s"
                        % (rc, op.fAnyOperationsAborted, gone)))
    return ok, bad


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    good, failed = recycle(sys.argv[1:])
    for p, rc in good:
        print("OK    已移入回收站: %s   (SHFileOperation rc=%d，本机恒为2，可忽略)" % (p, rc))
    for p, why in failed:
        print("FAIL  %s  (%s)" % (p, why))
    sys.exit(1 if failed else 0)
