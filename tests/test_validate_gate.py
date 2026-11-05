# -*- coding: utf-8 -*-
"""validate_chunks.py 的门禁：坏数据必须让进程以 1 退出，而不只是打印。"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALIDATE = os.path.join(ROOT, "输出文档", "scripts", "validate_chunks.py")

TEXT = "本条为测试用条文正文，长度足够通过超短门禁，句末带标点。"


def row(std, no, force, kind="article", text=TEXT):
    return {
        "standard": std, "standard_name": std, "version": "",
        "force_status": force, "status": "现行", "article_no": no,
        "chapter": no.split(".")[0], "page_start": 1, "page_end": 1,
        "kind": kind, "text": text,
    }


def good_rows():
    rows = [
        row("JGJ 38-2015", "1.0.1", "非强条"),
        row("JGJ 38-2015", "6.1.2", "部分强条"),
        row("JGJ 38-2015", "6.1.3", "部分强条"),
        row("JGJ 38-2015", "6.2.1", "部分强条"),
        row("JGJ 38-2015", "6.2.2", "部分强条"),
        row("GB 50352-2019", "3.1.1", "非强条"),
        row("GB 50352-2019", "4.3.1", "原强条已废止"),
        row("GB 50352-2019", "6.7.4", "原强条已废止"),
        row("GB 50352-2019", "6.8.6", "原强条已废止"),
        row("GB 50352-2019", "6.8.9", "原强条已废止"),
        row("GB 55031-2022", "3.1.1", "全文强制"),
    ]
    return rows


def run_validate(rows):
    tmp = tempfile.mkdtemp(prefix="bcr_validate_")
    with open(os.path.join(tmp, "chunks.jsonl"), "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    proc = subprocess.run(
        [sys.executable, VALIDATE, "--chunks", tmp],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        env=dict(os.environ, PYTHONIOENCODING="utf-8"), timeout=180,
    )
    return proc


class Gate(unittest.TestCase):
    def test_good_data_exits_zero(self):
        proc = run_validate(good_rows())
        self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
        self.assertIn("门禁通过", proc.stdout)

    def test_section_chunk_fails(self):
        rows = good_rows() + [row("JGJ 38-2015", "6.1", "非强条",
                                  kind="section", text="一般规定")]
        proc = run_validate(rows)
        self.assertEqual(proc.returncode, 1, proc.stdout)
        self.assertIn("kind=section", proc.stdout)

    def test_wrong_force_status_fails(self):
        rows = good_rows()
        rows[0]["force_status"] = "部分强条"          # 1.0.1 不是强条
        proc = run_validate(rows)
        self.assertEqual(proc.returncode, 1, proc.stdout)
        self.assertIn("force_status", proc.stdout)

    def test_empty_text_fails(self):
        rows = good_rows() + [row("GB 55031-2022", "4.0.1", "全文强制", text="   ")]
        proc = run_validate(rows)
        self.assertEqual(proc.returncode, 1, proc.stdout)
        self.assertIn("空文本", proc.stdout)

    def test_short_article_fails(self):
        rows = good_rows() + [row("GB 55031-2022", "4.0.1", "全文强制", text="过短")]
        proc = run_validate(rows)
        self.assertEqual(proc.returncode, 1, proc.stdout)
        self.assertIn("超短", proc.stdout)

    def test_missing_listed_article_fails(self):
        rows = [r for r in good_rows() if r["article_no"] != "6.2.2"]
        proc = run_validate(rows)
        self.assertEqual(proc.returncode, 1, proc.stdout)
        self.assertIn("6.2.2", proc.stdout)


if __name__ == "__main__":
    unittest.main()
