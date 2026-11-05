# -*- coding: utf-8 -*-
"""切分产物层验证：条文按条拿状态，节标题不入库。"""
import os
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "输出文档", "scripts"))

import split_articles_v2  # noqa: E402

MD = """<!-- page 1 -->
1 总则
1.0.1 为规范图书馆建筑设计，制定本规范。
6 防火设计
6.1 一般规定
6.1.1 图书馆建筑防火设计应符合现行国家标准的规定。
6.1.2 图书馆建筑的耐火等级不应低于二级。
6.1.3 书库的防火分区面积应符合现行国家标准的规定。
6.2 安全疏散
6.2.1 图书馆建筑的安全出口不应少于两个。
6.2.2 图书馆阅览室的疏散门数量应符合现行国家标准的规定。
"""


def split_once():
    tmp = tempfile.mkdtemp(prefix="bcr_split_")
    md = os.path.join(tmp, "JGJ 38-2015 图书馆建筑设计规范.md")
    with open(md, "w", encoding="utf-8") as fh:
        fh.write(MD)
    chunks, _ = split_articles_v2.split_file(md, os.path.join(tmp, "missing.json"))
    return chunks


class Split(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = split_once()
        cls.by_no = {c["article_no"]: c for c in cls.chunks}

    def test_no_section_chunks(self):
        self.assertEqual([c for c in self.chunks if c["kind"] == "section"], [])

    def test_listed_article_is_partial_force(self):
        for no in ("6.1.2", "6.1.3", "6.2.1", "6.2.2"):
            self.assertEqual(self.by_no[no]["force_status"], "部分强条", no)

    def test_other_article_is_not_force(self):
        self.assertEqual(self.by_no["1.0.1"]["force_status"], "非强条")
        self.assertEqual(self.by_no["6.1.1"]["force_status"], "非强条")

    def test_article_text_still_intact(self):
        self.assertTrue(self.by_no["6.1.2"]["text"].endswith("二级。"))
        self.assertEqual(self.by_no["6.2.1"]["page_start"], 1)


if __name__ == "__main__":
    unittest.main()
