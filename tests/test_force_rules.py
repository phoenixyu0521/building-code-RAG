# -*- coding: utf-8 -*-
"""条文级强制状态规则（force_rules.py）的边界测试。

审查问题 1：整本标准的状态被原样写给每个 chunk。这组测试把「哪一条是强条」
钉死成可审计的表，防止以后有人图省事又把标准级状态套回去。
"""
import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "输出文档", "scripts")
sys.path.insert(0, SCRIPTS)

import force_rules  # noqa: E402


class FullForceStandards(unittest.TestCase):
    """全文强制规范的每一条（含附录、表格）都是全文强制。"""

    def test_every_article_is_full_force(self):
        for std in ("GB 55031-2022", "GB 55037-2022", "GB 55019-2021"):
            for no in ("1.0.1", "3.1.1", "4.0.1", "附录B"):
                self.assertEqual(force_rules.resolve(std, no), "全文强制",
                                 f"{std} {no} 应为全文强制")

    def test_appendix_table_chunk_is_full_force(self):
        self.assertEqual(force_rules.resolve("GB 55037-2022", "4.1.2-表"), "全文强制")


class JGJ38(unittest.TestCase):
    """JGJ 38 的强条只有公告第 907 号列出的 4 条。"""

    LISTED = ("6.1.2", "6.1.3", "6.2.1", "6.2.2")

    def test_listed_articles_are_partial_force(self):
        for no in self.LISTED:
            self.assertEqual(force_rules.resolve("JGJ 38-2015", no), "部分强条")

    def test_other_articles_are_not_force(self):
        for no in ("1.0.1", "1.0.2", "6.1.1", "6.2.3", "8.2.1"):
            self.assertEqual(force_rules.resolve("JGJ 38-2015", no), "非强条",
                             f"{no} 不在公告名单，必须是 非强条")

    def test_list_is_exactly_the_four(self):
        self.assertEqual(list(force_rules.listed_articles("JGJ 38-2015")),
                         list(self.LISTED))


class GB50352(unittest.TestCase):
    """GB 50352 的原强条只有 4 条，其余条文不得再挂「已废止」。"""

    LISTED = ("4.3.1", "6.7.4", "6.8.6", "6.8.9")

    def test_listed_articles_are_repealed(self):
        for no in self.LISTED:
            self.assertEqual(force_rules.resolve("GB 50352-2019", no), "原强条已废止")

    def test_other_articles_are_not_force(self):
        for no in ("3.1.1", "4.3.2", "6.7.5", "6.8.10"):
            self.assertEqual(force_rules.resolve("GB 50352-2019", no), "非强条")

    def test_list_matches_official_announcement(self):
        self.assertEqual(list(force_rules.listed_articles("GB 50352-2019")),
                         list(self.LISTED))


class TableChunksInheritArticleStatus(unittest.TestCase):
    """表格 chunk 的条文号形如 `6.1.2-表`，状态必须跟所属条文走。"""

    def test_table_suffix_is_stripped(self):
        self.assertEqual(force_rules.resolve("JGJ 38-2015", "6.1.2-表"), "部分强条")
        self.assertEqual(force_rules.resolve("JGJ 38-2015", "6.1.1-表"), "非强条")
        self.assertEqual(force_rules.resolve("GB 50352-2019", "4.3.1-表"), "原强条已废止")


class UnknownStandard(unittest.TestCase):
    """未登记的标准不能被静默当成非强条。"""

    def test_unknown_standard_returns_marker(self):
        self.assertEqual(force_rules.resolve("GB 99999-2099", "1.0.1"),
                         force_rules.UNKNOWN)

    def test_every_meta_standard_is_covered(self):
        import split_articles_v2
        missing = [c for c in split_articles_v2.META if c not in force_rules.STANDARDS]
        self.assertEqual(missing, [], f"这些标准没有强制状态规则：{missing}")

    def test_rules_carry_audit_basis(self):
        for std, rule in force_rules.STANDARDS.items():
            self.assertTrue(rule.get("basis"), f"{std} 缺少依据说明")
            self.assertTrue(rule.get("source"), f"{std} 缺少依据出处")
            if rule["mode"] == "list":
                self.assertTrue(rule.get("articles"), f"{std} 是清单模式但清单为空")
                self.assertTrue(rule.get("default"), f"{std} 缺少默认状态")


if __name__ == "__main__":
    unittest.main()
