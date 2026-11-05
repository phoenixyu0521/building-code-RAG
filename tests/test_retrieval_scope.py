# -*- coding: utf-8 -*-
"""检索范围门禁：默认只查条文（kind=article），显式指定才放行表格/附录。

审查问题 6：111 条节标题空壳会被向量检索命中，且 force_only 过滤不过滤 kind。
"""
import os
import sys
import types
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# 该环境里 torch 的 DLL 在沙箱内不可载入；这两个依赖只用于运行期，
# 测 where 拼装时用轻量桩顶替即可（不涉及任何真实检索）。
if "chromadb" not in sys.modules:
    _chroma = types.ModuleType("chromadb")
    _chroma.PersistentClient = object
    sys.modules["chromadb"] = _chroma
if "sentence_transformers" not in sys.modules:
    _st = types.ModuleType("sentence_transformers")
    _st.SentenceTransformer = object
    sys.modules["sentence_transformers"] = _st

from app import api, retriever  # noqa: E402


class RetrieverDefaultKind(unittest.TestCase):
    def test_no_where_gets_article_only(self):
        self.assertEqual(retriever._with_default_kind(None), {"kind": "article"})
        self.assertEqual(retriever._with_default_kind({}), {"kind": "article"})

    def test_existing_condition_is_anded_with_article(self):
        where = {"standard": "JGJ 38-2015"}
        self.assertEqual(retriever._with_default_kind(where),
                         {"$and": [{"standard": "JGJ 38-2015"}, {"kind": "article"}]})

    def test_explicit_kind_is_kept(self):
        where = {"kind": "table"}
        self.assertIs(retriever._with_default_kind(where), where)

    def test_explicit_kind_inside_and_is_kept(self):
        where = {"$and": [{"kind": "appendix-table"},
                          {"standard": "JGJ 38-2015"}]}
        self.assertIs(retriever._with_default_kind(where), where)

    def test_and_without_kind_gets_kind(self):
        where = {"$and": [{"force_status": {"$in": ["全文强制"]}},
                          {"standard": "GB 55031-2022"}]}
        got = retriever._with_default_kind(where)
        self.assertEqual(got["$and"][-1], {"kind": "article"})
        self.assertEqual(got["$and"][0], where["$and"][0])


class ApiBuildWhere(unittest.TestCase):
    def test_force_only_adds_article_kind(self):
        got = api._build_where(True, [])
        self.assertEqual(got, {"$and": [{"force_status": {"$in": api.FORCE_OK}},
                                        {"kind": "article"}]})

    def test_force_only_with_standards_ands_all_three(self):
        got = api._build_where(True, ["JGJ 38-2015"])
        self.assertEqual(len(got["$and"]), 3)
        self.assertIn({"kind": "article"}, got["$and"])
        self.assertIn({"standard": {"$in": ["JGJ 38-2015"]}}, got["$and"])

    def test_plain_standard_filter_is_unchanged(self):
        self.assertEqual(api._build_where(False, ["JGJ 38-2015"]),
                         {"standard": {"$in": ["JGJ 38-2015"]}})

    def test_no_filter_returns_none(self):
        self.assertIsNone(api._build_where(False, []))


if __name__ == "__main__":
    unittest.main()
