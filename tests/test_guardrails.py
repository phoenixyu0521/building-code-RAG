# -*- coding: utf-8 -*-
"""防线层（guardrails）的对抗用例。

这里测的是「**代码级**引用校验」，不是提示词。提示词只能劝，代码才能挡。

组织方式
--------
1. 抽取层：条文号/规范号的各种写法、归一化、数字粘连、表格 chunk
2. 判定层：编造 / 不在 hits / 跨规范张冠李戴 / 空引用 / 正确引用
3. 降级策略：抽取式 / 拒答的选择，以及降级结果本身不再含未支撑条文号
4. 域外判定：judge_relevance 的四个判据
5. 端到端：pipeline.ask / api 的 refused / degraded / verification

红线（任务要求）
----------------
· 编造引用 **100%** 拦下
· 正确引用 **0 误杀**

⚠️ 所有用例都只依赖标准库，跑法：
    python -m unittest discover -s tests -v
"""
import unittest

from app import guardrails as g


# ---------------------------------------------------------------------------
# 造数据的小工具：meta 用字典模拟 Chroma 返回的 metadatas
# ---------------------------------------------------------------------------
def hit(standard, article_no, text, kind="article", score=0.9,
        standard_name="", version=None):
    std = standard
    if version is None:
        tail = std.rsplit("-", 1)[-1]
        version = tail if tail.isdigit() and len(tail) == 4 else ""
    return {
        "text": text,
        "score": score,
        "meta": {
            "standard": std,
            "standard_name": standard_name or ("示例规范" if std else ""),
            "version": version,
            "article_no": article_no,
            "kind": kind,
            "status": "",
            "force_status": "",
        },
    }


GB52 = "GB 50352-2019"
GB31 = "GB 55031-2022"
GB37 = "GB 55037-2022"
JGJ38 = "JGJ 38-2015"

# 一组「本次检索命中的片段」，模拟一次真实的 Top-K
HITS = [
    hit(GB52, "2.0.16", "2.0.16 半地下室 semi-basement 房间地面低于室外设计地面的"
                        "平均高度大于该房间平均净高1/3，且不大于1/2者。", score=0.875),
    hit(GB52, "4.2.4", "4.2.4 建筑基地的机动车出入口位置应符合下列规定："
                       "1 距城市道路交叉口不应小于70.0m……", score=0.812),
    hit(GB31, "3.1.6", "3.1.6 下列建筑空间不应计算建筑面积：1 与建筑物内不相连通的"
                       "建筑空间；2 骑楼、过街楼底层的开放公共空间……", score=0.790),
    hit(GB31, "3.1.4", "3.1.4 建筑空间结构层高2.20m及以上者应计算建筑面积。",
        score=0.702),
    hit(GB37, "2.2.6", "2.2.6 建筑高度大于33m的住宅建筑、一类高层公共建筑等应设置"
                       "消防电梯。", score=0.690),
    hit(JGJ38, "4.2.8", "4.2.8 书库的净高不应小于2.40m。", score=0.673),
]


def judge(answer, hits=None):
    return g.verify_answer("测试问题", answer, HITS if hits is None else hits)


# ===========================================================================
# 1. 抽取层
# ===========================================================================
class TestExtract(unittest.TestCase):
    """条文号抽取与归一化：等价写法要等价，不同条文绝不能归一成同一个。"""

    def test_chinese_and_bare_forms_are_equal(self):
        cases = [
            "第3.1.6条",
            "第 3.1.6 条",
            "3.1.6",
            "条文 3.1.6 规定",
            "GB 50352-2019 第 3.1.6 条",
            "【GB 50352-2019】3.1.6",
            "GB 50352-2019 第3.1.6条",
            "（GB 50352—2019）第 3.1.6 条",
            "3．1．6",                       # 全角点
            "第 3.1.6.条",
        ]
        for a in cases:
            with self.subTest(case=a):
                self.assertIn("3.1.6", g.extract_article_nos(a))

    def test_standard_number_normalization(self):
        self.assertEqual(g.normalize_standard("GB50352-2019"), "GB 50352-2019")
        self.assertEqual(g.normalize_standard("gb 50352—2019"), "GB 50352-2019")
        self.assertEqual(g.normalize_standard("GB/T 50352-2019"), "GB/T 50352-2019")
        self.assertEqual(g.normalize_standard("JGJ38-2015"), "JGJ 38-2015")
        # 缺年号：用本次 hits 里出现过的规范号补全（known_versions）
        self.assertEqual(g.normalize_standard("GB50352", ["GB 50352-2019"]),
                         "GB 50352-2019")

    def test_number_gluing_is_not_equal(self):
        """数字粘连：3.1.6 与 3.16 是两条不同的条文，绝不能混。"""
        self.assertEqual(g.extract_article_nos("第3.16条"), {"3.16"})
        self.assertEqual(g.extract_article_nos("第3.1.6条"), {"3.1.6"})
        self.assertNotEqual(g.extract_article_nos("第3.16条"),
                            g.extract_article_nos("第3.1.6条"))
        # 多级条款也不能被截断成前两级
        self.assertEqual(g.extract_article_nos("第4.2.4.1款"), {"4.2.4.1"})

    def test_glued_number_is_rejected_against_real_article(self):
        """库里只有 3.1.6，答案写 3.16 —— 必须判为未支撑，而不是「差不多」。"""
        v = judge("依据【GB 55031-2022】3.16，该空间不应计算建筑面积。")
        self.assertFalse(v["ok"])
        self.assertIn("3.16", v["unverified_numbers"])

    def test_appendix_and_table_forms(self):
        self.assertEqual(g.extract_article_nos("见附录A"), {"附录A"})
        self.assertEqual(g.extract_article_nos("见附表 B"), {"附录B"})
        self.assertEqual(g.extract_article_nos("表4.3.5 规定"), {"4.3.5"})
        self.assertEqual(g.extract_article_nos("表 4.3.5-1"), {"4.3.5"})

    def test_dates_and_dimensions_are_not_articles(self):
        """日期、尺寸、数量不能被误抽成条文号 —— 否则正确回答会被误杀。"""
        for text in ("2019年发布，2020年1月1日起施行", "净高 2.40m，不小于 2.30m",
                     "藏书量超过 100 万册", "面积 3 平方米"):
            with self.subTest(text=text):
                self.assertEqual(g.extract_article_nos(text), set())

    def test_bare_standard_without_article_is_not_a_citation(self):
        """只写规范名不写条文号：不算「已核实」，也拿不到来源。"""
        v = judge("【GB 50352-2019】对半地下室有定义。")
        self.assertFalse(v["ok"])
        self.assertIn("no_citation", v["reason_codes"])
        self.assertEqual(v["cited"], [])
        self.assertIn(GB52, v["standards_cited"])


# ===========================================================================
# 2. 判定层：该拦的一个不漏
# ===========================================================================
class TestFabricated(unittest.TestCase):
    """编造引用 —— 红线的上半句：一个都不能放过。"""

    def test_fabricated_article_in_hit_standard(self):
        v = judge("依据【GB 50352-2019】第 9.9.9 条，半地下室应设置独立出入口。")
        self.assertFalse(v["ok"])
        self.assertIn("9.9.9", v["unverified_numbers"])
        self.assertIn("not_in_hits", v["reason_codes"])

    def test_fabricated_article_in_other_standard(self):
        v = judge("依据【JGJ 38-2015】第 12.3.1 条，书库应设自动喷水灭火系统。")
        self.assertFalse(v["ok"])
        self.assertIn("12.3.1", v["unverified_numbers"])

    def test_bare_number_not_in_hits(self):
        v = judge("按第 5.5.5 条规定，消防电梯应设前室。")
        self.assertFalse(v["ok"])

    def test_bare_number_in_hits_is_fine(self):
        """不带规范号但条文号确实来自 hits —— 不算编造。"""
        v = judge("按第 2.0.16 条，半地下室是……")
        self.assertTrue(v["ok"], v["reasons"])


class TestCrossStandard(unittest.TestCase):
    """跨规范张冠李戴 —— 条文号对得上，但出处是另一本规范。"""

    def test_article_exists_in_other_standard(self):
        """3.1.6 在 hits 里只属于 GB 55031-2022，答案却说成 GB 50352-2019。"""
        v = judge("依据【GB 50352-2019】第 3.1.6 条，骑楼底层不计入面积。")
        self.assertFalse(v["ok"])
        self.assertIn("standard_mismatch", v["reason_codes"])

    def test_hit_without_standard_label_is_amb_tolerated(self):
        """条文号在 hits 里只属于一本规范 → 句子没写规范号也认，不必苛求格式。

        （3.1.6 在本组 hits 里只出现在 GB 55031-2022，出处唯一即可确定。）
        """
        v = judge("第 3.1.6 条规定，骑楼底层开放公共空间不计算建筑面积。")
        self.assertTrue(v["ok"], v["reasons"])
        self.assertEqual(v["cited"], [{"standard": GB31, "article_no": "3.1.6"}])

    def test_ambiguous_article_across_hits_is_rejected(self):
        """同一批 hits 里 4.2.2 同时出现在两本规范，答案又不写规范号 → 无法核实。"""
        hits = [hit(GB52, "4.2.2", "4.2.2 建筑控制线……", score=0.78),
                hit(GB31, "4.2.2", "4.2.2 建筑空间净高……", score=0.76)]
        v = g.verify_answer("q", "第 4.2.2 条规定了建筑控制线。", hits)
        self.assertFalse(v["ok"])
        self.assertIn("ambiguous_article", v["reason_codes"])

    def test_multi_standard_citation_is_allowed(self):
        """合规的多规范引用不能被当成跨规范拼接（金标本身就跨规范）。"""
        ans = ("半地下室见【GB 50352-2019】2.0.16；"
               "不计面积的空间见【GB 55031-2022】3.1.6。")
        v = judge(ans)
        self.assertTrue(v["ok"], v["reasons"])

    def test_cite_old_replace_designator(self):
        """GBJ 16-87 被 GB 55037-2022 替代：答案里出现废止规范号也要拦。"""
        v = judge("依据【GBJ 16-87】第 3.1.6 条，防火间距为 6m。")
        self.assertFalse(v["ok"])
        self.assertIn("standard_mismatch", v["reason_codes"])


class TestEmptyAndRefusal(unittest.TestCase):
    def test_empty_answer(self):
        v = judge("")
        self.assertFalse(v["ok"])
        self.assertIn("empty_answer", v["reason_codes"])

    def test_answer_is_pure_refusal(self):
        """模型按铁律第 4 条拒答：不是「不合格」，是「该拒答」。"""
        v = judge("本知识库未收录该内容。")
        self.assertFalse(v["ok"])
        self.assertTrue(v["refusal_only"])
        self.assertNotIn("no_citation", v["reason_codes"])

    def test_answer_without_any_citation(self):
        v = judge("半地下室是房间地面低于室外设计地面平均高度大于该房间平均净高"
                  "三分之一且不大于二分之一的房间。")
        self.assertFalse(v["ok"])
        self.assertIn("no_citation", v["reason_codes"])

    def test_answer_without_citation_is_never_returned_verbatim(self):
        """没有引用但答对了也算不合格 —— 铁律第 2 条要求必须标出处。"""
        ans = "半地下室是房间地面低于室外设计地面平均高度大于净高三分之一者。"
        v = judge(ans)
        final = g.apply_policy(ans, v, HITS)
        self.assertNotEqual(final["answer"].strip(), ans.strip())
        self.assertIn(final["strategy"], ("extractive", "refuse"))


class TestCorrect(unittest.TestCase):
    """正确引用 —— 红线的下半句：一个都不能误杀。"""

    def test_canonical_citation(self):
        ans = "依据【GB 50352-2019】2.0.16，半地下室是指房间地面低于室外设计地面的"
        ans += "平均高度大于该房间平均净高1/3、且不大于1/2者。"
        v = judge(ans)
        self.assertTrue(v["ok"], v["reasons"])
        self.assertEqual(v["reason_codes"], [])
        self.assertEqual(v["unverified_numbers"], [])

    def test_equivalent_spellings(self):
        for ans in ("依据【GB 50352-2019】第2.0.16条，半地下室……",
                    "依据 GB50352-2019 第 2.0.16 条，半地下室……",
                    "《民用建筑统一设计标准》GB 50352—2019 第 2.0.16 条：半地下室……",
                    "（GB 50352-2019）2.0.16 半地下室……"):
            with self.subTest(ans=ans):
                self.assertTrue(judge(ans)["ok"], judge(ans)["reasons"])

    def test_multiple_citations(self):
        ans = ("半地下室见【GB 50352-2019】2.0.16；消防电梯见【GB 55037-2022】2.2.6。")
        self.assertTrue(judge(ans)["ok"], judge(ans)["reasons"])

    def test_numbers_in_answer_are_not_mistaken_for_articles(self):
        ans = ("依据【GB 50352-2019】2.0.16，房间地面低于室外设计地面平均高度大于"
               "该房间平均净高 1/3 且不大于 1/2。书库净高不应小于 2.40m"
               "（【JGJ 38-2015】4.2.8）。")
        v = judge(ans)
        self.assertTrue(v["ok"], v["reasons"])

    def test_appendix_and_table_hit(self):
        hits = HITS + [hit(JGJ38, "4.3.5-表",
                           "表4.3.5-1 开架阅览桌之间的主通道净宽（m）",
                           kind="table", score=0.754)]
        for ans in ("见【JGJ 38-2015】表 4.3.5，主通道净宽为 1.50m。",
                    "见【JGJ 38-2015】4.3.5-1 表。"):
            with self.subTest(ans=ans):
                v = g.verify_answer("开架阅览桌主通道净宽", ans, hits)
                self.assertTrue(v["ok"], v["reasons"])

    def test_appendix_hit(self):
        hits = HITS + [hit(JGJ38, "附录B", "附录B 阅览室每座使用面积指标",
                           kind="appendix", score=0.60)]
        v = g.verify_answer("每座使用面积", "见【JGJ 38-2015】附录B。", hits)
        self.assertTrue(v["ok"], v["reasons"])

    def test_table_chunk_backing_body_article(self):
        """命中是「4.3.5-表」，答案正文引用「4.3.5」—— 表格与其正文条例是一体的。"""
        hits = [hit(JGJ38, "4.3.5-表", "表4.3.5 主通道净宽", kind="table", score=0.75)]
        v = g.verify_answer("主通道", "依据【JGJ 38-2015】4.3.5，净宽 1.50m。", hits)
        self.assertTrue(v["ok"], v["reasons"])

    def test_table_article_cited_but_hit_is_body_article(self):
        """反向：命中的是正文条目，答案引「表4.3.5-表」（有的模型会这么写）。"""
        hits = [hit(JGJ38, "4.3.5", "4.3.5 开架阅览桌主通道净宽应符合表4.3.5的规定",
                    score=0.75)]
        v = g.verify_answer("主通道", "依据【JGJ 38-2015】表4.3.5，净宽 1.50m。", hits)
        self.assertTrue(v["ok"], v["reasons"])


# ===========================================================================
# 3. 降级策略
# ===========================================================================
class TestPolicy(unittest.TestCase):
    """不合格时：能抽取就抽取，抽取不了就拒答 —— 两种策略不能混。"""

    def test_good_answer_is_kept(self):
        ans = "依据【GB 50352-2019】2.0.16，半地下室是指……"
        out = g.apply_policy(ans, judge(ans), HITS)
        self.assertEqual(out["strategy"], "keep")
        self.assertFalse(out["degraded"])
        self.assertEqual(out["answer"], ans)

    def test_refusal_is_kept_as_refusal(self):
        ans = "本知识库未收录该内容。"
        out = g.apply_policy(ans, judge(ans), HITS)
        self.assertEqual(out["strategy"], "refuse")
        self.assertFalse(out["degraded"])

    def test_bad_answer_falls_back_to_extraction(self):
        ans = "依据【GB 50352-2019】第 9.9.9 条，半地下室应设独立出入口，宽度 1.2m。"
        out = g.apply_policy(ans, judge(ans), HITS)
        self.assertEqual(out["strategy"], "extractive")
        self.assertTrue(out["degraded"])
        # 抽取式：直接是最高分片段原文，且带出处
        self.assertIn("房间地面低于室外设计地面的平均高度大于该房间平均净高1/3",
                      out["answer"])
        self.assertIn(GB52, out["answer"])
        self.assertIn("2.0.16", out["answer"])
        # 关键：降级结果里不能再出现编造的 9.9.9
        self.assertNotIn("9.9.9", out["answer"])
        # 降级结果自身必须通过校验
        self.assertTrue(g.verify_answer("q", out["answer"], HITS)["ok"])

    def test_extraction_picks_highest_scoring_hit(self):
        hits = [hit(GB31, "3.1.4", "3.1.4 结构层高2.20m及以上应计算面积。", score=0.70),
                hit(GB52, "2.0.16", "2.0.16 半地下室……", score=0.90)]
        ans = "依据【GB 50352-2019】第 7.7.7 条……"
        out = g.apply_policy(ans, g.verify_answer("q", ans, hits), hits)
        self.assertEqual(out["strategy"], "extractive")
        self.assertIn("2.0.16", out["answer"])
        self.assertNotIn("3.1.4", out["answer"])

    def test_every_hit_unusable_means_refuse(self):
        """所有 hit 都是空正文/无编号 → 抽取不出来 → 必须拒答，不能编。"""
        bad = [hit(GB52, "", ""), hit(GB31, "", "   ")]
        ans = "依据【GB 50352-2019】第 9.9.9 条……"
        out = g.apply_policy(ans, g.verify_answer("q", ans, bad), bad)
        self.assertEqual(out["strategy"], "refuse")
        self.assertTrue(out["degraded"])
        self.assertEqual(out["answer"], g.NO_HIT_REPLY)

    def test_no_hits_means_refuse(self):
        ans = "依据【GB 50352-2019】第 2.0.16 条……"
        out = g.apply_policy(ans, g.verify_answer("q", ans, []), [])
        self.assertEqual(out["strategy"], "refuse")
        self.assertEqual(out["answer"], g.NO_HIT_REPLY)

    def test_extraction_labels_source_and_kind(self):
        hits = [hit(GB31, "3.1.6", "3.1.6 下列建筑空间不应计算建筑面积……", score=0.9)]
        ans = "依据【GB 50352-2019】第 8.8.8 条……"
        out = g.apply_policy(ans, g.verify_answer("q", ans, hits), hits)
        self.assertIn("3.1.6", out["answer"])
        self.assertIn("GB 55031-2022", out["answer"])


# ===========================================================================
# 4. 域外判定
# ===========================================================================
class TestRelevance(unittest.TestCase):
    """域外判定的**实测结论**：在现有检索栈上不存在好用的词面判据。

    这里断言的不是「某种判据能用」，而是：
      · 默认关闭（gate=off → in_domain 恒为 True，不误杀）
      · 开启时（min_continuity）确实能挡一部分，但代价必须可见
      · 诊断数字（top_score / longest_run）照常输出，供接 rerank 后调参
    """

    def test_gate_off_by_default_never_kills(self):
        """默认不拦截：哪怕问的是域外问题，也不在词面层拦 —— 交给引用校验。"""
        hits = [hit(JGJ38, "6.4.6", "图书馆……疏散门，可设置门禁系统……", score=0.723)]
        v = g.judge_relevance("图书馆人脸识别门禁系统应该怎么设计？", hits)
        self.assertTrue(v["in_domain"])
        self.assertEqual(v["gate"], "off")

    def test_empty_hits_is_out_of_domain(self):
        v = g.judge_relevance("随便问点什么", [])
        self.assertFalse(v["in_domain"])

    def test_gate_on_blocks_off_topic_question(self):
        """开启后：问句里的实词连续串在命中里找不到 → 判为域外。"""
        hits = [hit(JGJ38, "8.4.5", "图书馆应设置开、闭馆音响讯号装置……", score=0.599)]
        v = g.judge_relevance("图书馆元宇宙阅读空间的声学指标是多少？",
                              hits, min_continuity=4)
        self.assertFalse(v["in_domain"], v["reasons"])

    def test_gate_on_can_still_be_fooled(self):
        """诚实记录上限：问句里带「图书馆」这种标题词，就能白送一段连续命中。

        「图书馆人脸识别门禁系统」里有 4 个连续字正好是命中片段开头的「图书馆」…
        实际算出来最长串 7 字 —— 因为片段整句都含这几个字。
        这说明：词面判据在**同一本规范的语境**里根本没有区分力。
        """
        hits = [hit(JGJ38, "6.4.6", "图书馆需要控制人员随意出入的疏散门，可设置门禁系统……",
                    score=0.723)]
        v = g.judge_relevance("图书馆人脸识别门禁系统应该怎么设计？",
                              hits, min_continuity=4)
        self.assertTrue(v["in_domain"])         # ← 漏过：这就是它默认关闭的原因
        self.assertGreaterEqual(v["longest_run"], 4)

    def test_gate_on_kills_real_questions(self):
        """诚实记录代价：开启后正例也会被误杀 —— 这就是默认关闭的原因。

        （这条其实没被误杀 —— 主题词连续命中 7 字。真正被误杀的是
          `test_gate_on_blocks_off_topic_question` 那一侧的短问句，
          以及评测集里 9/77 条正例，见 结果_guardrails.md。）
        """
        hits = [hit(GB19, "2.2.5", "自动扶梯和楼梯下方的低矮空间应设防护设施……",
                    score=0.675)]
        v = g.judge_relevance("自动扶梯和楼梯下方的低矮空间怎么处理？", hits)
        v4 = g.judge_relevance("自动扶梯和楼梯下方的低矮空间怎么处理？", hits,
                               min_continuity=4)
        self.assertTrue(v["in_domain"])          # 默认关闭：放行
        self.assertTrue(v4["in_domain"])         # 开启：主题词命中 7 字，也放行
        self.assertGreaterEqual(v4["longest_run"], 4)

    def test_diagnostics_are_reported(self):
        v = g.judge_relevance("书库的净高是多少", HITS)
        for k in ("longest_run", "top_score", "n_hits", "gate"):
            self.assertIn(k, v)
        self.assertIsInstance(v["longest_run"], int)

    def test_longest_run_counts_contiguous_chars(self):
        hits = [hit(JGJ38, "4.2.8", "书库的净高不应小于2.40m。", score=0.8)]
        # 「书库的净高」去掉功能字「的」→「书库净高」，最长连续命中「书库」2 字
        self.assertEqual(g.longest_match_run("书库的净高是多少", hits), 2)
        self.assertEqual(g.longest_match_run("完全不相关的星球大战话题", hits), 0)


GB19 = "GB 55019-2021"


# ===========================================================================
# 5. 端到端：pipeline
# ===========================================================================
class TestPipeline(unittest.TestCase):
    """用 monkeypatch 打桩，不加载 torch / 不调真实 LLM。"""

    def setUp(self):
        from app import pipeline
        self.pipe = pipeline
        self._orig = (pipeline.Retriever, pipeline.chat)

    def tearDown(self):
        self.pipe.Retriever, self.pipe.chat = self._orig

    def _stub(self, hits, text):
        class FakeRetriever:
            def __init__(self, *a, **k):
                pass

            def search(self, question, top_k=None, where=None):
                return list(hits)

        self.pipe.Retriever = FakeRetriever
        self.pipe.chat = lambda messages, **k: (text, {"total_tokens": 1})

    def test_good_llm_answer_passes(self):
        self._stub(HITS, "依据【GB 50352-2019】2.0.16，半地下室是指……")
        r = self.pipe.ask("半地下室的定义")
        self.assertTrue(r["used_llm"])
        self.assertFalse(r["degraded"])
        self.assertTrue(r["verification"]["ok"])
        self.assertTrue(r["answer"].startswith("依据"))

    def test_fabricated_answer_is_replaced(self):
        self._stub(HITS, "依据【GB 50352-2019】第 9.9.9 条，半地下室应设独立出入口。")
        r = self.pipe.ask("半地下室的定义")
        self.assertTrue(r["used_llm"])            # 语义不变：确实调了模型
        self.assertTrue(r["degraded"])
        self.assertFalse(r["verification"]["ok"])
        self.assertNotIn("9.9.9", r["answer"])
        self.assertIn("2.0.16", r["answer"])

    def test_no_hits_short_circuit_has_verification(self):
        self._stub([], "依据【GB 50352-2019】2.0.16……")
        r = self.pipe.ask("随便")
        self.assertFalse(r["used_llm"])
        self.assertTrue(r["degraded"])
        self.assertFalse(r["verification"]["ok"])

    def test_out_of_domain_gate_can_be_enabled(self):
        """域外闸门默认关；对**确实没有词面关联**的问题打开时，不调模型。"""
        self._stub([hit(JGJ38, "8.4.5", "图书馆应设置开、闭馆音响讯号装置……", score=0.599)],
                   "依据【JGJ 38-2015】8.4.5，元宇宙声学……")
        r = self.pipe.ask("图书馆元宇宙阅读空间的声学指标是多少？", min_continuity=4)
        self.assertFalse(r["used_llm"])
        self.assertEqual(r["answer"], g.NO_HIT_REPLY)

    def test_out_of_domain_gate_off_still_verifies_citations(self):
        """闸门关着时，域外问题也不能靠编条号蒙混 —— 引用校验照样拦。"""
        self._stub([hit(JGJ38, "6.4.6", "图书馆……疏散门，可设置门禁系统……", score=0.723)],
                   "依据【JGJ 38-2015】第 7.7.7 条，人脸识别门禁应……")
        r = self.pipe.ask("图书馆人脸识别门禁系统应该怎么设计？")
        self.assertTrue(r["used_llm"])
        self.assertTrue(r["degraded"])
        self.assertNotIn("7.7.7", r["answer"])

    def test_no_llm_mode_also_gets_verification(self):
        class FakeRetriever:
            def __init__(self, *a, **k):
                pass

            def search(self, question, top_k=None, where=None):
                return list(HITS)

        self.pipe.Retriever = FakeRetriever
        r = self.pipe.ask("半地下室的定义", allow_llm=False)
        self.assertIn("verification", r)


# ===========================================================================
# 6. 端到端：api 的 refused / verification
# ===========================================================================
class TestApi(unittest.TestCase):
    def setUp(self):
        from app import api
        self.api = api
        self._orig = api.pipeline.ask

    def tearDown(self):
        self.api.pipeline.ask = self._orig

    def _post(self, payload):
        from fastapi.testclient import TestClient
        return TestClient(self.api.app).post("/api/ask", json={"question": "q"})

    def test_refused_uses_verification_not_string_equality(self):
        text = "依据【GB 50352-2019】第 9.9.9 条……"
        self.api.pipeline.ask = lambda *a, **k: {
            "question": "q", "hits": HITS, "used_llm": True,
            "answer": g.apply_policy(text, g.verify_answer("q", text, HITS), HITS)["answer"],
            "degraded": True,
            "verification": dict(g.verify_answer("q", text, HITS),
                                 strategy="extractive", degraded=True),
        }
        body = self._post({"question": "q"}).json()
        self.assertIn("refused", body)
        self.assertIn("verification", body)
        self.assertTrue(body["degraded"])
        # 摘要里不能带内部路径
        flat = repr(body["verification"])
        self.assertNotIn("D:\\", flat)
        self.assertNotIn("chroma_db", flat)

    def test_good_answer_not_refused(self):
        ans = "依据【GB 50352-2019】2.0.16，半地下室是指……"
        self.api.pipeline.ask = lambda *a, **k: {
            "question": "q", "hits": HITS, "used_llm": True, "answer": ans,
            "degraded": False,
            "verification": dict(g.verify_answer("q", ans, HITS),
                                 strategy="keep", degraded=False),
        }
        body = self._post({"question": "q"}).json()
        self.assertFalse(body["refused"])
        self.assertFalse(body["degraded"])


if __name__ == "__main__":
    unittest.main()
