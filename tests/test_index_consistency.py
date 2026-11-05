# -*- coding: utf-8 -*-
"""P0-C v2 索引一致性的纯逻辑测试（不 import chromadb / sentence_transformers）。

工作包要求覆盖：
  · emb_meta：sha 变、model 变、count 变、dim 变、meta 缺失
  · 原子重建：模拟 add 中途异常，断言旧 collection 仍在（且从未被删）
  · .bin 检查：别的集合目录里的同名文件不能让它通过；目录名按 segment 规则算
  · 配置解析：环境变量覆盖优先级

跑法（项目根目录下）：
    python -m unittest discover -s tests -v
"""
import json
import os
import shutil
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from app import index_ops                      # noqa: E402
from app.config import Config                  # noqa: E402

SCRIPTS = os.path.join(ROOT, "输出文档", "scripts")


def make_cfg(tmp, **kw):
    """在临时目录里造一份配置（语料/缓存/库目录都指向 tmp）。"""
    chunks = os.path.join(tmp, "chunks.jsonl")
    cache = os.path.join(tmp, "cache")
    db = os.path.join(tmp, "db")
    os.makedirs(cache, exist_ok=True)
    os.makedirs(db, exist_ok=True)
    return Config(chunks_file=chunks, cache_dir=cache, chroma_dir=db,
                  npy_file=os.path.join(cache, "emb.npy"),
                  meta_file=os.path.join(cache, "emb_meta.json"),
                  embed_model="Qwen/Qwen3-Embedding-0.6B", embed_dim=4,
                  collection="standards", **kw)


def write_chunks(path, texts):
    with open(path, "w", encoding="utf-8") as f:
        for i, t in enumerate(texts):
            f.write(json.dumps({"id": f"c{i}", "text": t, "article_no": f"3.{i}",
                                "standard": "GB 50000-2020",
                                "standard_name": "测试标准"}, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# 1) emb_meta 五项硬门禁（问题 3）
# ---------------------------------------------------------------------------
class TestEmbMeta(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="p0c-meta-")
        self.cfg = make_cfg(self.tmp)
        write_chunks(self.cfg.CHUNKS_FILE, ["甲", "乙", "丙"])
        self.rows, self.sha1 = index_ops.read_chunks_file(self.cfg.CHUNKS_FILE)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _ok_meta(self):
        return {"sha1": self.sha1, "count": len(self.rows),
                "model": self.cfg.EMBED_MODEL, "dim": 4,
                "shape": [len(self.rows), 4]}

    def _problems(self, meta):
        return index_ops.validate_emb_meta(meta, self.rows, self.sha1, self.cfg)

    def test_baseline_passes(self):
        self.assertEqual(self._problems(self._ok_meta()), [])

    def test_sha_changed_fails(self):                      # 条数没变但正文变了
        write_chunks(self.cfg.CHUNKS_FILE, ["甲", "乙", "丙（改）"])
        rows2, sha2 = index_ops.read_chunks_file(self.cfg.CHUNKS_FILE)
        self.assertEqual(len(rows2), len(self.rows))       # 前提：条数不变
        problems = index_ops.validate_emb_meta(self._ok_meta(), rows2, sha2, self.cfg)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("sha1", problems[0])

    def test_model_changed_fails(self):
        problems = self._problems(dict(self._ok_meta(), model="other/model"))
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("向量模型不一致", problems[0])

    def test_count_changed_fails(self):
        problems = self._problems(dict(self._ok_meta(), count=2, shape=[2, 4]))
        self.assertTrue(any("条数不一致" in p for p in problems), problems)

    def test_dim_changed_fails(self):
        problems = self._problems(dict(self._ok_meta(), dim=8, shape=[3, 8]))
        self.assertTrue(any("向量维度不一致" in p for p in problems), problems)

    def test_shape_inconsistent_fails(self):
        problems = self._problems(dict(self._ok_meta(), shape=[3, 8]))
        self.assertTrue(any("不自洽" in p for p in problems), problems)

    def test_meta_missing_fails(self):
        problems = self._problems(None)
        self.assertEqual(len(problems), 1, problems)
        self.assertIn("emb_meta.json 缺失", problems[0])

    def test_meta_missing_field_fails(self):
        meta = self._ok_meta()
        del meta["shape"]
        problems = self._problems(meta)
        self.assertTrue(any("缺字段" in p and "shape" in p for p in problems), problems)

    def test_corrupt_meta_file_raises(self):
        with open(self.cfg.META_FILE, "w", encoding="utf-8") as f:
            f.write("{ not json")
        with self.assertRaises(index_ops.EmbMetaError):
            index_ops.load_emb_meta(self.cfg.META_FILE)

    def test_absent_meta_file_returns_none(self):
        self.assertIsNone(index_ops.load_emb_meta(self.cfg.META_FILE))

    def test_rebuild_hint_mentions_force(self):
        self.assertIn("--stage vec --force",
                      index_ops.EmbMetaError(["x"]).rebuild_hint)

    def test_legit_shrink_is_not_a_source_problem(self):
        """合法缩容：正文 3 条 → 2 条、缓存同步重算 → 同源判断必须通过。

        「旧集合 1083 条、新正文 972 条」正是这种情形；同源只看 sha1 + model，
        不看旧集合条数。
        """
        write_chunks(self.cfg.CHUNKS_FILE, ["甲", "乙"])
        rows2, sha2 = index_ops.read_chunks_file(self.cfg.CHUNKS_FILE)
        meta2 = {"sha1": sha2, "count": 2, "model": self.cfg.EMBED_MODEL,
                 "dim": 4, "shape": [2, 4]}
        self.assertEqual(index_ops.validate_emb_meta(meta2, rows2, sha2, self.cfg), [])


# ---------------------------------------------------------------------------
# 2) 集合元数据（模型 / 维度 / 正文版本）
# ---------------------------------------------------------------------------
class TestCollectionMeta(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(embed_model="m/current", embed_dim=4)

    def test_meta_carries_model_dim_sha(self):
        meta = index_ops.collection_meta(self.cfg, sha1="abc123")
        self.assertEqual(meta[index_ops.MODEL_META_KEY], "m/current")
        self.assertEqual(meta[index_ops.DIM_META_KEY], 4)
        self.assertEqual(meta[index_ops.SHA1_META_KEY], "abc123")

    def test_matching_model_passes(self):
        index_ops.assert_collection_model(index_ops.collection_meta(self.cfg), self.cfg)

    def test_mismatched_model_fails_loudly(self):
        meta = dict(index_ops.collection_meta(self.cfg),
                    **{index_ops.MODEL_META_KEY: "m/old"})
        with self.assertRaises(SystemExit) as ctx:
            index_ops.assert_collection_model(meta, self.cfg)
        self.assertIn("m/old", str(ctx.exception))

    def test_missing_model_meta_fails(self):
        with self.assertRaises(SystemExit) as ctx:
            index_ops.assert_collection_model({"hnsw:space": "cosine"}, self.cfg)
        self.assertIn(index_ops.MODEL_META_KEY, str(ctx.exception))

    def test_dim_mismatch_reported(self):
        meta = dict(index_ops.collection_meta(self.cfg), **{index_ops.DIM_META_KEY: 8})
        self.assertTrue(any("维度" in p
                            for p in index_ops.collection_problems(meta, self.cfg)))

    def test_sha_mismatch_reported_only_when_known(self):
        meta = index_ops.collection_meta(self.cfg, sha1="aaa")
        self.assertEqual(index_ops.collection_problems(meta, self.cfg, sha1="aaa"), [])
        self.assertTrue(index_ops.collection_problems(meta, self.cfg, sha1="bbb"))


# ---------------------------------------------------------------------------
# 3) .bin 归属：按 Chroma 的 segment 规则找目录（问题 2）
# ---------------------------------------------------------------------------
SEG_A = "cb1492d7-05be-45e7-85a1-b4743f545fb2"
SEG_META = "aaaaaaaa-0000-0000-0000-000000000001"
SEG_OTHER = "12345678-0000-0000-0000-000000000002"


def vec_dir(seg_ids):
    return str(sum(int(s.replace("-", ""), 16) for s in seg_ids) + 1)


class TestBinScope(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="p0c-bins-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _touch(self, db, seg, names=index_ops.NEED_BINS):
        d = os.path.join(db, seg)
        os.makedirs(d, exist_ok=True)
        for n in names:
            with open(os.path.join(d, n), "wb") as f:
                f.write(b"x")

    def test_vector_segment_dir_is_uuid_sum_plus_one(self):
        self.assertEqual(index_ops.vector_segment_for([(SEG_A, "VECTOR", "coll")]),
                         vec_dir([SEG_A]))
        segs = [(SEG_A, "VECTOR", "coll"), (SEG_META, "METADATA", "coll")]
        self.assertEqual(index_ops.vector_segment_for(segs), vec_dir([SEG_A]))

    def test_dir_never_equals_collection_id(self):
        """回归：v1 拿集合 id 当目录名 → 972 条集合永远「缺 4 个文件」。

        实测 Chroma 1.5.9 的目录名就是**向量段的 uuid 本身**（不是 uuid+1）。
        """
        cid = "fda6a24a-0000-0000-0000-000000000000"
        cands = index_ops.segment_dirs_for_collection(
            cid, [(SEG_A, "VECTOR", cid)])
        self.assertEqual(cands[0], SEG_A)
        self.assertNotEqual(cands[0], cid)

    def test_other_collection_bins_do_not_satisfy(self):
        self._touch(self.tmp, vec_dir([SEG_OTHER]))           # 别的集合 4 个都齐
        mine = vec_dir([SEG_A])
        os.makedirs(os.path.join(self.tmp, mine), exist_ok=True)   # 我的目录空着
        scan = index_ops.scan_bins(self.tmp)
        d, missing = index_ops.find_bins_dir(self.tmp, [mine], scan)
        self.assertIsNone(d)                                  # 全局 glob 会误判通过
        self.assertEqual(sorted(missing), sorted(index_ops.NEED_BINS))

    def test_own_dir_passes(self):
        mine = vec_dir([SEG_A])
        self._touch(self.tmp, mine)
        self._touch(self.tmp, vec_dir([SEG_OTHER]))
        scan = index_ops.scan_bins(self.tmp)
        d, missing = index_ops.find_bins_dir(self.tmp, [mine], scan)
        self.assertEqual(d, mine, missing)

    def test_tmp_dir_accepted(self):
        """小于 sync_threshold 的集合可能先搭 ``<dir>-tmp``。"""
        mine = vec_dir([SEG_A])
        self._touch(self.tmp, mine + "-tmp")
        scan = index_ops.scan_bins(self.tmp)
        d, _ = index_ops.find_bins_dir(self.tmp, [mine, mine + "-tmp"], scan)
        self.assertEqual(d, mine + "-tmp")

    def test_partial_bins_fail(self):
        mine = vec_dir([SEG_A])
        self._touch(self.tmp, mine, names=("header.bin", "data_level0.bin"))
        scan = index_ops.scan_bins(self.tmp)
        ok, missing = index_ops.collection_have_bins(self.tmp, mine, scan)
        self.assertFalse(ok)
        self.assertEqual(sorted(missing), ["length.bin", "link_lists.bin"])

    def test_wait_bins_db_uses_segment_mapping(self):
        import sqlite3
        db = os.path.join(self.tmp, "db")
        os.makedirs(db)
        self._touch(db, SEG_A)
        sqlite_path = os.path.join(db, "chroma.sqlite3")
        con = sqlite3.connect(sqlite_path)
        con.executescript(
            "create table collections (id text, name text);"
            "create table segments (id text, scope text, collection text);")
        con.execute("insert into collections values (?,?)",
                    ("fda6a24a-0000-0000-0000-000000000000", "standards_v6"))
        con.execute("insert into segments values (?,?,?)",
                    (SEG_A, "VECTOR", "fda6a24a-0000-0000-0000-000000000000"))
        con.commit()
        con.close()
        ok, d, missing, _ = index_ops.wait_bins_db(db, "standards_v6", timeout_s=1,
                                                   poll_s=0, sleep=lambda _s: None)
        self.assertTrue(ok, missing)
        self.assertEqual(d, SEG_A)


# ---------------------------------------------------------------------------
# 4) 原子重建：失败不删旧集合（问题 1）
# ---------------------------------------------------------------------------
class FakeCollection:
    def __init__(self, name, fail_at=None):
        self.name = name
        self.ids = []
        self.vectors = []
        self.metadata = {}
        self.fail_at = fail_at
        self.add_calls = 0

    def add(self, ids, documents, embeddings, metadatas):
        self.add_calls += 1
        if self.fail_at is not None and self.add_calls == self.fail_at:
            raise RuntimeError("模拟写入中断（磁盘满 / 断电前的状态）")
        self.ids.extend(ids)
        self.vectors.extend(embeddings)

    def count(self):
        return len(self.ids)

    def get(self, ids=None, include=None):
        return {"ids": [ids[0]], "embeddings": [self.vectors[self.ids.index(ids[0])]]}

    def query(self, query_embeddings, n_results, include=None):
        q = query_embeddings[0]
        qn = sum(a * a for a in q) ** 0.5 or 1.0

        def cos(v):
            vn = sum(a * a for a in v) ** 0.5 or 1.0
            return sum(a * b for a, b in zip(v, q)) / (vn * qn)
        best = max(range(len(self.vectors)), key=lambda i: cos(self.vectors[i]))
        return {"ids": [[self.ids[best]]], "documents": [[""]],
                "distances": [[1 - cos(self.vectors[best])]]}

    def modify(self, name=None, metadata=None):
        if name:
            self.name = name
        if metadata:
            self.metadata = metadata


class FakeClient:
    """够用的假 Chroma：建 / 删 / 改名 / 列表；可选「某次 add 抛异常」。

    与真 Chroma 一样：集合改名要显式 ``modify``；删除会记录到 ``deleted``。
    """

    def __init__(self, fail_add_at=None):
        self.cols = {}
        self.fail_add_at = fail_add_at
        self.deleted = []
        self.created = []

    def get_collection(self, name):
        if name not in self.cols:
            raise ValueError(f"collection {name} does not exist")
        return self.cols[name]

    def create_collection(self, name, metadata=None):
        if name in self.cols:
            raise ValueError(f"collection {name} already exists")
        col = FakeCollection(name, fail_at=self.fail_add_at)
        col.metadata = metadata or {}
        self.cols[name] = col
        self.created.append(name)
        return col

    def get_or_create_collection(self, name, metadata=None):
        return self.cols[name] if name in self.cols else self.create_collection(name, metadata)

    def delete_collection(self, name):
        if name not in self.cols:
            raise ValueError(f"collection {name} does not exist")
        self.deleted.append(name)
        del self.cols[name]

    def rename(self, old, new):
        assert old in self.cols, old
        self.cols[new] = self.cols.pop(old)
        self.cols[new].name = new
    def list_collections(self):
        return list(self.cols)


class TestAtomicRebuild(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="p0c-atomic-")
        self.cfg = make_cfg(self.tmp)
        self.rows = [{"id": f"c{i}", "text": f"正文 {i}", "article_no": str(i)}
                     for i in range(5)]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _embs(self, n=5):
        return [[float(i + 1), 0.0, 0.0, 1.0] for i in range(n)]

    def _clean(self, r):
        return {k: str(r.get(k)) for k in ("id", "article_no")}

    def _seed_old(self, client):
        old = client.create_collection("standards",
                                       metadata=index_ops.collection_meta(self.cfg))
        old.add(ids=[r["id"] for r in self.rows],
                documents=[r["text"] for r in self.rows],
                embeddings=self._embs(), metadatas=[self._clean(r) for r in self.rows])
        return old

    def _build(self, client, n=5):
        rows = self.rows[:n]
        return index_ops.atomic_rebuild(
            client, self.cfg, rows, self._embs(n), self._clean,
            db_batch=2, verify_sample=1, log=lambda *_: None,
            now=lambda: "T1", keep_backup=False)

    def test_success_switches_without_deleting_active(self):
        client = FakeClient()
        self._seed_old(client)
        active, stats = self._build(client)
        self.assertEqual(active, "standards")
        self.assertEqual(stats["count"], 5)
        self.assertNotIn("standards", client.deleted)       # 正式名从未被删
        self.assertEqual(set(client.cols), {"standards"})

    def test_failed_add_keeps_old_untouched(self):
        client = FakeClient(fail_add_at=2)                  # staging 第 2 批 add 抛异常
        old = self._seed_old(client)
        before = (list(old.ids), list(old.vectors), old.metadata)
        with self.assertRaises(index_ops.AtomicRebuildError):
            self._build(client)
        self.assertIn("standards", client.cols)             # 旧集合还在
        now = client.cols["standards"]
        self.assertEqual((now.ids, now.vectors, now.metadata), before)
        self.assertNotIn("standards", client.deleted)       # 从未被删
        self.assertEqual([n for n in client.cols if n.startswith("staging-")], [])

    def test_first_build_failure_leaves_db_empty(self):
        client = FakeClient(fail_add_at=1)
        with self.assertRaises(index_ops.AtomicRebuildError):
            self._build(client)
        self.assertEqual(client.cols, {})

    def test_vectors_shorter_than_rows_refused_without_touching_db(self):
        client = FakeClient()
        old = self._seed_old(client)
        with self.assertRaises(index_ops.AtomicRebuildError):
            index_ops.atomic_rebuild(client, self.cfg, self.rows, self._embs(4),
                                     self._clean, log=lambda *_: None,
                                     keep_backup=False)
        self.assertEqual(len(client.cols["standards"].ids), 5)
        self.assertTrue(old is client.cols["standards"])

    def test_swap_failure_keeps_old(self):
        client = FakeClient()
        self._seed_old(client)

        def broken_swap():
            raise RuntimeError("模拟 S1 之后失效")

        with self.assertRaises(index_ops.AtomicRebuildError):
            index_ops.atomic_rebuild(
                client, self.cfg, self.rows, self._embs(), self._clean,
                db_batch=2, log=lambda *_: None, now=lambda: "T1",
                swap=broken_swap, keep_backup=False)
        self.assertIn("standards", client.cols)
        self.assertEqual(client.cols["standards"].count(), 5)

    def test_legit_shrink_switches(self):
        """合法缩容：旧 5 条 → 新 3 条，必须能切换（不要求条数相等）。"""
        client = FakeClient()
        self._seed_old(client)
        active, stats = self._build(client, n=3)
        self.assertEqual(stats["count"], 3)
        self.assertEqual(client.cols["standards"].count(), 3)

    def test_staging_and_archive_names_are_legal_for_chroma(self):
        for n in (index_ops.staging_name("standards", "20260101120000"),
                  index_ops.archive_name("standards", "20260101120000")):
            self.assertFalse(n.startswith("_"), n)
            self.assertTrue(all(c.isalnum() or c in "._-" for c in n), n)

    def test_cleanup_restores_archive_when_active_missing(self):
        client = FakeClient()
        self._seed_old(client)
        client.rename("standards", "archive-standards-20260101")
        acted, _removed = index_ops.cleanup_stale(client, "standards",
                                                  log=lambda *_: None)
        self.assertIn("archive-standards-20260101", acted or "")
        self.assertIn("standards", client.cols)
        self.assertEqual(client.cols["standards"].count(), 5)

    def test_cleanup_removes_stale_staging(self):
        client = FakeClient()
        self._seed_old(client)
        client.create_collection("staging-standards-OLD")
        _acted, removed = index_ops.cleanup_stale(client, "standards",
                                                  log=lambda *_: None)
        self.assertEqual(removed, ["staging-standards-OLD"])
        self.assertIn("standards", client.cols)


# ---------------------------------------------------------------------------
# 5) 配置解析：环境变量覆盖优先级（问题 4）
# ---------------------------------------------------------------------------
CONFIG_PROBE = (
    "import json\n"
    "from app.config import CHROMA_DIR, COLLECTION, EMBED_MODEL, EMBED_DIM, "
    "CHUNKS_FILE, TOP_K\n"
    "print(json.dumps({'CHROMA_DIR':CHROMA_DIR,'COLLECTION':COLLECTION,"
    "'EMBED_MODEL':EMBED_MODEL,'EMBED_DIM':EMBED_DIM,"
    "'CHUNKS_FILE':CHUNKS_FILE,'TOP_K':TOP_K}))\n"
)

BAD_ASCII_PROBE = (
    "import sys\n"
    "try:\n"
    "    import app.config\n"
    "except SystemExit:\n"
    "    print('SystemExit', file=sys.stderr); sys.exit(7)\n"
)


class TestConfigResolution(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="p0c-cfg-")
        self.env_keys = ("CHROMA_DIR", "CHROMA_COLLECTION", "EMBED_MODEL",
                         "EMBED_DIM", "CHUNKS_FILE", "TOP_K", "PROJECT_ROOT",
                         "EMB_CACHE_DIR")
        self.saved = {k: os.environ.get(k) for k in self.env_keys}
        for k in self.env_keys:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, probe, **env):
        import subprocess
        e = dict(os.environ)
        e.update(env)
        e["PYTHONPATH"] = ROOT
        return subprocess.run([sys.executable, "-c", probe], cwd=self.tmp, env=e,
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace")

    def test_env_overrides_defaults(self):
        p = self._run(CONFIG_PROBE, CHROMA_DIR=r"D:\tmp\db_override",
                      CHROMA_COLLECTION="coll_override",
                      EMBED_MODEL="custom/model", EMBED_DIM="256",
                      CHUNKS_FILE=os.path.join(self.tmp, "x.jsonl"), TOP_K="9")
        self.assertEqual(p.returncode, 0, p.stderr)
        got = json.loads(p.stdout.strip().splitlines()[-1])
        self.assertEqual(got["CHROMA_DIR"], r"D:\tmp\db_override")
        self.assertEqual(got["COLLECTION"], "coll_override")
        self.assertEqual(got["EMBED_MODEL"], "custom/model")
        self.assertEqual(got["EMBED_DIM"], 256)
        self.assertEqual(got["CHUNKS_FILE"], os.path.join(self.tmp, "x.jsonl"))
        self.assertEqual(got["TOP_K"], 9)

    def test_defaults_when_unset(self):
        p = self._run(CONFIG_PROBE)
        self.assertEqual(p.returncode, 0, p.stderr)
        got = json.loads(p.stdout.strip().splitlines()[-1])
        self.assertEqual(got["COLLECTION"], "standards")
        self.assertEqual(got["EMBED_MODEL"], "Qwen/Qwen3-Embedding-0.6B")
        self.assertEqual(got["EMBED_DIM"], 1024)

    def test_empty_env_falls_back_to_default(self):
        p = self._run(CONFIG_PROBE, CHROMA_COLLECTION="   ", EMBED_DIM="")
        self.assertEqual(p.returncode, 0, p.stderr)
        got = json.loads(p.stdout.strip().splitlines()[-1])
        self.assertEqual(got["COLLECTION"], "standards")
        self.assertEqual(got["EMBED_DIM"], 1024)

    def test_non_ascii_db_dir_refused(self):
        p = self._run(BAD_ASCII_PROBE, CHROMA_DIR="D:\\WorkBuddy\\项目\\db")
        self.assertNotEqual(p.returncode, 0)


# ---------------------------------------------------------------------------
# 6) 脚本接线：不硬编码、配置壳能被 git 跟踪、坏库退出码 1、--json
# ---------------------------------------------------------------------------
class TestScriptsWiring(unittest.TestCase):
    def _read(self, *parts):
        with open(os.path.join(ROOT, *parts), encoding="utf-8") as f:
            return f.read()

    def test_build_index_uses_shared_config(self):
        src = self._read("输出文档", "scripts", "build_index.py")
        self.assertIn("from _config import load_config", src)
        self.assertNotIn('DB_DIR    = r"', src)
        self.assertNotIn('COLL      = "standards"', src)
        self.assertIn("atomic_rebuild", src)

    def test_verify_index_uses_shared_config(self):
        src = self._read("输出文档", "scripts", "verify_index.py")
        self.assertIn("from _config import load_config", src)
        self.assertNotIn('DB_DIR = r"', src)
        self.assertIn("sys.exit(main())", src)

    def test_run_eval_uses_shared_config(self):
        src = self._read("输出文档", "检索评测", "run_eval.py")
        self.assertIn("_config", src)
        self.assertNotIn('CHROMA_DIR = r"', src)

    def test_config_shell_not_gitignored(self):
        """回归：``输出文档/scripts/_*`` 会吞掉下划线开头的脚本。

        配置壳原来叫 ``_bcr_config.py`` → 提交不进去 → 合并后 ImportError。
        现在叫 ``_config.py``（不以 _ 开头），git 必须看得见它。
        """
        import subprocess
        rel = "输出文档/scripts/_config.py"
        path = os.path.join(ROOT, *rel.split("/"))
        self.assertTrue(os.path.exists(path))
        try:
            out = subprocess.run(["git", "-c", f"safe.directory={ROOT}",
                                  "check-ignore", "-q", "--", rel],
                                 cwd=ROOT, capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            self.skipTest("没有 git，跳过")
        if out.returncode == 128:
            self.skipTest(f"git 读不了这个仓库：{(out.stderr or '').strip()[:120]}")
        self.assertEqual(out.returncode, 1, f"{rel} 仍被 .gitignore 忽略")
        self.assertFalse(os.path.exists(
            os.path.join(ROOT, "输出文档", "scripts", "_bcr_config.py")))

    def test_no_script_imports_old_shell(self):
        import glob
        for p in glob.glob(os.path.join(ROOT, "输出文档", "**", "*.py"), recursive=True):
            with open(p, encoding="utf-8") as f:
                self.assertNotIn("from _bcr_config", f.read(), p)

    def tmp_dir(self):
        if not hasattr(self, "_tmp"):
            self._tmp = tempfile.mkdtemp(prefix="p0c-script-")
            self.addCleanup(shutil.rmtree, self._tmp, True)
        return self._tmp

    def test_verify_index_bad_db_exit_code_1(self):
        import subprocess
        db = os.path.join(self.tmp_dir(), "nope_db")
        env = dict(os.environ, PYTHONPATH=ROOT)
        env.pop("CHROMA_DIR", None)
        p = subprocess.run(
            [sys.executable, os.path.join(SCRIPTS, "verify_index.py"),
             "--db-dir", db, "--collection", "standards"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        self.assertEqual(p.returncode, 1, p.stdout + p.stderr)

    def test_verify_index_json_summary(self):
        import subprocess
        db = os.path.join(self.tmp_dir(), "nope_db2")
        p = subprocess.run(
            [sys.executable, os.path.join(SCRIPTS, "verify_index.py"),
             "--db-dir", db, "--json"],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
            env=dict(os.environ, PYTHONPATH=ROOT))
        self.assertEqual(p.returncode, 1)
        data = json.loads(p.stdout)
        self.assertFalse(data["ok"])
        self.assertEqual(data["exit_code"], 1)


class TestExitCode(unittest.TestCase):
    def test_all_pass(self):
        self.assertEqual(index_ops.summary_exit_code(
            [{"status": "PASS"}, {"status": "PASS"}]), 0)

    def test_any_fail(self):
        self.assertEqual(index_ops.summary_exit_code(
            [{"status": "PASS"}, {"status": "FAIL"}]), 1)

    def test_skip_counts_as_failure(self):
        self.assertEqual(index_ops.summary_exit_code(
            [{"status": "PASS"}, {"status": "SKIP"}]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
