"""013 app.py 本地 CLI / planning 测试；不连接 Modal。"""
from __future__ import annotations
import sys, types, unittest, tempfile
from pathlib import Path

class _Chain:
    @classmethod
    def from_name(cls,*a,**k): return cls()
    @classmethod
    def debian_slim(cls,*a,**k): return cls()
    @classmethod
    def from_registry(cls,*a,**k): return cls()
    def __getattr__(self,n): return lambda *a,**k:self
class _DummyApp:
    def __init__(self,*a,**k): pass
    def function(self,*a,**k): return lambda fn:fn
    def local_entrypoint(self,*a,**k): return lambda fn:fn
sys.modules['modal']=types.SimpleNamespace(App=_DummyApp,Image=_Chain,Volume=_Chain,Secret=_Chain)
EXP_DIR=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXP_DIR))
import app  # noqa:E402

class TestCli(unittest.TestCase):
    def test_status(self):
        s=app.local_status()
        self.assertEqual(s['default_gpu'],'L40S')
        self.assertEqual(s['default_stage1'],'en-cot')
        self.assertEqual(s['license'],'Apache-2.0')

    def test_stage1_aliases(self):
        self.assertEqual(app._stage1_id('zh-cot'),'m-a-p/YuE-s1-7B-anneal-zh-cot')
        self.assertEqual(app._stage1_id('en-icl'),'m-a-p/YuE-s1-7B-anneal-en-icl')

    def test_smoke_plan_keeps_generation_controls(self):
        p=app.generation_plan(app.parse_cli([
            'smoke','--run-n-segments','3','--max-new-tokens','2500',
            '--stage2-batch-size','1','--seed','7','--repetition-penalty','1.2'
        ]))
        self.assertEqual(p['lyrics'],app.SMOKE_LYRICS)
        self.assertEqual(p['run_n_segments'],3)
        self.assertEqual(p['max_new_tokens'],2500)
        self.assertEqual(p['stage2_batch_size'],1)
        self.assertEqual(p['seed'],7)
        self.assertEqual(p['repetition_penalty'],1.2)

    def test_generate_reads_local_lyrics_file(self):
        with tempfile.NamedTemporaryFile('w',encoding='utf-8',delete=False) as f:
            f.write('[verse]\nhello world\n')
            path=Path(f.name)
        try:
            p=app.generation_plan(app.parse_cli([
                'generate','--genre','pop','--lyrics-file',str(path)
            ]))
            self.assertEqual(p['lyrics'],'[verse]\nhello world')
            self.assertEqual(p['genre'],'pop')
        finally:
            path.unlink(missing_ok=True)

    def test_inline_lyrics_unescapes_newlines(self):
        p=app.generation_plan(app.parse_cli([
            'generate','--genre','rock','--lyrics','[verse]\\nhello'
        ]))
        self.assertEqual(p['lyrics'],'[verse]\nhello')

    def test_invalid_positive_controls_fail_locally(self):
        with self.assertRaisesRegex(ValueError,'run-n-segments'):
            app.generation_plan(app.parse_cli(['smoke','--run-n-segments','0']))
        with self.assertRaisesRegex(ValueError,'max-new-tokens'):
            app.generation_plan(app.parse_cli(['smoke','--max-new-tokens','0']))
        with self.assertRaisesRegex(ValueError,'stage2-batch-size'):
            app.generation_plan(app.parse_cli(['smoke','--stage2-batch-size','0']))

    def test_dry_run(self):
        self.assertTrue(app.parse_cli(['download','--dry-run']).dry_run)
        self.assertTrue(app.parse_cli(['smoke','--dry-run']).dry_run)

if __name__=='__main__': unittest.main()
