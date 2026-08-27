"""028 app.py 本地 CLI / planning 测试；不连接 Modal。"""
from __future__ import annotations
import sys, types, unittest
from pathlib import Path

class _Chain:
    @classmethod
    def from_name(cls,*a,**k): return cls()
    @classmethod
    def debian_slim(cls,*a,**k): return cls()
    def __getattr__(self,n): return lambda *a,**k:self
class _DummyApp:
    def __init__(self,*a,**k): pass
    def function(self,*a,**k): return lambda fn:fn
    def local_entrypoint(self,*a,**k): return lambda fn:fn
sys.modules['modal']=types.SimpleNamespace(App=_DummyApp,Image=_Chain,Volume=_Chain)
EXP_DIR=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXP_DIR))
import app  # noqa:E402

class TestCli(unittest.TestCase):
    def test_status(self):
        s=app.local_status()
        self.assertEqual(s['default_gpu'],'L40S')
        self.assertEqual(s['default_model'],'s2-pro')
        self.assertIn('Research',s['license'])

    def test_clone_defaults(self):
        p=app.smoke_plan(app.parse_cli(['smoke','--kind','clone']))
        self.assertEqual(p['ref_audio'],app.DEFAULT_CLONE_URL)
        self.assertEqual(p['ref_text'],app.DEFAULT_CLONE_TEXT)
        self.assertEqual(p['run_name'],'smoke_clone_en')

    def test_explicit_clone_reference_wins(self):
        p=app.smoke_plan(app.parse_cli([
            'smoke','--kind','clone','--ref-audio','voice1','--ref-text','transcript'
        ]))
        self.assertEqual(p['ref_audio'],'voice1')
        self.assertEqual(p['ref_text'],'transcript')

    def test_tags_are_fixed_benchmark_text(self):
        p=app.smoke_plan(app.parse_cli(['smoke','--kind','tags']))
        self.assertEqual(p['text'],app.SMOKE_TAGS)
        self.assertIn('[excited]',p['text'])

    def test_generation_controls_are_not_lost(self):
        p=app.t2s_plan(app.parse_cli([
            't2s','--text',' hello ','--temperature','0.7','--top-p','0.9',
            '--repetition-penalty','1.2','--max-new-tokens','800','--chunk-length','160',
            '--seed','7','--compile'
        ]))
        self.assertEqual(p['text'],'hello')
        self.assertEqual(p['temperature'],0.7)
        self.assertEqual(p['top_p'],0.9)
        self.assertEqual(p['repetition_penalty'],1.2)
        self.assertEqual(p['max_new_tokens'],800)
        self.assertEqual(p['chunk_length'],160)
        self.assertEqual(p['seed'],7)
        self.assertTrue(p['compile'])

    def test_dry_run(self):
        self.assertTrue(app.parse_cli(['download','--dry-run']).dry_run)
        self.assertTrue(app.parse_cli(['smoke','--dry-run']).dry_run)

if __name__=='__main__': unittest.main()
