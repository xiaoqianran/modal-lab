"""030 app.py 的本地 CLI / planning 测试；不连接 Modal。"""
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
        self.assertEqual(s['default_speaker'],'Carter')
        self.assertIn('Emma',s['voice_presets'])
    def test_en_smoke(self):
        p=app.smoke_plan(app.parse_cli(['smoke']))
        self.assertEqual((p['speaker'],p['run_name']),('Carter','smoke_en'))
    def test_long_defaults_emma(self):
        p=app.smoke_plan(app.parse_cli(['smoke','--kind','long']))
        self.assertEqual(p['speaker'],'Emma')
        self.assertEqual(p['run_name'],'smoke_long')
    def test_emma_forces_emma(self):
        p=app.smoke_plan(app.parse_cli(['smoke','--kind','emma','--speaker','Carter']))
        self.assertEqual(p['speaker'],'Emma')
    def test_t2s_exposes_ddpm_steps(self):
        p=app.t2s_plan(app.parse_cli(['t2s','--text',' hi ','--speaker','Grace','--ddpm-steps','7']))
        self.assertEqual(p['text'],'hi')
        self.assertEqual(p['speaker'],'Grace')
        self.assertEqual(p['ddpm_steps'],7)
    def test_dry_run(self):
        self.assertTrue(app.parse_cli(['smoke','--dry-run']).dry_run)
        self.assertTrue(app.parse_cli(['download','--dry-run']).dry_run)

if __name__=='__main__': unittest.main()
