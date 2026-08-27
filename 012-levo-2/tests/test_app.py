"""012 app.py 本地 CLI / planning 测试；不连接 Modal。"""
from __future__ import annotations
import sys, types, unittest
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
        self.assertEqual(s['default_model'],'v2-medium')
        self.assertEqual(s['runtime_repo'],app.HF_RUNTIME)

    def test_model_aliases(self):
        self.assertEqual(app._norm_model('medium'),'v2-medium')
        self.assertEqual(app._norm_model('large'),'v2-large')

    def test_large_on_small_gpu_auto_low_mem(self):
        p=app.generation_plan(app.parse_cli(['smoke','--model','v2-large','--gpu','L4']))
        self.assertTrue(p['low_mem'])
        self.assertEqual(p['model'],'v2-large')

    def test_large_on_big_gpu_not_forced_low_mem(self):
        p=app.generation_plan(app.parse_cli(['smoke','--model','v2-large','--gpu','RTX-PRO-6000']))
        self.assertFalse(p['low_mem'])

    def test_t2a_preserves_generation_controls(self):
        p=app.generation_plan(app.parse_cli([
            't2a','--lyrics','[verse] hello','--descriptions','male rock',
            '--generate-type','separate','--no-flash','--idx','song1'
        ]))
        self.assertEqual(p['lyrics_item']['gt_lyric'],'[verse] hello')
        self.assertEqual(p['lyrics_item']['descriptions'],'male rock')
        self.assertEqual(p['lyrics_item']['idx'],'song1')
        self.assertFalse(p['use_flash_attn'])
        self.assertEqual(p['generate_type'],'separate')
        self.assertEqual(p['run_name'],'t2a_song1')

    def test_dry_run(self):
        self.assertTrue(app.parse_cli(['download','--dry-run']).dry_run)
        self.assertTrue(app.parse_cli(['smoke','--dry-run']).dry_run)

if __name__=='__main__': unittest.main()
