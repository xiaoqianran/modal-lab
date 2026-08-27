"""016 app.py 本地 CLI / planning 测试；不连接 Modal。"""
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
        self.assertEqual(s['default_gpu'],'T4')
        self.assertEqual(s['default_model'],'small')
        self.assertEqual(s['license'],'CC-BY-NC 4.0')

    def test_model_normalization(self):
        self.assertEqual(app._norm_model('facebook/musicgen-medium'),'medium')
        self.assertEqual(app._norm_model('melody'),'melody')

    def test_smoke_plan(self):
        p=app.generation_plan(app.parse_cli([
            'smoke','--duration','10','--guidance-scale','2.5','--temperature','0.9'
        ]))
        self.assertEqual(p['prompt'],app.SMOKE_PROMPT)
        self.assertEqual(p['duration'],10.0)
        self.assertEqual(p['guidance_scale'],2.5)
        self.assertEqual(p['temperature'],0.9)

    def test_t2a_plan(self):
        p=app.generation_plan(app.parse_cli([
            't2a','--prompt',' jazz piano ','--model','medium','--duration','20','--seed','7'
        ]))
        self.assertEqual(p['prompt'],'jazz piano')
        self.assertEqual(p['model'],'medium')
        self.assertEqual(p['duration'],20.0)
        self.assertEqual(p['seed'],7)

    def test_dry_run(self):
        self.assertTrue(app.parse_cli(['download','--dry-run']).dry_run)
        self.assertTrue(app.parse_cli(['smoke','--dry-run']).dry_run)

if __name__=='__main__': unittest.main()
