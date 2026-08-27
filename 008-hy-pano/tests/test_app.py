"""008 app.py 本地 CLI / planning 测试；不连接 Modal。"""
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
sys.modules['modal']=types.SimpleNamespace(App=_DummyApp,Image=_Chain,Volume=_Chain)
EXP_DIR=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXP_DIR))
import app  # noqa:E402

class TestCli(unittest.TestCase):
    def test_status(self):
        s=app.local_status()
        self.assertEqual(s['default_backend'],'qwen')
        self.assertEqual(s['default_gpu_qwen'],'RTX-PRO-6000')
        self.assertEqual(s['default_gpu_full'],'H100:4')

    def test_default_gpu_by_backend(self):
        self.assertEqual(app.default_gpu_for('qwen'),'RTX-PRO-6000')
        self.assertEqual(app.default_gpu_for('full'),'H100:4')

    def test_qwen_smoke_plan(self):
        p=app.inference_plan(app.parse_cli(['smoke','--steps','30']))
        self.assertEqual(p['gpu'],'RTX-PRO-6000')
        self.assertEqual(p['run_name'],'smoke_qwen')
        self.assertEqual(p['steps'],30)
        self.assertEqual(p['load_mode'],'gpu')

    def test_full_plan_uses_multi_gpu_and_taylor(self):
        p=app.inference_plan(app.parse_cli([
            'infer','--backend','full','--use-taylor-cache','--image','desk.jpg'
        ]))
        self.assertEqual(p['gpu'],'H100:4')
        self.assertTrue(p['use_taylor_cache'])
        self.assertEqual(p['run_name'],'')

    def test_explicit_gpu_wins(self):
        p=app.inference_plan(app.parse_cli([
            'smoke','--backend','qwen','--gpu','H100'
        ]))
        self.assertEqual(p['gpu'],'H100')

    def test_download_both(self):
        a=app.parse_cli(['download','--backend','both','--dry-run'])
        self.assertEqual(a.backend,'both')
        self.assertTrue(a.dry_run)

if __name__=='__main__': unittest.main()
