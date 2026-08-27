"""006 app.py 本地 CLI / planning 测试；不连接 Modal。"""
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
        self.assertEqual(s['default_gpu'],'L4')
        self.assertEqual(s['smoke']['example'],'Bright_Room')
        self.assertEqual(s['smoke']['max_images'],2)

    def test_smoke_enforces_baseline(self):
        p=app.inference_plan(app.parse_cli(['smoke','--gpu','T4','--max-images','8']))
        self.assertEqual(p['example'],'Bright_Room')
        self.assertEqual(p['max_images'],2)
        self.assertEqual(p['target_size'],518)
        self.assertEqual(p['run_name'],'smoke_bright_room')
        self.assertFalse(p['save_gs'])

    def test_infer_controls(self):
        p=app.inference_plan(app.parse_cli([
            'infer','--example','Ireland_Landscape','--max-images','4',
            '--target-size','640','--gpu','L40S','--save-gs','--run-name','ireland4'
        ]))
        self.assertEqual(p['max_images'],4)
        self.assertEqual(p['target_size'],640)
        self.assertEqual(p['gpu'],'L40S')
        self.assertTrue(p['save_gs'])
        self.assertEqual(p['run_name'],'ireland4')

    def test_invalid_infer_sizes_fail_locally(self):
        with self.assertRaisesRegex(ValueError,'max-images'):
            app.inference_plan(app.parse_cli(['infer','--max-images','0']))
        with self.assertRaisesRegex(ValueError,'target-size'):
            app.inference_plan(app.parse_cli(['infer','--target-size','0']))

    def test_dry_run(self):
        self.assertTrue(app.parse_cli(['download','--dry-run']).dry_run)
        self.assertTrue(app.parse_cli(['smoke','--dry-run']).dry_run)

if __name__=='__main__': unittest.main()
