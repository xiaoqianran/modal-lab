"""015 app.py 本地 CLI / planning 测试；不连接 Modal。"""
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
        self.assertEqual(s['default_gpu'],'A100-40GB')
        self.assertEqual(s['robot_type'],'robocasa365')
        self.assertEqual(s['action_dim'],12)
        self.assertIn('not full RoboCasa365',s['scope'])

    def test_smoke_defaults(self):
        p=app.inference_plan(app.parse_cli(['smoke']))
        self.assertEqual(p['instruction'],'close the blender lid')
        self.assertEqual(p['run_name'],'smoke_close_blender_lid')
        self.assertEqual(p['attn'],'sdpa')
        self.assertEqual(p['num_steps'],5)
        self.assertEqual(p['obs_history'],4)

    def test_infer_controls(self):
        p=app.inference_plan(app.parse_cli([
            'infer','--instruction','open the drawer','--gpu','L40S','--attn','eager',
            '--num-steps','7','--obs-history','3','--run-name','drawer'
        ]))
        self.assertEqual(p['instruction'],'open the drawer')
        self.assertEqual(p['gpu'],'L40S')
        self.assertEqual(p['attn'],'eager')
        self.assertEqual(p['num_steps'],7)
        self.assertEqual(p['obs_history'],3)
        self.assertEqual(p['run_name'],'drawer')

    def test_invalid_positive_controls_fail_locally(self):
        with self.assertRaisesRegex(ValueError,'num-steps'):
            app.inference_plan(app.parse_cli(['smoke','--num-steps','0']))
        with self.assertRaisesRegex(ValueError,'obs-history'):
            app.inference_plan(app.parse_cli(['smoke','--obs-history','0']))

    def test_list_outputs_is_domain_command(self):
        self.assertEqual(app.parse_cli(['list-outputs']).command,'list-outputs')

    def test_dry_run(self):
        self.assertTrue(app.parse_cli(['download','--dry-run']).dry_run)
        self.assertTrue(app.parse_cli(['smoke','--dry-run']).dry_run)

if __name__=='__main__': unittest.main()
