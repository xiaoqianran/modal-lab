"""011 app.py 本地 CLI / planning 测试；不连接 Modal。"""
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
        self.assertEqual(s['default_gpu'],'L4')
        self.assertEqual(s['default_model'],'medium')
        self.assertIn('T4 unsupported',s['gpu_note'])

    def test_smoke_keeps_benchmark_invariants(self):
        p=app.generation_plan(app.parse_cli(['smoke','--duration','12','--seed','7']))
        self.assertEqual(p['prompt'],app.SMOKE_PROMPT)
        self.assertEqual(p['steps'],8)
        self.assertEqual(p['cfg_scale'],1.0)
        self.assertEqual(p['audio_format'],'flac')
        self.assertEqual(p['duration'],12.0)
        self.assertEqual(p['seed'],7)

    def test_t2a_controls(self):
        p=app.generation_plan(app.parse_cli([
            't2a','--prompt',' dreamy synthpop ','--negative-prompt','noise',
            '--duration','30','--steps','12','--cfg-scale','1.5','--seed','9','--format','wav'
        ]))
        self.assertEqual(p['prompt'],'dreamy synthpop')
        self.assertEqual(p['negative_prompt'],'noise')
        self.assertEqual(p['steps'],12)
        self.assertEqual(p['cfg_scale'],1.5)
        self.assertEqual(p['audio_format'],'wav')

    def test_invalid_positive_controls_fail_locally(self):
        with self.assertRaisesRegex(ValueError,'duration'):
            app.generation_plan(app.parse_cli(['smoke','--duration','0']))
        with self.assertRaisesRegex(ValueError,'steps'):
            app.generation_plan(app.parse_cli(['t2a','--prompt','x','--steps','0']))

    def test_list_outputs_is_domain_command(self):
        self.assertEqual(app.parse_cli(['list-outputs']).command,'list-outputs')

    def test_dry_run(self):
        self.assertTrue(app.parse_cli(['download','--dry-run']).dry_run)
        self.assertTrue(app.parse_cli(['smoke','--dry-run']).dry_run)

if __name__=='__main__': unittest.main()
