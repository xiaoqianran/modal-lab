"""010 app.py 本地 CLI / planning 测试；不连接 Modal。"""
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
        self.assertEqual(s['default_dit'],app.DEFAULT_DIT)
        self.assertFalse(s['smoke']['thinking'])

    def test_smoke_enforces_benchmark_invariants(self):
        p=app.generation_plan(app.parse_cli(['smoke','--duration','-1','--seed','7']))
        self.assertEqual(p['duration'],20.0)
        self.assertFalse(p['thinking'])
        self.assertFalse(p['init_lm'])
        self.assertTrue(p['instrumental'])
        self.assertEqual(p['inference_steps'],8)
        self.assertEqual(p['audio_format'],'flac')

    def test_t2m_controls(self):
        p=app.generation_plan(app.parse_cli([
            't2m','--caption','dreamy synthwave','--lyrics','[verse]\\nhello',
            '--duration','25','--bpm','120','--thinking','--init-lm','--vocal',
            '--steps','12','--format','wav','--dit','custom-dit','--lm','custom-lm'
        ]))
        self.assertEqual(p['lyrics'],'[verse]\nhello')
        self.assertEqual(p['bpm'],120)
        self.assertTrue(p['thinking'])
        self.assertTrue(p['init_lm'])
        self.assertFalse(p['instrumental'])
        self.assertEqual(p['inference_steps'],12)
        self.assertEqual(p['audio_format'],'wav')
        self.assertEqual(p['dit_model'],'custom-dit')
        self.assertEqual(p['lm_model'],'custom-lm')

    def test_invalid_t2m_controls_fail_locally(self):
        with self.assertRaisesRegex(ValueError,'duration'):
            app.generation_plan(app.parse_cli(['t2m','--duration','0']))
        with self.assertRaisesRegex(ValueError,'steps'):
            app.generation_plan(app.parse_cli(['t2m','--steps','0']))
        with self.assertRaisesRegex(ValueError,'bpm'):
            app.generation_plan(app.parse_cli(['t2m','--bpm','-1']))

    def test_list_outputs_is_domain_command(self):
        self.assertEqual(app.parse_cli(['list-outputs']).command,'list-outputs')

    def test_dry_run(self):
        self.assertTrue(app.parse_cli(['download','--dry-run']).dry_run)
        self.assertTrue(app.parse_cli(['smoke','--dry-run']).dry_run)

if __name__=='__main__': unittest.main()
