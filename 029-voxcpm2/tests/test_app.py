"""029 app.py 本地 CLI / planning 测试；不连接 Modal。"""
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
        self.assertEqual(s['default_gpu'],'L4')
        self.assertEqual(s['hf_repo'],'openbmb/VoxCPM2')
        self.assertEqual(s['default_clone_reference'],'reference_speaker')

    def test_clone_smoke_uses_downloaded_reference(self):
        p=app.smoke_plan(app.parse_cli(['smoke','--kind','clone']))
        self.assertEqual(p['reference_wav'],'reference_speaker')
        self.assertEqual(p['run_name'],'smoke_clone')

    def test_design_is_text_level_voice_description(self):
        p=app.smoke_plan(app.parse_cli(['smoke','--kind','design']))
        self.assertEqual(p['reference_wav'],'')
        self.assertTrue(p['text'].startswith('(A young woman'))

    def test_generation_controls_are_not_lost(self):
        p=app.t2s_plan(app.parse_cli([
            't2s','--text',' hello ','--cfg-value','1.7','--timesteps','8','--seed','7','--optimize',
            '--prompt-wav','my_prompt','--prompt-text','reference text'
        ]))
        self.assertEqual(p['text'],'hello')
        self.assertEqual(p['cfg_value'],1.7)
        self.assertEqual(p['inference_timesteps'],8)
        self.assertEqual(p['seed'],7)
        self.assertTrue(p['optimize'])
        self.assertEqual(p['prompt_wav'],'my_prompt')

    def test_explicit_clone_reference_wins(self):
        p=app.smoke_plan(app.parse_cli([
            'smoke','--kind','clone','--reference-wav','custom_voice'
        ]))
        self.assertEqual(p['reference_wav'],'custom_voice')

    def test_dry_run(self):
        self.assertTrue(app.parse_cli(['download','--dry-run']).dry_run)
        self.assertTrue(app.parse_cli(['smoke','--dry-run']).dry_run)
        self.assertTrue(app.parse_cli(['t2s','--text','hi','--dry-run']).dry_run)

if __name__=='__main__': unittest.main()
