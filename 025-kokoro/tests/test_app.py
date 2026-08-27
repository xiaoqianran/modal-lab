"""025 app.py 本地 CLI / planning 测试；不连接 Modal。"""
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
        self.assertEqual(s['default_gpu'],'T4')
        self.assertEqual(s['default_voice'],'af_heart')
        self.assertEqual(s['sample_rate'],24000)

    def test_model_aliases(self):
        self.assertEqual(app._norm_model('v1'),'v1')
        self.assertEqual(app._norm_model('zh'),'v1.1-zh')

    def test_zh_smoke_switches_model_voice_and_lang(self):
        p=app.smoke_plan(app.parse_cli(['smoke','--lang','zh']))
        self.assertEqual(p['model'],'v1.1-zh')
        self.assertEqual(p['voice'],'zf_001')
        self.assertEqual(p['lang'],'z')
        self.assertEqual(p['run_name'],'smoke_zh')

    def test_explicit_zh_voice_wins(self):
        p=app.smoke_plan(app.parse_cli(['smoke','--lang','zh','--voice','zm_010']))
        self.assertEqual(p['voice'],'zm_010')

    def test_t2s_controls_survive(self):
        p=app.t2s_plan(app.parse_cli([
            't2s','--text',' hello ','--voice','af_bella','--lang','a','--speed','0.9'
        ]))
        self.assertEqual(p['text'],'hello')
        self.assertEqual(p['voice'],'af_bella')
        self.assertEqual(p['lang'],'a')
        self.assertEqual(p['speed'],0.9)

    def test_voices_is_real_domain_command(self):
        a=app.parse_cli(['voices','--model','v1','--dry-run'])
        self.assertEqual(a.command,'voices')
        self.assertTrue(a.dry_run)

if __name__=='__main__': unittest.main()
