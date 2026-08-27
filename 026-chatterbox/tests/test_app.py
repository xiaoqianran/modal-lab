"""026 app.py 本地 CLI / planning 测试；不连接 Modal。"""
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
        self.assertEqual(s['default_model'],'multilingual')
        self.assertEqual(s['default_voice'],'Lucy')
        self.assertIn('modal volume put',s['prompt_note'])

    def test_model_aliases(self):
        self.assertEqual(app._norm_model('mtl'),'multilingual')
        self.assertEqual(app._norm_model('nano'),'turbo')
        self.assertEqual(app._norm_model('english'),'original')

    def test_multilingual_smoke_does_not_invent_voice(self):
        p=app.smoke_plan(app.parse_cli(['smoke','--kind','mtl_zh']))
        self.assertEqual(p['model'],'multilingual')
        self.assertEqual(p['lang'],'zh')
        self.assertEqual(p['voice'],'')
        self.assertFalse(p['nano'])

    def test_turbo_smoke_owns_prompt_voice_and_nano(self):
        p=app.smoke_plan(app.parse_cli(['smoke','--kind','turbo','--voice','Lucy','--nano']))
        self.assertEqual(p['model'],'turbo')
        self.assertEqual(p['voice'],'Lucy')
        self.assertTrue(p['nano'])
        self.assertEqual(p['run_name'],'smoke_turbo_lucy')

    def test_nano_is_ignored_for_non_turbo(self):
        p=app.t2s_plan(app.parse_cli(['t2s','--model','original','--text','hello','--nano']))
        self.assertEqual(p['model'],'original')
        self.assertFalse(p['nano'])

    def test_original_controls_survive(self):
        p=app.t2s_plan(app.parse_cli([
            't2s','--model','original','--text',' hello ','--exaggeration','0.7','--cfg-weight','0.4'
        ]))
        self.assertEqual(p['text'],'hello')
        self.assertEqual(p['exaggeration'],0.7)
        self.assertEqual(p['cfg_weight'],0.4)

    def test_dry_run(self):
        self.assertTrue(app.parse_cli(['download','--dry-run']).dry_run)
        self.assertTrue(app.parse_cli(['smoke','--dry-run']).dry_run)

if __name__=='__main__': unittest.main()
