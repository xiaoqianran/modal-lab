"""032 app.py 的本地 CLI / planning 测试；不连接 Modal。"""
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
    def test_status_documents_prompt_boundary(self):
        s=app.local_status()
        self.assertEqual(s['prompt_wav'],'ref.wav')
        self.assertIn('modal volume put',s['prompt_note'])

    def test_zh_smoke(self):
        p=app.smoke_plan(app.parse_cli(['smoke']))
        self.assertEqual(p['run_name'],'smoke_zh')
        self.assertFalse(p['use_emo_text'])

    def test_emo_smoke(self):
        p=app.smoke_plan(app.parse_cli(['smoke','--kind','emo']))
        self.assertEqual(p['run_name'],'smoke_emo')
        self.assertEqual(p['emo_text'],'极度悲伤')
        self.assertTrue(p['use_emo_text'])

    def test_t2s_remote_prompt_and_emotion(self):
        p=app.t2s_plan(app.parse_cli([
            't2s','--text',' 你好 ','--emo-text','开心','--spk-audio','/prompts/ref.wav'
        ]))
        self.assertEqual(p['text'],'你好')
        self.assertEqual(p['spk_audio'],'/prompts/ref.wav')
        self.assertTrue(p['use_emo_text'])

    def test_dry_run(self):
        self.assertTrue(app.parse_cli(['smoke','--dry-run']).dry_run)
        self.assertTrue(app.parse_cli(['download','--dry-run']).dry_run)

if __name__=='__main__': unittest.main()
