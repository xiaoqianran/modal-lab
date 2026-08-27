"""014 app.py 本地 CLI / planning 测试；不连接 Modal。"""
from __future__ import annotations
import sys, types, unittest, tempfile
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
        self.assertEqual(s['model'],'ASLP-lab/DiffRhythm2')
        self.assertEqual(s['smoke']['steps'],16)

    def test_smoke_plan(self):
        p=app.generation_plan(app.parse_cli(['smoke','--steps','12','--cfg-strength','1.8']))
        self.assertEqual(p['lyrics'],app.SMOKE_LYRICS)
        self.assertEqual(p['steps'],12)
        self.assertEqual(p['cfg_strength'],1.8)

    def test_inline_lyrics_unescapes_newlines(self):
        p=app.generation_plan(app.parse_cli([
            'generate','--lyrics','[verse]\\nhello','--style','Pop'
        ]))
        self.assertEqual(p['lyrics'],'[verse]\nhello')
        self.assertEqual(p['style_prompt'],'Pop')

    def test_lyrics_file_is_local_input_boundary(self):
        with tempfile.NamedTemporaryFile('w',encoding='utf-8',delete=False) as f:
            f.write('[verse]\nfrom file\n')
            path=Path(f.name)
        try:
            p=app.generation_plan(app.parse_cli([
                'generate','--lyrics-file',str(path),'--style','Piano'
            ]))
            self.assertEqual(p['lyrics'],'[verse]\nfrom file')
        finally:
            path.unlink(missing_ok=True)

    def test_dry_run(self):
        self.assertTrue(app.parse_cli(['download','--dry-run']).dry_run)
        self.assertTrue(app.parse_cli(['smoke','--dry-run']).dry_run)

if __name__=='__main__': unittest.main()
