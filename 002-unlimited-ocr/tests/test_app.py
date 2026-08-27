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
class _App:
    def __init__(self,*a,**k): pass
    def function(self,*a,**k): return lambda fn: fn
    def cls(self,*a,**k): return lambda cls: cls
    def local_entrypoint(self,*a,**k): return lambda fn: fn
def _decorator(*a,**k): return lambda obj: obj
sys.modules['modal']=types.SimpleNamespace(App=_App,Image=_Chain,Volume=_Chain,enter=_decorator,exit=_decorator,method=_decorator)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import app  # noqa:E402
class TestCli(unittest.TestCase):
    def test_status_has_explicit_gpu_protocol(self):
        s=app.local_status(); self.assertEqual(s['default_gpu'],'H100!'); self.assertIn('with_options',s['gpu_selection'])
    def test_concurrency_parser(self):
        self.assertEqual(app._parse_concurrencies('16,24,32'),[16,24,32])
        with self.assertRaises(ValueError): app._parse_concurrencies('')
        with self.assertRaises(ValueError): app._parse_concurrencies('24,0')
        with self.assertRaises(ValueError): app._parse_concurrencies('x')
    def test_benchmark_plan(self):
        p=app.benchmark_plan(app.parse_cli(['benchmark','--seconds','300','--concurrencies','16,24,32','--gpu','RTX-PRO-6000']))
        self.assertEqual(p['concurrencies'],[16,24,32]); self.assertEqual(p['gpu'],'RTX-PRO-6000'); self.assertEqual(p['runtime_seconds'],300)
    def test_parse_plan(self):
        p=app.parse_plan(app.parse_cli(['parse','--start-page','10','--end-page','20','--concurrency','24','--no-resume']))
        self.assertEqual((p['start_page'],p['end_page']),(10,20)); self.assertEqual(p['concurrency'],24); self.assertFalse(p['resume'])
    def test_invalid_parse_controls(self):
        with self.assertRaises(ValueError): app.parse_plan(app.parse_cli(['parse','--start-page','0']))
        with self.assertRaises(ValueError): app.parse_plan(app.parse_cli(['parse','--start-page','10','--end-page','5']))
        with self.assertRaises(ValueError): app.parse_plan(app.parse_cli(['parse','--retries','-1']))
    def test_upload_is_explicit_local_boundary(self):
        a=app.parse_cli(['upload','--pdf','/tmp/book.pdf','--remote-pdf','/books/book.pdf']); self.assertEqual(a.pdf,Path('/tmp/book.pdf')); self.assertEqual(a.remote_pdf,'/books/book.pdf')
    def test_dry_run(self):
        self.assertTrue(app.parse_cli(['download','--dry-run']).dry_run); self.assertTrue(app.parse_cli(['benchmark','--dry-run']).dry_run); self.assertTrue(app.parse_cli(['parse','--dry-run']).dry_run)
if __name__=='__main__': unittest.main()
