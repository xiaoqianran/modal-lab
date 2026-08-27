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
class _App:
    def __init__(self,*a,**k): pass
    def function(self,*a,**k): return lambda fn: fn
    def local_entrypoint(self,*a,**k): return lambda fn: fn
def _decorator(*a,**k): return lambda fn: fn
sys.modules['modal']=types.SimpleNamespace(App=_App,Image=_Chain,Volume=_Chain,fastapi_endpoint=_decorator)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import app  # noqa:E402
class TestCli(unittest.TestCase):
    def test_status(self):
        s=app.local_status(); self.assertEqual(s['default_gpu'],'H100'); self.assertEqual(s['hf_model'],'TencentARC/Pixal3D')
    def test_gpu_normalization(self):
        self.assertEqual(app._normalize_gpu('A100-80GB'),'A100-80GB'); self.assertEqual(app._normalize_gpu('l40s'),'L40S'); self.assertEqual(app._normalize_gpu('pro6000'),'RTX-PRO-6000')
    def test_default_i2v_source(self):
        p=app.i2v_plan(app.parse_cli(['i2v'])); self.assertTrue(p['local_image'] or p['image_url']); self.assertEqual(p['resolution'],1024); self.assertTrue(p['low_vram'])
    def test_local_image_boundary(self):
        with tempfile.NamedTemporaryFile('wb',suffix='.webp',delete=False) as f:
            f.write(b'x'); path=Path(f.name)
        try:
            p=app.i2v_plan(app.parse_cli(['i2v','--image',str(path),'--full-vram','--resolution','1536','--output-name','chair']))
            self.assertEqual(p['local_image'],str(path.resolve())); self.assertFalse(p['low_vram']); self.assertEqual(p['resolution'],1536); self.assertEqual(p['output_name'],'chair')
        finally: path.unlink(missing_ok=True)
    def test_image_and_url_are_exclusive(self):
        with tempfile.NamedTemporaryFile('wb',delete=False) as f: path=Path(f.name)
        try:
            with self.assertRaisesRegex(ValueError,'二选一'): app.i2v_plan(app.parse_cli(['i2v','--image',str(path),'--image-url','https://example.com/a.webp']))
        finally: path.unlink(missing_ok=True)
    def test_list_outputs_is_domain_command(self): self.assertEqual(app.parse_cli(['list-outputs']).command,'list-outputs')
    def test_dry_run(self): self.assertTrue(app.parse_cli(['download','--dry-run']).dry_run); self.assertTrue(app.parse_cli(['build-natten','--dry-run']).dry_run); self.assertTrue(app.parse_cli(['i2v','--dry-run']).dry_run)
if __name__=='__main__': unittest.main()
