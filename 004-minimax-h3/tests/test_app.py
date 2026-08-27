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
    def local_entrypoint(self,*a,**k): return lambda fn: fn
def _decorator(*a,**k): return lambda fn: fn
sys.modules['modal']=types.SimpleNamespace(App=_App,Image=_Chain,Volume=_Chain,fastapi_endpoint=_decorator)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import app  # noqa:E402
class TestCli(unittest.TestCase):
    def test_status(self):
        s=app.local_status(); self.assertEqual(s['default_gpu'],'RTX-PRO-6000'); self.assertEqual(s['outputs_volume'],app.VOLUME_OUTPUTS_NAME)
    def test_default_t2v(self):
        p=app.t2v_plan(app.parse_cli(['t2v'])); self.assertEqual(p['prompt'],app.DEFAULT_PROMPT); self.assertEqual((p['width'],p['height']),(864,480)); self.assertEqual(p['seconds'],5.0); self.assertEqual(p['steps'],20)
    def test_t2v_controls(self):
        p=app.t2v_plan(app.parse_cli(['t2v','--prompt',' cinematic ','--width','640','--height','360','--seconds','3','--steps','12','--seed','7','--output-name','demo','--gpu','L40S']))
        self.assertEqual(p['prompt'],'cinematic'); self.assertEqual(p['output_name'],'demo'); self.assertEqual(p['gpu'],'L40S'); self.assertEqual(p['seed'],7)
    def test_invalid_sizes_fail(self):
        with self.assertRaises(ValueError): app.t2v_plan(app.parse_cli(['t2v','--width','0']))
        with self.assertRaises(ValueError): app.t2v_plan(app.parse_cli(['t2v','--seconds','0']))
        with self.assertRaises(ValueError): app.t2v_plan(app.parse_cli(['t2v','--steps','0']))
    def test_list_outputs_is_domain_command(self): self.assertEqual(app.parse_cli(['list-outputs']).command,'list-outputs')
    def test_dry_run(self): self.assertTrue(app.parse_cli(['download','--dry-run']).dry_run); self.assertTrue(app.parse_cli(['smoke','--dry-run']).dry_run); self.assertTrue(app.parse_cli(['t2v','--dry-run']).dry_run)
if __name__=='__main__': unittest.main()
