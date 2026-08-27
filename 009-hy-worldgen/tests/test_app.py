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
sys.modules['modal']=types.SimpleNamespace(App=_App,Image=_Chain,Volume=_Chain)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import app  # noqa:E402
class TestCli(unittest.TestCase):
    def test_status(self):
        s=app.status_payload(); self.assertEqual(s['default_gpu'],'RTX-PRO-6000'); self.assertEqual(s['vlm']['modes'],['share','split']); self.assertIn('modal run --detach',s['detach'])
    def test_stage12_defaults(self):
        p=app.stage_plan(app.parse_cli(['stage12','--dry-run'])); self.assertEqual(p['nframe'],16); self.assertEqual(p['vlm_mode'],'share'); self.assertTrue(p['force_vlm']); self.assertFalse(p['apply_recon_iteration'])
    def test_stage_controls(self):
        p=app.stage_plan(app.parse_cli(['stage','3','--dry-run','--gpu','H100:2','--nframe','24','--vlm-mode','split','--max-steps','5000','--keep-vlm']))
        self.assertEqual(p['stage'],3); self.assertEqual(p['gpu'],'H100:2'); self.assertEqual(p['nframe'],24); self.assertEqual(p['vlm_mode'],'split'); self.assertEqual(p['max_steps'],5000); self.assertTrue(p['keep_vlm'])
    def test_smoke_pipeline(self):
        p=app.stage_plan(app.parse_cli(['smoke','--dry-run'])); self.assertEqual(p['pipeline'][0],'prepare'); self.assertEqual(p['pipeline'][-1],'stage5'); self.assertEqual(p['max_steps'],4000)
    def test_invalid_common_controls(self):
        with self.assertRaises(ValueError): app.stage_plan(app.parse_cli(['stage12','--nframe','0']))
        with self.assertRaises(ValueError): app.stage_plan(app.parse_cli(['stage12','--vlm-mem-util','1']))
        with self.assertRaises(ValueError): app.stage_plan(app.parse_cli(['stage12','--vlm-max-model-len','0']))
    def test_download_does_not_duplicate_alias_schema(self):
        a=app.parse_cli(['download','--which','qwen','--dry-run']); self.assertEqual(a.which,'qwen'); self.assertTrue(a.dry_run)
    def test_dry_run(self): self.assertTrue(app.parse_cli(['prepare','--dry-run']).dry_run); self.assertTrue(app.parse_cli(['stage12','--dry-run']).dry_run)
if __name__=='__main__': unittest.main()
