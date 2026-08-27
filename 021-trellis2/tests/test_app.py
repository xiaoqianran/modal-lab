"""021 app.py 本地 CLI / planning 测试；不连接 Modal。"""
from __future__ import annotations
import sys, types, unittest
from pathlib import Path

class _Chain:
    @classmethod
    def from_name(cls,*a,**k): return cls()
    @classmethod
    def from_registry(cls,*a,**k): return cls()
    def __getattr__(self,n): return lambda *a,**k:self
class _DummyApp:
    def __init__(self,*a,**k): pass
    def cls(self,*a,**k): return lambda cls:cls
    def local_entrypoint(self,*a,**k): return lambda fn:fn
def _decorator(*a,**k): return lambda obj:obj
sys.modules['modal']=types.SimpleNamespace(
    App=_DummyApp,Image=_Chain,Volume=_Chain,enter=_decorator,method=_decorator,
)
EXP_DIR=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(EXP_DIR))
import app  # noqa:E402

class TestCli(unittest.TestCase):
    def test_status_uses_volume_names_not_mount_paths(self):
        s=app.local_status()
        self.assertEqual(s['wheels_volume'],app.VOLUME_WHEELS)
        self.assertEqual(s['weights_volume'],app.VOLUME_WEIGHTS)
        self.assertEqual(s['outputs_volume'],app.VOLUME_OUTPUTS)

    def test_gpu_worker_selection(self):
        self.assertFalse(app._is_pro6000('L40S'))
        self.assertTrue(app._is_pro6000('RTX-PRO-6000'))

    def test_download_force_is_not_lost(self):
        p=app.command_plan(app.parse_cli(['download','--dry-run','--force']))
        self.assertTrue(p['force'])

    def test_smoke_exposes_remote_quality_controls(self):
        p=app.command_plan(app.parse_cli([
            'smoke','--dry-run','--gpu','RTX-PRO-6000','--output-name','chair',
            '--pipeline-type','1024_cascade','--texture-size','1024','--decimation-target','200000'
        ]))
        self.assertEqual(p['output_name'],'chair')
        self.assertEqual(p['pipeline_type'],'1024_cascade')
        self.assertEqual(p['texture_size'],1024)
        self.assertEqual(p['decimation_target'],200000)

    def test_i2v_requires_explicit_image(self):
        p=app.command_plan(app.parse_cli([
            'i2v','--dry-run','--image-url','https://example.com/a.png'
        ]))
        self.assertEqual(p['image_url'],'https://example.com/a.png')

    def test_paid_generation_ack(self):
        with self.assertRaises(SystemExit):
            app.require_cost_ack(app.parse_cli(['smoke']))
        app.require_cost_ack(app.parse_cli(['smoke','--i-know-this-costs-money']))

if __name__=='__main__': unittest.main()
