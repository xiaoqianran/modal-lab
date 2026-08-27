"""005-v3 app.py 的本地 CLI / planning 测试；不连接 Modal。"""
from __future__ import annotations
import sys, types, unittest
from pathlib import Path

class _Chain:
    @classmethod
    def from_name(cls, *args, **kwargs): return cls()
    @classmethod
    def from_registry(cls, *args, **kwargs): return cls()
    def __getattr__(self, name): return lambda *args, **kwargs: self

class _DummyApp:
    def __init__(self, *args, **kwargs): pass
    def cls(self, *args, **kwargs): return lambda cls: cls
    def local_entrypoint(self, *args, **kwargs): return lambda fn: fn

def _decorator(*args, **kwargs): return lambda obj: obj

sys.modules['modal'] = types.SimpleNamespace(
    App=_DummyApp, Image=_Chain, Volume=_Chain, enter=_decorator, method=_decorator,
)
EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_DIR))
import app  # noqa: E402

class TestCli(unittest.TestCase):
    def test_status(self):
        status = app.local_status()
        self.assertEqual(status['gpu'], 'RTX-PRO-6000')
        self.assertEqual(status['cuda_arch'], '12.0')
        self.assertEqual(status['torch'], '2.11.0+cu128')

    def test_probe_plan(self):
        self.assertEqual(
            app.command_plan(app.parse_cli(['probe', '--dry-run'])),
            {'action': 'probe', 'gpu': 'RTX-PRO-6000'},
        )

    def test_build_only(self):
        plan = app.command_plan(app.parse_cli(['build-sm120', '--dry-run', '--only', 'natten']))
        self.assertEqual(plan['only'], 'natten')

    def test_smoke_honors_output_name(self):
        plan = app.command_plan(app.parse_cli([
            'smoke', '--dry-run', '--output-name', 'custom_pro', '--resolution', '512'
        ]))
        self.assertEqual(plan['output_name'], 'custom_pro')
        self.assertEqual(plan['resolution'], 512)
        self.assertEqual(plan['image_url'], app.SAMPLE_IMAGE_URL)

    def test_i2v_custom_image_and_vram_mode(self):
        plan = app.command_plan(app.parse_cli([
            'i2v', '--dry-run', '--image-url', 'https://example.com/chair.png', '--no-low-vram'
        ]))
        self.assertEqual(plan['image_url'], 'https://example.com/chair.png')
        self.assertFalse(plan['low_vram'])

    def test_force_download(self):
        plan = app.command_plan(app.parse_cli(['download', '--dry-run', '--force']))
        self.assertTrue(plan['force'])

    def test_cost_ack(self):
        with self.assertRaises(SystemExit):
            app.require_cost_ack(app.parse_cli(['verify']))
        app.require_cost_ack(app.parse_cli(['verify', '--i-know-this-costs-money']))

if __name__ == '__main__':
    unittest.main()
