"""027 app.py 的本地 CLI / planning 测试；不连接 Modal。"""
from __future__ import annotations
import sys, types, unittest
from pathlib import Path

class _Chain:
    @classmethod
    def from_name(cls, *args, **kwargs): return cls()
    @classmethod
    def debian_slim(cls, *args, **kwargs): return cls()
    def __getattr__(self, name): return lambda *args, **kwargs: self

class _DummyApp:
    def __init__(self, *args, **kwargs): pass
    def function(self, *args, **kwargs): return lambda fn: fn
    def local_entrypoint(self, *args, **kwargs): return lambda fn: fn

sys.modules['modal'] = types.SimpleNamespace(App=_DummyApp, Image=_Chain, Volume=_Chain)
EXP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXP_DIR))
import app  # noqa: E402

class TestCli(unittest.TestCase):
    def test_status(self):
        status = app.local_status()
        self.assertEqual(status['default_gpu'], 'L4')
        self.assertEqual(status['default_model'], 'custom_1.7')
        self.assertEqual(status['models']['design_1.7'], app.HF_REPOS['design_1.7'])

    def test_model_aliases_have_one_canonical_source(self):
        self.assertEqual(app._norm_model('design'), 'design_1.7')
        self.assertEqual(app._norm_model('clone'), 'base_1.7')
        self.assertEqual(app._norm_model('0.6b'), 'custom_0.6')

    def test_smoke_scenarios(self):
        expected = {
            'custom_zh': ('custom_1.7', 'Vivian', 'smoke_custom_zh_vivian'),
            'custom_en': ('custom_1.7', 'Ryan', 'smoke_custom_en_ryan'),
            'design': ('design_1.7', '', 'smoke_design_zh'),
            'clone': ('base_1.7', '', 'smoke_clone_en'),
        }
        for kind, values in expected.items():
            with self.subTest(kind=kind):
                plan = app.smoke_plan(app.parse_cli(['smoke', '--kind', kind]))
                self.assertEqual((plan['model'], plan['speaker'], plan['run_name']), values)

    def test_design_forces_design_model(self):
        plan = app.generation_plan(app.parse_cli([
            'design', '--text', '要抱抱', '--instruct', '撒娇萝莉女声'
        ]))
        self.assertEqual(plan['model'], 'design_1.7')
        self.assertEqual(plan['lang'], 'Chinese')
        self.assertEqual(plan['instruct'], '撒娇萝莉女声')

    def test_clone_has_official_default_reference(self):
        plan = app.generation_plan(app.parse_cli(['clone', '--text', 'Hello']))
        self.assertEqual(plan['model'], 'base_1.7')
        self.assertEqual(plan['ref_audio'], app.DEFAULT_CLONE_REF_URL)
        self.assertEqual(plan['ref_text'], app.DEFAULT_CLONE_REF_TEXT)

    def test_t2s_normalizes_language_and_model(self):
        plan = app.generation_plan(app.parse_cli([
            't2s', '--model', '0.6b', '--text', ' hello ', '--lang', 'en', '--speaker', 'Ryan'
        ]))
        self.assertEqual(plan['model'], 'custom_0.6')
        self.assertEqual(plan['text'], 'hello')
        self.assertEqual(plan['lang'], 'English')
        self.assertEqual(plan['speaker'], 'Ryan')

    def test_dry_run(self):
        self.assertTrue(app.parse_cli(['smoke', '--dry-run']).dry_run)
        self.assertTrue(app.parse_cli(['download', '--dry-run']).dry_run)
        self.assertTrue(app.parse_cli(['clone', '--text', 'hello', '--dry-run']).dry_run)

if __name__ == '__main__':
    unittest.main()
