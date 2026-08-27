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
    def test_status_distinguishes_sim_scope(self):
        s=app.local_status()
        self.assertEqual(s['default_gpu'],'L40S')
        self.assertEqual(s['default_task'],'CloseBlenderLid')
        self.assertEqual(s['policy_horizon'],100)
        self.assertIn('closed-loop RoboCasa simulation',s['scope'])
        self.assertFalse(s['assets_full_cache_default'])

    def test_random_plan(self):
        p=app.random_plan(app.parse_cli(['smoke-random','--steps','80','--seed','9','--gpu','L40S']))
        self.assertEqual(p['task'],'CloseBlenderLid')
        self.assertEqual(p['steps'],80)
        self.assertEqual(p['seed'],9)

    def test_policy_plan_exposes_hidden_remote_controls(self):
        p=app.policy_plan(app.parse_cli([
            'smoke-policy','--horizon','120','--attn','eager','--crop-ratio','0.9','--num-denoise-steps','7'
        ]))
        self.assertEqual(p['horizon'],120)
        self.assertEqual(p['attn'],'eager')
        self.assertEqual(p['crop_ratio'],0.9)
        self.assertEqual(p['num_denoise_steps'],7)

    def test_eval_plan(self):
        p=app.eval_plan(app.parse_cli([
            'eval-mini','--tasks','OpenDrawer,CloseFridge','--num-seeds','3','--seed','11',
            '--horizon','250','--long-horizon','600','--long-task','CloseBlenderLid',
            '--no-long','--no-save-every-video','--num-denoise-steps','6'
        ]))
        self.assertEqual(p['tasks'],['OpenDrawer','CloseFridge'])
        self.assertEqual(p['tasks_csv'],'OpenDrawer,CloseFridge')
        self.assertEqual(p['num_seeds'],3)
        self.assertEqual(p['base_seed'],11)
        self.assertFalse(p['run_long_track'])
        self.assertFalse(p['save_every_video'])
        self.assertEqual(p['num_denoise_steps'],6)

    def test_invalid_controls_fail_locally(self):
        with self.assertRaisesRegex(ValueError,'steps'):
            app.random_plan(app.parse_cli(['smoke-random','--steps','0']))
        with self.assertRaisesRegex(ValueError,'horizon'):
            app.policy_plan(app.parse_cli(['smoke-policy','--horizon','0']))
        with self.assertRaisesRegex(ValueError,'crop-ratio'):
            app.policy_plan(app.parse_cli(['smoke-policy','--crop-ratio','1.1']))
        with self.assertRaisesRegex(ValueError,'num-seeds'):
            app.eval_plan(app.parse_cli(['eval-mini','--num-seeds','0']))

    def test_remote_function_selection(self):
        self.assertIs(app._remote_fn(app.smoke_policy_fn,app.DEFAULT_GPU),app.smoke_policy_fn)

    def test_dry_run(self):
        self.assertTrue(app.parse_cli(['download-weights','--dry-run']).dry_run)
        assets = app.parse_cli(['download-assets','--full-cache','--dry-run'])
        self.assertTrue(assets.dry_run)
        self.assertTrue(assets.full_cache)
        self.assertTrue(app.parse_cli(['smoke-policy','--dry-run']).dry_run)
        self.assertTrue(app.parse_cli(['eval-mini','--dry-run']).dry_run)

if __name__=='__main__': unittest.main()
