from __future__ import annotations

import contextlib
import io
import unittest
import main


class TestRootDispatcher(unittest.TestCase):
    def test_entry_archetypes(self):
        self.assertEqual(main.entry_for(main.ROOT / '001-longcat-video').name, 'app.py')
        self.assertEqual(main.entry_for(main.ROOT / '040-modal-2d-provider').name, 'run.py')

    def test_unique_numeric_prefix(self):
        self.assertEqual(main.resolve_exp_id('001'), '001-longcat-video')
        self.assertEqual(main.resolve_exp_id('040'), '040-modal-2d-provider')

    def test_ambiguous_numeric_prefix(self):
        with self.assertRaisesRegex(SystemExit, 'ambiguous experiment'):
            main.resolve_exp_id('005')

    def test_list_is_root_command(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            handled = main.handle_root_cli(['--list'])
        self.assertTrue(handled)
        names = out.getvalue().splitlines()
        self.assertIn('001-longcat-video', names)
        self.assertIn('040-modal-2d-provider', names)
        self.assertEqual(names, main.list_experiments())

    def test_no_args_show_root_help(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            handled = main.handle_root_cli([])
        self.assertTrue(handled)
        self.assertIn('workspace launcher', out.getvalue())

    def test_help_is_root_command(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            handled = main.handle_root_cli(['--help'])
        self.assertTrue(handled)
        self.assertIn('workspace launcher', out.getvalue())
        self.assertIn('--list', out.getvalue())

    def test_experiment_args_are_not_swallowed(self):
        self.assertFalse(main.handle_root_cli(['001', 'status']))
        self.assertFalse(main.handle_root_cli(['001', '--help']))

    def test_default_experiment_shorthand_still_works(self):
        entry, rest, exp_id = main.resolve_experiment(['status'])
        self.assertEqual(exp_id, main.DEFAULT_EXP)
        self.assertEqual(entry.name, 'app.py')
        self.assertEqual(rest, ['status'])

    def test_provider_scripts_are_self_describing(self):
        self.assertTrue(main.has_inline_script_metadata(main.ROOT / '040-modal-2d-provider' / 'run.py'))
        self.assertTrue(main.has_inline_script_metadata(main.ROOT / '041-modal-3d-provider' / 'run.py'))

    def test_local_invocation_detection(self):
        self.assertTrue(main.is_local_invocation(['status']))
        self.assertTrue(main.is_local_invocation(['smoke', '--dry-run']))
        self.assertTrue(main.is_local_invocation(['--help']))
        self.assertFalse(main.is_local_invocation(['smoke']))


if __name__ == '__main__':
    unittest.main()
