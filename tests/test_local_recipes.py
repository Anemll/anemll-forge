"""Checkout relocation and source-integrity checks; no research jobs run."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LocalRecipesTest(unittest.TestCase):
    def environment(self, work, python):
        env = {k: v for k, v in os.environ.items()
               if not k.startswith('FORGE_') and k not in
               {'MODEL', 'DRAFTER', 'WORK', 'WIKI', 'TRACE', 'OUT',
                'KL_TRACE', 'HEAD_EXPORT', 'REF_CODE'}}
        env.update(FORGE_WORK_DIR=str(work), FORGE_PYTHON=str(python),
                   FORGE_ROOT='/incorrect/former/checkout')
        return env

    @unittest.skipUnless(shutil.which('zsh'), 'Historical recipes require Zsh')
    def test_relocated_checkout_and_space_paths_from_another_cwd(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp).resolve()
            checkout = base / 'Forge checkout'
            (checkout / 'pipelines/m3u').mkdir(parents=True)
            shutil.copy2(ROOT / 'pipelines/common.zsh', checkout / 'pipelines/common.zsh')
            # Same source expression as the recipes, without their model/job commands.
            probe = checkout / 'pipelines/m3u/path_probe.sh'
            probe.write_text('source "${0:A:h}/../common.zsh" || exit $?\n'
                             'print -rl -- "$FORGE_ROOT" "$FORGE_PIPELINE_DIR" '
                             '"$MODEL" "$WIKI" "$OUT" "$REF_CODE"\n')
            fake_python = base / 'environment with spaces/python'
            fake_python.parent.mkdir()
            fake_python.write_text('#!/bin/sh\nexit 0\n')
            fake_python.chmod(0o755)
            work = base / 'work with spaces'
            env = self.environment(work, fake_python)
            env['FORGE_MODEL_DIR'] = str(base / 'BF16 weights')
            result = subprocess.run(['zsh', str(probe)], cwd=base, env=env,
                                    capture_output=True, text=True, check=True)
            self.assertEqual(result.stdout.splitlines(), [str(checkout),
                str(checkout / 'pipelines/m3u'), str(base / 'BF16 weights'),
                str(work / 'wikitext'), str(work / 'runs'),
                str(checkout / 'vendor/dflash_reference')])
            self.assertTrue((work / 'kl').is_dir())
            self.assertFalse((checkout / 'runs').exists())

    @unittest.skipUnless(shutil.which('zsh'), 'Historical recipes require Zsh')
    def test_missing_python_fails_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / 'unused work'
            result = subprocess.run(['zsh', '-c', 'source "$1"', 'probe',
                                     str(ROOT / 'pipelines/common.zsh')],
                                    env=self.environment(work, Path(tmp) / 'missing'),
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Missing Python:', result.stderr)
            self.assertFalse(work.exists())

    def test_pipeline_manifest_records_localized_bytes(self):
        folder = ROOT / 'pipelines/m3u'
        manifest = json.loads((folder / 'manifest.json').read_text())
        self.assertEqual({r['file'] for r in manifest['files']},
                         {p.name for p in folder.glob('*.sh')})
        for record in manifest['files']:
            digest = hashlib.sha256((folder / record['file']).read_bytes()).hexdigest()
            self.assertEqual(digest, record['localized_sha256'], record['file'])

    def test_vendored_reference_and_license_match_recorded_sources(self):
        vendor = ROOT / 'vendor/dflash_reference'
        manifest = json.loads((vendor / 'PROVENANCE.json').read_text())
        for record in manifest['files']:
            digest = hashlib.sha256((vendor / record['source']).read_bytes()).hexdigest()
            self.assertEqual(digest, record['sha256'], record['source'])
        self.assertIn('Copyright (c) 2026 Z Lab', (vendor / 'LICENSE').read_text())


if __name__ == '__main__':
    unittest.main()
