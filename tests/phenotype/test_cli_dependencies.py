from __future__ import annotations

import os
import subprocess
import sys
import textwrap


def test_phenotype_help_does_not_require_rajlab_utils():
    code = textwrap.dedent("""
        import importlib.abc
        import sys

        class BlockInternalPackage(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'rajlab_utils' or fullname.startswith('rajlab_utils.'):
                    raise ModuleNotFoundError('Internal rajlab_utils package is unavailable')

        sys.meta_path.insert(0, BlockInternalPackage())
        from forgephast.cli import main
        sys.argv = ['forgephast', 'phenotype', '--help']
        main()
    """)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "usage:" in result.stdout
