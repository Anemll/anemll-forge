"""Select the temporary research instrumentation for precisely one standard Forge entry point."""
import os
import sys

if os.environ.get("FORGE_GATE_ARM") and sys.argv[0] == "/Users/anemll/Documents/Codex/ios-prefill-20261030/v8_work/pipe/v8_server.py":
    import importlib.abc
    import importlib.machinery
    import importlib.util
    from pathlib import Path

    class ResearchLoader(importlib.machinery.SourceFileLoader):
        def exec_module(self, module):
            super().exec_module(module)
            from research_hook import install
            install(module, os.environ["FORGE_GATE_ARM"])

    class ResearchFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "tree_verify":
                source = str(Path(sys.argv[0]).with_name("tree_verify.py"))
                loader = ResearchLoader(fullname, source)
                return importlib.util.spec_from_file_location(fullname, source, loader=loader)
            return None
    sys.meta_path.insert(0, ResearchFinder())
