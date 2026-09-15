"""
Shim so index pickles built before the src/ layout move still unpickle.

`data/index/*.pkl` files were written by `pickle.dump(tree)` while `Node`,
`Tree`, `BuildStats`, `LayerStat`, etc. lived at the top-level module paths
`tree.*` and `raptor_tree.*`. Moving those packages under `slidegen_rag.*`
changes their dotted module path, and pickle resolves each class by
(module path, qualname) at load time — so without this shim, every existing
`data/index/*.pkl` fails to unpickle with a ModuleNotFoundError, and *which*
submodule it fails on depends on which fields happen to be non-None
(e.g. `Tree.build_stats` pulls in `tree.stats.BuildStats` only when it was
actually set at build time).

A naive fix — registering a bare package stub whose `__path__` points at the
real on-disk directory — looks tempting (no need to enumerate every
submodule a pickle might reference) but is wrong: Python's import system
would then *re-execute* e.g. `tree/structure/tree_structure.py` a second
time under the name `tree.structure.tree_structure`, producing a `Tree` class
that is a distinct object from `slidegen_rag.tree.structure.tree_structure.Tree`
— structurally identical, but `isinstance(loaded_tree, Tree)` (used by both
`RetrievalAugmentation.__init__` implementations) then fails.

Instead this installs a `sys.meta_path` finder that redirects any import of
`tree` / `tree.<anything>` (and `raptor_tree` likewise) to the *same* module
object already imported as `slidegen_rag.tree.<anything>` — so old and new
dotted paths refer to the exact same class objects, and `isinstance` checks
succeed regardless of which name pickle used to resolve them. This also
requires no enumeration: it covers every present and future submodule.

Call `install_legacy_aliases()` once, before any `pickle.load` of an old
index file — both `RetrievalAugmentation` implementations import this module
at the top for that reason. `data/index/` is gitignored, so this only
matters for locally-built indices; it costs nothing and is safe to call more
than once.
"""

import importlib
import sys
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec

_OLD_TO_NEW_PREFIX = {
    "tree": "slidegen_rag.tree",
    # Vết tích: backend raptor_tree đã bị gỡ và slidegen_rag.raptor_tree KHÔNG
    # còn tồn tại, nên .pkl thời đó không mở lại được (main.py bỏ qua *_raptor.pkl,
    # scripts/migrate_index_v2.py cũng báo "no migration path").
    "raptor_tree": "slidegen_rag.raptor_tree",
}


class _LegacyPrefixRedirectFinder(MetaPathFinder, Loader):
    """Redirects `<old>` / `<old>.<rest>` imports to the identical module
    object already (or about to be) imported as `<new>` / `<new>.<rest>`."""

    def find_spec(self, fullname, path, target=None):
        for old_prefix in _OLD_TO_NEW_PREFIX:
            if fullname == old_prefix or fullname.startswith(old_prefix + "."):
                return ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        for old_prefix, new_prefix in _OLD_TO_NEW_PREFIX.items():
            if spec.name == old_prefix or spec.name.startswith(old_prefix + "."):
                new_name = new_prefix + spec.name[len(old_prefix):]
                return importlib.import_module(new_name)
        raise ImportError(spec.name)  # pragma: no cover — find_spec already filtered this

    def exec_module(self, module):
        pass  # already fully initialized by the import_module call above


_FINDER_INSTALLED = False


def install_legacy_aliases() -> None:
    """Install the `tree.*` / `raptor_tree.*` -> `slidegen_rag.*` redirect
    finder, idempotently. Safe to call more than once."""
    global _FINDER_INSTALLED
    if _FINDER_INSTALLED:
        return
    sys.meta_path.insert(0, _LegacyPrefixRedirectFinder())
    _FINDER_INSTALLED = True


install_legacy_aliases()
