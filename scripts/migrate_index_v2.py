"""
migrate_index_v2.py — backfill schema-v2 metadata onto existing tree pickles.

`data/index/*.pkl` files built before the Node/Tree schema gained line spans,
heading paths, depth, token counts, etc. (see the restructuring plan) can be
upgraded WITHOUT a full re-ingest: the parser rules (heading detection,
level/title extraction, layer assignment) were not changed when span tracking
was added, only extended — so re-running `TreeBuilder.parse_markdown_to_hierarchical_json`
+ `_assign_json_layers` + `_collect_json_nodes_by_layer` against the same
source file reproduces the *exact same* node boundaries and index-assignment
order as the original ingest run, with zero LLM/embedding calls. This script
walks that reconstruction in lockstep with the existing pickle's node indices,
verifies the two agree (leaf text equality, parent title-prefix match) before
touching anything, and then applies the metadata in place.

Usage:
    uv run python scripts/migrate_index_v2.py                 # all data/index/*.pkl
    uv run python scripts/migrate_index_v2.py --file data/index/pldc.pkl

Writes `<stem>_v2.pkl` next to the original — never overwrites it, and skips
(reporting so) any pickle that's already schema_version >= 2 or has no
matching `data/src/<stem>.md` (e.g. an orphaned raptor_tree-era index).
"""

import argparse
import collections
import hashlib
import logging
import pickle
import sys
from pathlib import Path

# Installs the tree.*/raptor_tree.* -> slidegen_rag.tree.*/slidegen_rag.raptor_tree.*
# redirect finder as an import side effect (see tree/legacy_pickle.py) — required
# for these pre-src-move pickles to unpickle at all.
from slidegen_rag.tree import legacy_pickle  # noqa: F401
from slidegen_rag.tree.builder.tree_builder import TreeBuilder, TreeBuilderConfig
from slidegen_rag.tree.models.EmbeddingModels import EmbeddingModel
from slidegen_rag.tree.models.SummarizationModels import ChatNVIDIASummarizationModel
from slidegen_rag.tree.structure.tree_structure import Tree

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class MigrationError(Exception):
    """Raised when the re-parsed source disagrees with the existing pickle —
    they've drifted apart (source edited since ingest, or a parser change
    that isn't purely additive) and a real re-ingest is required instead."""


def _make_parse_only_builder() -> TreeBuilder:
    """A real TreeBuilder — constructing EmbeddingModel/ChatNVIDIASummarizationModel
    is cheap (no network call, no model load at __init__ time; both are lazy)
    — but only its pure parsing methods are used here, never create_node /
    batch_create_nodes / summarize / batch_summarize."""
    config = TreeBuilderConfig(
        summarization_model=ChatNVIDIASummarizationModel(),
        embedding_models={"SBERT": EmbeddingModel()},
    )
    return TreeBuilder(config)


def _flatten_layers(builder: TreeBuilder, root_jsons, layer_map: dict) -> None:
    """Populate layer_map exactly as TreeBuilder.build_from_text does, via
    _assign_json_layers + _collect_json_nodes_by_layer, so json nodes land in
    layer_map in the identical order used for the original next_index walk."""
    for r in root_jsons:
        builder._assign_json_layers(r)
    for r in root_jsons:
        builder._collect_json_nodes_by_layer(r, layer_map)


def migrate_one(pkl_path: Path, src_dir: Path, force: bool = False) -> bool:
    """Returns True if a _v2.pkl was written, False if skipped."""
    stem = pkl_path.stem
    if stem.endswith("_raptor"):
        logger.info(f"[SKIP] {pkl_path.name}: raptor_tree backend was removed, no migration path")
        return False

    md_path = src_dir / f"{stem}.md"
    if not md_path.exists():
        logger.warning(f"[SKIP] {pkl_path.name}: no matching source {md_path}")
        return False

    out_path = pkl_path.with_name(f"{stem}_v2.pkl")
    if out_path.exists() and not force:
        logger.info(f"[SKIP] {out_path.name} already exists (use --force to rebuild)")
        return False

    with open(pkl_path, "rb") as f:
        tree: Tree = pickle.load(f)
    if not isinstance(tree, Tree):
        raise MigrationError(f"{pkl_path}: unpickled object is not a Tree")

    if getattr(tree, "schema_version", 1) >= 2:
        logger.info(f"[SKIP] {pkl_path.name}: already schema_version={tree.schema_version}")
        return False

    logger.info(f"[START] Migrating {pkl_path.name} ({len(tree.all_nodes)} nodes)")

    markdown_text = md_path.read_text(encoding="utf-8", errors="ignore")
    source_text = markdown_text.strip()
    source_lines = source_text.split("\n")
    source_sha1 = hashlib.sha1(source_text.encode("utf-8")).hexdigest()

    builder = _make_parse_only_builder()
    preamble, root_jsons = builder.parse_markdown_to_hierarchical_json(markdown_text)

    if not root_jsons:
        # Degenerate no-heading case: exactly one node in both the old tree
        # and the reconstruction.
        if len(tree.all_nodes) != 1:
            raise MigrationError(
                f"{pkl_path}: source now parses as headingless (1 node) but "
                f"the pickle has {len(tree.all_nodes)} — source/index have diverged"
            )
        node = next(iter(tree.all_nodes.values()))
        node.doc_id = stem
        node.n_tokens = len(builder.tokenizer.encode(node.text))
        node.subtree_tokens = node.n_tokens
        node.line_start = 0
        node.line_end = len(source_lines)
        node.body_line_start = 0
        node.body_line_end = len(source_lines)
        node.kind = "root"
        tree.schema_version = 2
        tree.doc_id = stem
        tree.source_path = str(md_path)
        tree.source_sha1 = source_sha1
        tree.source_text = source_text
        tree.parent_of = {}
        with open(out_path, "wb") as f:
            pickle.dump(tree, f)
        logger.info(f"[DONE] {out_path.name}")
        return True

    layer_map: dict = collections.defaultdict(list)
    _flatten_layers(builder, root_jsons, layer_map)
    max_layer = max(layer_map.keys())

    # ── Pre-flight structural guard ──────────────────────────────────────
    reconstructed_layer_counts = {layer: len(nodes) for layer, nodes in layer_map.items()}
    existing_layer_counts = {
        layer: len(nodes) for layer, nodes in tree.layer_to_nodes.items()
        if layer <= max_layer
    }
    if reconstructed_layer_counts != existing_layer_counts:
        raise MigrationError(
            f"{pkl_path}: reconstructed layer node counts {reconstructed_layer_counts} "
            f"!= existing (up to layer {max_layer}) {existing_layer_counts} — "
            f"source and index have diverged; re-run ingest --force instead"
        )

    # ── Walk layers in the same next_index order as build_from_text ─────
    next_index = 0
    for layer in range(0, max_layer + 1):
        json_nodes = layer_map[layer]
        for json_node in json_nodes:
            if next_index not in tree.all_nodes:
                raise MigrationError(
                    f"{pkl_path}: expected node index {next_index} (layer {layer}) "
                    f"not present in pickle — source and index have diverged"
                )
            node_obj = tree.all_nodes[next_index]

            title = json_node.get("title", "").strip()
            content = json_node.get("content", "").strip()
            if layer == 0:
                expected_text = f"{title}\n{content}".strip() if title else content
                if node_obj.text != expected_text:
                    raise MigrationError(
                        f"{pkl_path}: leaf node {next_index} text mismatch — "
                        f"source and index have diverged (re-run ingest --force)"
                    )
            else:
                expected_prefix = f"{title}:" if title else ""
                if expected_prefix and not node_obj.text.startswith(expected_prefix):
                    raise MigrationError(
                        f"{pkl_path}: parent node {next_index} text does not start with "
                        f"{expected_prefix!r} — source and index have diverged (re-run ingest --force)"
                    )

            builder._apply_json_metadata(node_obj, json_node, stem, layer)
            next_index += 1

    # ── Super root (extra node beyond the json-covered layers) ──────────
    if next_index < len(tree.all_nodes):
        remaining = [i for i in tree.all_nodes if i >= next_index]
        if len(remaining) != 1:
            raise MigrationError(
                f"{pkl_path}: expected exactly one leftover super-root node, found {len(remaining)}"
            )
        super_idx = remaining[0]
        super_node = tree.all_nodes[super_idx]
        super_node.doc_id = stem
        super_node.layer = max_layer + 1
        super_node.depth = 0
        super_node.line_start = 0
        super_node.line_end = len(source_lines)
        super_node.n_tokens = len(builder.tokenizer.encode(super_node.text))

    # ── Final pass: parent pointers, subtree_tokens, root kind ───────────
    parent_of: dict = {}
    for node in tree.all_nodes.values():
        for child_idx in node.children:
            parent_of[child_idx] = node.index
            if child_idx in tree.all_nodes:
                tree.all_nodes[child_idx].parent = node.index

    for layer_idx in sorted(tree.layer_to_nodes.keys()):
        for node in tree.layer_to_nodes[layer_idx]:
            node.subtree_tokens = node.n_tokens + sum(
                tree.all_nodes[c].subtree_tokens for c in node.children if c in tree.all_nodes
            )

    final_root_node = next(iter(tree.root_nodes.values()))
    final_root_node.kind = "root"

    tree.schema_version = 2
    tree.doc_id = stem
    tree.source_path = str(md_path)
    tree.source_sha1 = source_sha1
    tree.source_text = source_text
    tree.parent_of = parent_of

    with open(out_path, "wb") as f:
        pickle.dump(tree, f)
    logger.info(f"[DONE] {out_path.name} ({len(tree.all_nodes)} nodes, schema_version=2)")
    return True


def main() -> None:
    from slidegen_rag.config import DATA_SRC_DIR, DATA_INDEX_DIR

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", type=str, default=None, help="Migrate one specific .pkl.")
    parser.add_argument("--index-dir", type=str, default=str(DATA_INDEX_DIR))
    parser.add_argument("--src-dir", type=str, default=str(DATA_SRC_DIR))
    parser.add_argument("--force", action="store_true", help="Rebuild _v2.pkl even if it exists.")
    args = parser.parse_args()

    index_dir = Path(args.index_dir)
    src_dir = Path(args.src_dir)

    pkl_files = [Path(args.file)] if args.file else sorted(index_dir.glob("*.pkl"))
    pkl_files = [p for p in pkl_files if not p.stem.endswith("_v2")]

    if not pkl_files:
        logger.warning(f"No .pkl files found in {index_dir}")
        return

    migrated, skipped = 0, 0
    for pkl_path in pkl_files:
        try:
            if migrate_one(pkl_path, src_dir, force=args.force):
                migrated += 1
            else:
                skipped += 1
        except MigrationError as e:
            logger.error(f"[ERROR] {e}")
            skipped += 1

    logger.info(f"Hoàn tất: {migrated} migrated, {skipped} skipped/errored")


if __name__ == "__main__":
    main()
