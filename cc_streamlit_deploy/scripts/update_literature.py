"""Discover and stage v1.1 literature candidates; never auto-promote them."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.literature_discovery import LiteratureDiscoveryManager, TOPIC_QUERIES, write_discovery_audit
from src.literature_sources import (
    CORESource, CrossrefSource, NASANTRSSource, NCBISource, OpenAlexSource,
    OSTISource, UnpaywallSource, WebOfScienceSource,
)

SOURCE_CLASSES = {
    "openalex": OpenAlexSource, "crossref": CrossrefSource, "unpaywall": UnpaywallSource,
    "core": CORESource, "osti": OSTISource, "nasa_ntrs": NASANTRSSource,
    "ncbi": NCBISource, "web_of_science": WebOfScienceSource,
}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--topic", choices=["all", *TOPIC_QUERIES], default="all")
    value.add_argument("--since", default="")
    value.add_argument("--max-candidates", type=int, default=300)
    value.add_argument("--max-downloads", type=int, default=25)
    value.add_argument("--dry-run", action="store_true")
    value.add_argument("--sources", default="all", help="Comma-separated connector names or 'all'")
    value.add_argument("--output", type=Path, default=None)
    return value


def main() -> int:
    args = parser().parse_args()
    if args.since:
        datetime.strptime(args.since, "%Y-%m-%d")
    if not 1 <= args.max_candidates <= 2000:
        raise SystemExit("--max-candidates must be in [1, 2000]")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or ROOT / "outputs" / "literature_expansion_v1_1" / stamp
    if args.sources == "all":
        manager = LiteratureDiscoveryManager()
    else:
        names = [x.strip().casefold() for x in args.sources.split(",") if x.strip()]
        unknown = sorted(set(names) - set(SOURCE_CLASSES))
        if unknown:
            raise SystemExit("Unknown sources: " + ",".join(unknown))
        manager = LiteratureDiscoveryManager(sources=[SOURCE_CLASSES[name]() for name in names])
    rows = manager.discover(topic=args.topic, since=args.since, max_candidates=args.max_candidates)
    attempts = [] if args.dry_run else manager.stage_legal_pdfs(rows, ROOT / "paper" / "incoming_v1_1", max_downloads=args.max_downloads)
    summary = write_discovery_audit(output, rows, manager.source_status, attempts, dry_run=args.dry_run)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
