"""Collect final measurements and historical attempts; never delete source files."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from io import StringIO
import json
from pathlib import Path
import sys
import zipfile

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from cbrs_quant.config import CPU_MODES
from cbrs_quant.results_store import export_data, load_data, seal
from cbrs_quant.rt_report import load_run, sanitize


def rows(data):
    return list(csv.DictReader(StringIO(data.decode("utf-8-sig"))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Empty consolidated data directory")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--history-root", type=Path, action="append", default=[])
    parser.add_argument("--batches", nargs="+", type=int, default=[1,4,8])
    parser.add_argument("--include-cpu", action="store_true")
    parser.add_argument("--weights-manifest", type=Path)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Use an empty output directory; existing consolidated data is never overwritten")
    bundle = load_run(args.run)
    first = bundle["models"][bundle["order"][0]]["meta"]
    modes = [m for m in first["experiment"]["modes"] if args.include_cpu or m not in CPU_MODES]
    bundle["report_scope"] = {"models": bundle["order"], "batches": args.batches, "modes": modes}
    expected = len(bundle["order"]) * len(args.batches) * len(modes)
    count = sum(bundle["models"][m]["meta"]["routes"].get(f"{b}:{mode}",{}).get("status")=="passed"
                for m in bundle["order"] for b in args.batches for mode in modes)
    if count != expected:
        raise ValueError(f"Requested scope is incomplete: {count}/{expected}")
    verified_weights = []
    if args.weights_manifest:
        manifest = json.loads(args.weights_manifest.read_text(encoding="utf-8"))
        for entry in manifest:
            checkpoint = PROJECT / entry["path"]
            exported = args.run / entry["model_id"] / "best.pt"
            measured = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if measured != entry["sha256"] or hashlib.sha256(exported.read_bytes()).hexdigest() != measured:
                raise ValueError(f"Checkpoint copy differs from retained original: {entry['model_id']}")
            verified_weights.append({"model": entry["model_id"], "sha256": measured})
    export_data(bundle,args.output,args.dataset)
    current = {(model,key,route.get("started_at")) for model,item in bundle["models"].items()
               for key,route in item["meta"]["routes"].items() if route.get("started_at")}
    attempts, evidence = {}, []

    def consume(meta_bytes, raw_bytes, source):
        meta = json.loads(meta_bytes)
        model = meta["model_id"]
        if model not in bundle["models"]:
            raise ValueError(f"Unexpected model in historical data: {model}")
        original = rows(raw_bytes)
        grouped = defaultdict(list)
        for row in original:
            grouped[f"{row['batch']}:{row['mode']}"].append(row)
        evidence.append({"source":source,"metadata_sha256":hashlib.sha256(meta_bytes).hexdigest(),
                         "rounds_sha256":hashlib.sha256(raw_bytes).hexdigest()})
        for key,route in meta.get("routes",{}).items():
            if not route.get("started_at"):
                continue
            identity=(model,key,route["started_at"])
            if identity in current:
                continue
            batch, mode=key.split(":",1)
            entry={"model":model,"batch":int(batch),"mode":mode,"route":route,"rounds":grouped[key],
                   "environment":meta.get("environments",{}).get(route.get("environment"),{}),
                   "checkpoint_sha256":meta.get("checkpoint_sha256"),"dataset":meta.get("dataset"),
                   "experiment":meta["experiment"],"sources":[source]}
            entry=sanitize(entry)
            if identity in attempts:
                old=attempts[identity]
                sources=sorted(set(old["sources"]+[source]))
                if (str(route.get("finished_at") or ""),len(entry["rounds"])) > (str(old["route"].get("finished_at") or ""),len(old["rounds"])):
                    attempts[identity]=entry
                attempts[identity]["sources"]=sources
            else:
                attempts[identity]=entry

    seen = set()
    for folder in args.history_root:
        for path in folder.rglob("model.json"):
            raw=path.parent/"results.csv"
            if raw.is_file() and path.resolve() not in seen:
                seen.add(path.resolve())
                consume(path.read_bytes(),raw.read_bytes(),f"{folder.name}/{path.relative_to(folder).as_posix()}")
        for path in folder.rglob("*.zip"):
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            with zipfile.ZipFile(path) as archive:
                names=set(archive.namelist())
                for name in sorted(names):
                    if name.endswith("/model.json") and name.removesuffix("model.json")+"results.csv" in names:
                        raw=name.removesuffix("model.json")+"results.csv"
                        consume(archive.read(name),archive.read(raw),f"{path.name}/{name}")
    with (args.output/"history.jsonl").open("w",encoding="utf-8") as stream:
        for identity,entry in sorted(attempts.items()):
            stream.write(json.dumps(entry,ensure_ascii=False,separators=(",",":"))+"\n")
    audit={"completed_routes":count,"planned_routes":expected,"history_attempts":len(attempts),
           "historical_sources":evidence,"verified_checkpoint_copies":verified_weights,
           "current_round_states":dict(Counter(r["status"] for item in bundle["models"].values() for r in item["raw"])),
           "note":"Original metadata and numeric rows are retained; only private home prefixes are sanitized."}
    seal(args.output,audit)
    loaded=load_data(args.output)
    for model in bundle["order"]:
        assert loaded["models"][model]["meta"]==sanitize(bundle["models"][model]["meta"]),model
        assert loaded["models"][model]["raw"]==sanitize(bundle["models"][model]["raw"]),model
        assert loaded["models"][model]["rows"]==bundle["models"][model]["rows"],model
    print(json.dumps({"output":str(args.output),"completed_routes":count,"history_attempts":len(attempts),
                      "verified_original_weights":len(verified_weights),"exact_roundtrip":True},indent=2))


if __name__ == "__main__":
    main()
