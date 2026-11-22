"""Hailo Model Zoo docs/public_models/HAILO8/*.rst -> one CSV row per network (task, dataset, metrics, FPS, links).

Usage: python parse_public.py DOCS_DIR OUT.csv
"""
import csv
import glob
import os
import re
import sys

rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "HAILO8_*.rst"))):
    task = os.path.basename(f)[7:-4]
    lines = open(f, encoding="utf-8").read().splitlines()
    dataset, header, cur = None, None, None
    for i, ln in enumerate(lines):
        if i + 1 < len(lines) and re.fullmatch(r"-{3,}", lines[i + 1].strip()) and ln.strip() and ln.strip() != "Link Legend":
            dataset = ln.strip()
        m = re.match(r"^   \* - (.*)$", ln)
        m2 = re.match(r"^     - (.*)$", ln)
        if m:
            if cur is not None and header and len(cur) == len(header):
                rows.append(cur)
            cur = [m.group(1).strip()]
            if cur[0] == "Network Name":
                header, cur = ["Network Name"], None
                hdr_mode = True
            else:
                hdr_mode = False
        elif m2 is not None:
            if cur is None and header is not None and hdr_mode:
                header.append(m2.group(1).strip())
            elif cur is not None:
                cur.append(m2.group(1).strip())
        if cur is not None and header and len(cur) == len(header) and (i + 1 == len(lines) or not lines[i + 1].startswith("     - ")):
            d = dict(zip(header, cur))
            name = re.sub(r"[ ⭐]", "", d["Network Name"]).strip()
            links = d.get("Links", "")
            hef = re.search(r"`HEF <([^>]+)>`", links)
            pt = re.search(r"`PT <([^>]+)>`", links)
            vals = list(d.values())
            rows_out = {"task": task, "dataset": dataset, "network": name, "float": vals[1], "hw": vals[2],
                        "fps_b1": d.get("FPS (Batch Size=1)", ""), "fps_b8": d.get("FPS (Batch Size=8)", ""),
                        "input": next((v for k, v in d.items() if k.startswith("Input Resolution")), ""),
                        "params_M": d.get("Params (M)", ""), "ops_G": next((v for k, v in d.items() if k.startswith(("OPS", "MAC"))), ""),
                        "metric": header[1], "hef_url": hef.group(1) if hef else "", "pt_url": pt.group(1) if pt else ""}
            rows.append(rows_out)
            cur = None
rows = [r for r in rows if isinstance(r, dict)]
with open(sys.argv[2], "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(len(rows), "networks;", len({r["network"] for r in rows}), "unique")
