"""Make a context partition legal for the external pipeline (rules found by probing the compiler) and emit the
per-context sub-compile hints.

  R0 a const_input lives in its earliest consumer's context
  R1 a producer and its slice family (feature_splitter / conv_feature_splitter*) stay in one context
  R5/R3 hints (entry layer = fed across a cut; the sub-compile would split it spatially, which the full model
     cannot place): dw -> INPUT_FEATURES defuse, conv -> output-feature defuse x2 (INPUT_FEATURES is dw only);
     written to c{i}.fs.alls, used by hcc's *_fs variants
  R10 a context never starts at a format_conversion: its producer moves forward into that context
  R16 ditto for a bilinear resize (nearest is fine)
  R14 ditto for a mean-subtract fan-out (tensor feeding a reduce_mean and an ew_sub: a norm block)
  R13 a context never ends at a feature_multiplier (x^2): it moves forward to its consumers
  R15 an ew_* layer that is both entry (all inputs earlier) and exit (an output later) of a context moves back
  R17 a format_conversion's consumers never sit in a later context (map::at / "can't spatial reshape")
  R18 no token-strip tensor (shape [., 1, N, C]: inside attention) crosses a context ("can't spatial reshape
     (1,400,800) to (50,8,800) without a successor format_conversion")
  R19 the pieces of one decomposed layer_normalizationN / softmaxN stay together ("Shapes at dim #2 aren't equal")
  R20 a feature_splitter's consumers never sit in a later context ("non-unique elements" ValidationError)
  -- R17-R20 are retry tier 1, R19 + the next three tier 2 (env PART_FIX_STRICT, set by hcc after a failure):
  R21 a dense output (SE squeeze vector) never crosses; R16b no resize (nearest too) starts a context;
     R22 a conv_slice piece stays with its producer (fz_1011 / fz_1055 / fz_1103: "Splitter failed", microcode size)
     (R17-R22 pull the CONSUMER back into the producer's context; stock's yolo26m partition has no such edge)
  R12 softmax x V matmuls (> 400 tokens) get pinned SCs (c{i}.extra.alls) so the sub-compile keeps them whole

Moves only toward an earlier context (to the latest context of the inputs), so topological order holds.
Usage: python part_fix.py FULL.har PART_DIR [OUT_DIR]   (default: in place)
"""
import json
import os
import re
import sys
import tarfile

STRICT = int(os.environ.get("PART_FIX_STRICT") or 0)  # retry tiers, set by hcc only after a compile failure:
# 1 = R17/R18/R20 (fuzz: crossings that fail 12/16, mostly), 2 = also R19/R21/R22/R16b (usually fine, 39/2 ...).
# Tier 0 (default) keeps a partition that compiles as it is: applying them up front cost fz_1068 -18% FPS.


def hn_of(har):
    with tarfile.open(har) as t:
        nm = next(m for m in t.getnames() if m.endswith(".hn") and not m.endswith((".native.hn", ".fp.hn")))
        return json.loads(t.extractfile(nm).read())


def fix(har, part, out=None, fs=4):
    out = out or part
    hn = hn_of(har)
    L = hn["layers"]
    full = lambda s: s if "/" in s else f"{hn['name']}/{s}"  # noqa: E731
    parts = []
    i = 0
    while os.path.exists(f"{part}/c{i}.layers.json"):
        parts.append([full(n) for n in json.load(open(f"{part}/c{i}.layers.json"))])
        i += 1
    ctx = {n: i for i, p in enumerate(parts) for n in p}
    moves = []
    for n, v in L.items():  # R0: a const_input (no inputs; part_gen's topological order puts them all first) lives in
        if n in ctx and v["type"] == "const_input":  # its earliest consumer's context (LM rotary tables: 150 of them)
            t = min((ctx[o] for o in v["output"] if o in ctx), default=ctx[n])
            if t != ctx[n]:
                moves.append(("R0>", n.split("/")[-1], ctx[n], t))
                ctx[n] = t
    grp = lambda n: (lambda m: m and m.group(0))(re.search(r"(layer_normalization|softmax)\d+$", n))  # noqa
    for _ in range(50):
        changed = False
        for n, v in L.items():
            if n not in ctx:
                continue
            ins = [s for s in v["input"] if s in ctx]
            if (v["type"] in ("feature_splitter", "conv_feature_splitter") or "feature_splitter" in n) and not \
                    os.environ.get("PART_FIX_NO_R1"):  # R1 (stricter than needed: stock's yolo26m splits conv22)
                src = [s for s in v["input"] if s in ctx]
                if src and ctx[src[0]] != ctx[n] and ctx[src[0]] < ctx[n]:
                    p_ = src[0]
                    fam = [o for o in L[p_]["output"] if o in ctx and "feature_splitter" in o]
                    fwd = max(ctx[f] for f in fam)
                    # forward: producer + family join the latest member (keeps an attention block with its
                    # matmul), legal when the producer's inputs are earlier and its other consumers not earlier
                    ok_fwd = (all(ctx.get(s2, -1) <= fwd for s2 in L[p_]["input"]) and
                              all(ctx.get(o, 99) >= fwd for o in L[p_]["output"] if o not in fam) and
                              all(ctx.get(o2, 99) >= fwd for f in fam for o2 in L[f]["output"]))
                    if ok_fwd:
                        for m_ in [p_] + fam:
                            if ctx[m_] != fwd:
                                moves.append(("R1>", m_.split("/")[-1], ctx[m_], fwd))
                                ctx[m_] = fwd
                    else:
                        moves.append(("R1<", n.split("/")[-1], ctx[n], ctx[p_]))
                        ctx[n] = ctx[p_]
                    changed = True
        for n, v in L.items():  # R10: a context never starts at a format_conversion (attention reshape): pull
            # its producer forward, so a compute layer leads into it (e0 attention fails). R14: same for a
            # mean-subtract fan-out (input of a reduce_mean AND an ew_sub, i.e. a norm block; Splitter failed)
            r14 = v["type"] == "reduce_mean" and any(
                any(L.get(o, {}).get("type") == "ew_sub" for o in L[s]["output"]) for s in v["input"] if s in L)
            r16 = v["type"] == "resize" and v.get("params", {}).get("method") == "bilinear"  # R16 (fuzz: "No
            # format for conv -> resize1" / "Can't find mutual format"): a bilinear resize never starts a context
            if n not in ctx or (v["type"] != "format_conversion" and not r14 and not r16):
                continue
            for p_ in [s for s in v["input"] if s in ctx and ctx[s] < ctx[n]]:
                t = ctx[n]
                g_ = grp(p_)
                if STRICT >= 2 and g_ and any(grp(m) == g_ and ctx[m] < t for m in ctx if m != p_) and all(
                        ctx.get(s2, -1) <= ctx[p_] for s2 in v["input"]):  # forward would split p_'s norm/softmax
                    moves.append(("R10<", n.split("/")[-1], ctx[n], ctx[p_]))  # group (R19): n comes back instead
                    ctx[n] = ctx[p_]
                    changed = True
                    break
                if all(ctx.get(s2, -1) <= t for s2 in L[p_]["input"]) and all(
                        ctx.get(o, 99) >= t for o in L[p_]["output"]) and "feature_splitter" not in p_:
                    moves.append(("R10>", p_.split("/")[-1], ctx[p_], t))
                    ctx[p_] = t
                    changed = True
        for n, v in L.items():  # R15: an elementwise op that is both entry and exit of a context (every input
            if n not in ctx or not v["type"].startswith("ew_"):  # from earlier contexts, an output to a later one)
                continue                                     # fails the Splitter (fz_1003 SE): it moves back
            ins = [ctx[s] for s in v["input"] if s in ctx]
            if ins and max(ins) < ctx[n] and any(ctx.get(o, -1) > ctx[n] for o in v["output"]):
                t = max(ins)
                moves.append(("R15<", n.split("/")[-1], ctx[n], t))
                ctx[n] = t
                changed = True
        for n, v in L.items():  # R13: a context never ends at a feature_multiplier (x^2): its output can't feed a
            if n not in ctx or v["type"] != "feature_multiplier":  # cut (core assert in set_default_power_table),
                continue                                           # so it moves forward to its consumers
            outs = [o for o in v["output"] if o in ctx]
            if outs and all(ctx[o] > ctx[n] for o in outs):
                t = min(ctx[o] for o in outs)
                moves.append(("R13>", n.split("/")[-1], ctx[n], t))
                ctx[n] = t
                changed = True
        for n, v in L.items():  # R17-R20: pull a consumer back across the cut when the edge must not cross
            if n not in ctx:
                continue
            for o in v["output"]:
                if o not in ctx or ctx[o] <= ctx[n]:
                    continue
                sh = v["output_shapes"][0]
                why = (None if not STRICT else
                       "R17" if v["type"] == "format_conversion" else
                       "R18" if len(sh) == 4 and sh[1] == 1 and sh[2] > 1 else
                       "R20" if v["type"] == "feature_splitter" else
                       None if STRICT < 2 else  # strict tier: crossings that are USUALLY fine (fuzz exact/fail
                       "R19" if grp(n) and grp(n) == grp(o) else  # 39/2, 7/1, 7/1, 18/2): hcc's retry only
                       "R21" if v["type"] == "dense" else
                       "R22" if "conv_slice" in o else
                       "R16b" if L[o]["type"] == "resize" else None)
                if why and all(ctx.get(s2, -1) <= ctx[n] for s2 in L[o]["input"]):
                    moves.append((why + "<", o.split("/")[-1], ctx[o], ctx[n]))
                    ctx[o] = ctx[n]
                    changed = True
        if not changed:
            break
    k0 = len(parts)
    keep = [i for i in range(k0) if any(c == i for c in ctx.values())]  # moves can empty a context: drop it
    ctx = {n: keep.index(c) for n, c in ctx.items()}
    k = len(keep)
    os.makedirs(out, exist_ok=True)
    for j in range(k, k0):  # stale files of dropped trailing indices
        for suf in ("layers.json", "fs.alls", "extra.alls"):
            if os.path.exists(f"{out}/c{j}.{suf}"):
                os.remove(f"{out}/c{j}.{suf}")
    new = [[n for n in L if ctx.get(n) == i] for i in range(k)]
    for i, p in enumerate(new):
        json.dump(p, open(f"{out}/c{i}.layers.json", "w"))
        hint = []
        for n in p:
            v = L[n]
            entry = any(s in ctx and ctx[s] != i for s in v["input"])
            s = n.split("/")[-1]
            if entry and v["type"] == "dw":  # R5: an entry dw may only be FEATURE-defused (input features)
                pieces = ", ".join([f"{s}_fs"] + [f"{s}_d{j}" for j in range(fs)] + [f"{s}_dc"])
                hint.append(f"{pieces} = defuse({s}, {fs}, defuse_type=INPUT_FEATURES)")
            elif entry and v["type"] == "conv":  # R3: never spatially; output features (INPUT_FEATURES: dw only)
                hint.append(f"{s}_d0, {s}_d1, {s}_dc = defuse({s}, 2)")
        # R12: the softmax x V matmul (an ew_mult input) is otherwise INPUT_FEATURES-defused: at > 400 tokens a
        # solution the compiler cannot replay from script (map::at), and at ANY token count silently wrong on the
        # chip (single-head probes N=64/784: 1-5 dB vs emulator 40, RULES section 26); pinning its SCs keeps it
        # whole (-1% FPS). (at exactly 400 tokens the crash: fuzz fz_1063/1064/1068; stock's yolo26m pins 2 SC)
        pin = [f"compilation_param({n.split('/')[-1]}, resources_allocation_strategy=manual_scs_selection, "
               f"number_of_subclusters={8 if L[n]['output_shapes'][0][2] > 400 else 2})" for n in p
               if L[n]["type"] == "matmul" and any(L[s]["type"] == "ew_mult" for s in L[n]["input"] if s in L)]
        # R23: a deconv with kernel == stride is lowered to DECONV -> FEATURE_INTERLEAVE, which is ~100% wrong on the
        # chip (k2s2: 99-100% of outputs, stock and ours, RULES section 26); the SUPER_DECONV lowering of the same
        # layer is exact -> force it
        pin += [f"{n.split('/')[-1]}_defuse_conv, {n.split('/')[-1]}_defuse_d2s = defuse({n.split('/')[-1]}, 1, "
                f"defuse_type=SUPER_DECONV)" for n in p if L[n]["type"] == "deconv"
                and list(L[n]["params"].get("kernel_shape", [0, 0])[:2]) == list(L[n]["params"].get("strides", [1, 0, 0, 1])[1:3])]
        xp = f"{out}/c{i}.extra.alls"
        if pin:
            open(xp, "w").write("\n".join(pin) + "\n")
        elif os.path.exists(xp):
            os.remove(xp)
        path = f"{out}/c{i}.fs.alls"
        if hint:
            open(path, "w").write("\n".join(hint) + "\n")
        elif os.path.exists(path):
            os.remove(path)
    return moves, [len(p) for p in new]


if __name__ == "__main__":
    mv, sizes = fix(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
    print("moves", mv, "sizes", sizes)
