"""Revert one layer's defuse in an assembled script, consistently: drop its defuse statement and every statement that
names one of its pieces (shortcuts, concats, portals, compilation_param, buffers), and map the pieces back to the
layer in context/place lists. Used to bisect a crashing full script down to the layer decision that breaks it.
Usage: python undefuse.py IN.alls LAYER OUT.alls   |   python undefuse.py IN.alls --list"""
import re
import sys


def pieces_of(text, layer):
    m = re.search(rf"^(.+?) = defuse\({re.escape(layer)}, ", text, re.M)
    return [p.strip() for p in m.group(1).split(",")] if m else []


def undefuse(text, layer):
    ps = set(pieces_of(text, layer))
    out = []
    word = lambda s, n: re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", s)  # noqa: E731
    for ln in text.splitlines():
        if re.match(rf"^.+ = defuse\({re.escape(layer)}, ", ln):
            continue
        lm = re.match(r"^(\w+_context_\d+(?:\.place\(\d+, | = context\())\[([^\]]*)\](\).*)$", ln)
        if lm:
            names = [n.strip() for n in lm.group(2).split(",")]
            new, seen = [], set()
            for n in names:
                n2 = layer if n in ps else n
                if n2 not in seen and not any(word(n2, p) and n2 != p for p in ()):
                    seen.add(n2)
                    new.append(n2)
            out.append(f"{lm.group(1)}[{', '.join(new)}]{lm.group(3)}")
            continue
        if any(word(ln, p) for p in ps):
            continue
        out.append(ln)
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    t = open(sys.argv[1]).read()
    if sys.argv[2] == "--list":
        print("\n".join(re.findall(r"^.+ = defuse\((\w+), ", t, re.M)))
    else:
        open(sys.argv[3], "w").write(undefuse(t, sys.argv[2]))
