"""extract_auto_alls.py COMPILED_HAR OUT_ALLS: the allocator's full solution (contexts, defuse, place, buffers, portals)."""
import sys, tarfile
with tarfile.open(sys.argv[1]) as t:
    m = [x for x in t.getnames() if x.endswith(".auto.alls")]
    assert m, "no .auto.alls in the compiled HAR"
    open(sys.argv[2], "wb").write(t.extractfile(m[0]).read())
print("auto.alls ->", sys.argv[2])
