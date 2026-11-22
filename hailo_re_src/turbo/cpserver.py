"""Modern CP-SAT solve server for the Hailo DFC compiler (used by the turbo hook, HAILO_CPSERVER=SOCKET).

The compiler embeds OR-tools 7.x CP-SAT and runs every placement solve on ONE worker. Its hardest solves (6k vars,
8k constraints) end UNKNOWN after ~50 s; modern CP-SAT with 8 workers proves them OPTIMAL in ~1-3 s (cpbench.py).
The hook sends each model (old-schema bytes) here; we upgrade it, solve it with the current CP-SAT portfolio, and
return an old-schema CpSolverResponse (status, solution, objective, bound).

Protocol per connection: 8-byte little-endian length + model bytes -> 8-byte length + response bytes (0 = refuse,
the hook then solves in-process as stock).
Usage: ortools_env/bin/python turbo/cpserver.py SOCKET [--workers 8] [--limit 60]
"""
import argparse
import os
import socket
import socketserver
import struct
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from google.protobuf import text_format  # noqa: E402
from ortools.sat import cp_model_pb2  # noqa: E402
from ortools.sat.python import cp_model  # noqa: E402

from cpsolve import upgrade  # noqa: E402
from oldproto import old_classes  # noqa: E402

OldM, OldR = old_classes()
A = None
LOG_LOCK = threading.Lock()


def _recv(sock, n):
    b = bytearray()
    while len(b) < n:
        c = sock.recv(min(1 << 20, n - len(b)))
        if not c:
            raise ConnectionError("short read")
        b += c
    return bytes(b)


def solve(req):
    t0 = time.time()
    om = OldM(); om.ParseFromString(req)
    model = cp_model.CpModel(); model.proto.parse_text_format(upgrade(text_format.MessageToString(om)))
    s = cp_model.CpSolver(); p = s.parameters
    p.num_workers = A.workers; p.max_time_in_seconds = A.limit
    st = s.Solve(model)
    r = OldR(); r.status = int(st)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        r.solution.extend(int(v) for v in s.response_proto.solution)
        if om.HasField("objective"):
            r.objective_value = s.ObjectiveValue(); r.best_objective_bound = s.BestObjectiveBound()
    r.wall_time = time.time() - t0
    r.num_conflicts = s.NumConflicts(); r.num_branches = s.NumBranches()
    with LOG_LOCK:
        print(f"{time.strftime('%H:%M:%S')} vars={len(om.variables)} cons={len(om.constraints)} "
              f"{s.StatusName(st)} {time.time() - t0:.2f}s", flush=True)
    return r.SerializeToString()


class H(socketserver.BaseRequestHandler):
    def handle(self):
        try:
            n = struct.unpack("<Q", _recv(self.request, 8))[0]
            out = solve(_recv(self.request, n))
        except Exception as e:  # noqa: BLE001  refuse: the hook falls back to the stock in-process solve
            with LOG_LOCK:
                print("REFUSE", type(e).__name__, str(e)[:200], flush=True)
            out = b""
        self.request.sendall(struct.pack("<Q", len(out)) + out)


class S(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("sock"); ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=float, default=60.0); A = ap.parse_args()
    if os.path.exists(A.sock):
        try:
            socket.socket(socket.AF_UNIX).connect(A.sock); print("already running"); sys.exit(0)
        except OSError:
            os.unlink(A.sock)
    print(f"cpserver on {A.sock}: workers {A.workers}, limit {A.limit}s", flush=True)
    S(A.sock, H).serve_forever()
