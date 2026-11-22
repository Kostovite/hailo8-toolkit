"""Pull the OR-tools 7.x CP-SAT schema (serialized FileDescriptorProto embedded by protoc) out of the Hailo compiler
binary and build message classes from it, to read the memo cache's binary models / responses exactly.
Usage (any python with protobuf): from oldproto import old_classes; Model, Response = old_classes(BINARY)"""
import sys
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

BIN = "/mnt/storage/anaconda3/envs/hailo_env/lib/python3.10/site-packages/hailo_tools/build/compiler"


def _varint(b, i):
    v = s = 0
    while True:
        c = b[i]; i += 1; v |= (c & 0x7F) << s; s += 7
        if c < 0x80:
            return v, i


def extract(data, name):
    key = b"\n" + bytes([len(name)]) + name.encode()
    at = data.find(key)
    if at < 0:
        raise KeyError(name)
    i = at
    while i < len(data):  # walk FileDescriptorProto fields (1..12) until an invalid tag
        tag, j = _varint(data, i); f, wt = tag >> 3, tag & 7
        if not (1 <= f <= 14) or wt not in (0, 2):
            break
        if wt == 0:
            _, j = _varint(data, j)
        else:
            n, j = _varint(data, j); j += n
        i = j
    fd = descriptor_pb2.FileDescriptorProto(); fd.ParseFromString(data[at:i])
    return fd


def old_classes(binary=BIN):
    data = open(binary, "rb").read(); pool = descriptor_pool.DescriptorPool()
    for dep in ("ortools/util/sorted_interval_list.proto",):
        pass
    fd = extract(data, "ortools/sat/cp_model.proto")
    for d in fd.dependency:  # load dependencies first
        pool.Add(extract(data, d))
    pool.Add(fd)
    get = lambda n: message_factory.GetMessageClass(pool.FindMessageTypeByName(n))  # noqa: E731
    return get(fd.package + ".CpModelProto"), get(fd.package + ".CpSolverResponse")


if __name__ == "__main__":
    M, R = old_classes(sys.argv[1] if len(sys.argv) > 1 else BIN)
    print(M.DESCRIPTOR.full_name, [f.name for f in M.DESCRIPTOR.fields])
