"""
Inspect a PyTorch .pt / .pth checkpoint without importing torch -- including
the actual tensor values.

Modern checkpoints are zip archives: a pickle index (data.pkl) naming the
tensors, plus one raw byte blob per storage. This script unpickles the index
with torch's classes replaced by harmless stand-ins, then reads each storage
blob with numpy.frombuffer. No torch, and nothing from the file is executed.

Usage:
    python inspect_checkpoint2.py model.pt                # names, shapes, dtypes
    python inspect_checkpoint2.py model.pt --stats        # + mean/std/min/max, NaN check
    python inspect_checkpoint2.py model.pt --dump LAYER   # print one tensor's values
"""

import argparse
import io
import pickle
import zipfile

import numpy as np

# Torch storage class name -> numpy dtype. Covers the classic typed storages;
# newer checkpoints may say UntypedStorage, handled separately below.
DTYPES = {
    "DoubleStorage": np.float64,
    "FloatStorage": np.float32,
    "HalfStorage": np.float16,
    "LongStorage": np.int64,
    "IntStorage": np.int32,
    "ShortStorage": np.int16,
    "CharStorage": np.int8,
    "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
    "float64": np.float64,
    "float32": np.float32,
    "float16": np.float16,
    "int64": np.int64,
    "int32": np.int32,
    "uint8": np.uint8,
    "bool": np.bool_,
}


def numpy_dtype(name):
    for key, dtype in DTYPES.items():
        if key in name:
            return dtype
    return None


class Opaque:
    """Placeholder for classes we don't need. Tolerates pickle's BUILD opcode."""

    def __init__(self, name):
        self.name = name

    def __setstate__(self, state):
        self.state = state

    def __repr__(self):
        return f"<{self.name}>"


class Stub:
    def __init__(self, module, name):
        self.module, self.name = module, name

    def __call__(self, *args, **kwargs):
        return Opaque(f"{self.module}.{self.name}")

    def __repr__(self):
        return f"{self.module}.{self.name}"


class Reader:
    """Decodes one checkpoint. Holds the zip open so storages can be read lazily."""

    def __init__(self, path):
        self.zip = zipfile.ZipFile(path)
        names = self.zip.namelist()
        index_name = next(n for n in names if n.endswith("data.pkl"))
        self.prefix = index_name.rsplit("/", 1)[0] if "/" in index_name else ""
        self.root = self._unpickle(self.zip.read(index_name))

    def _unpickle(self, raw):
        reader = self

        class CheckpointUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if name == "_rebuild_tensor_v2":
                    return reader._rebuild
                if name == "_rebuild_parameter":
                    return lambda tensor, *a: tensor
                if module == "collections" and name == "OrderedDict":
                    import collections
                    return collections.OrderedDict
                return Stub(module, name)

            def persistent_load(self, pid):
                # ('storage', storage_type, key, location, numel)
                return {"key": str(pid[2]), "type": str(pid[1]), "numel": pid[4]}

        return CheckpointUnpickler(io.BytesIO(raw)).load()

    def _rebuild(self, storage, offset, size, stride, *rest):
        dtype = numpy_dtype(storage["type"])
        if dtype is None:
            return Opaque(f"unsupported dtype {storage['type']}")

        member = f"{self.prefix}/data/{storage['key']}" if self.prefix else f"data/{storage['key']}"
        buffer = self.zip.read(member)
        flat = np.frombuffer(buffer, dtype=dtype)

        count = int(np.prod(size)) if size else 1
        flat = flat[offset:offset + count]
        if flat.size != count:
            return Opaque(f"size mismatch: got {flat.size}, expected {count}")
        return flat.reshape(tuple(size))


def walk(obj, path=""):
    """Yield (name, array) for every numpy array anywhere in the object tree."""
    if isinstance(obj, np.ndarray):
        yield path, obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{path}.{key}" if path else str(key)
            yield from walk(value, child)
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            yield from walk(value, f"{path}[{i}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--stats", action="store_true", help="show value statistics")
    parser.add_argument("--dump", metavar="NAME", help="print one tensor's values")
    args = parser.parse_args()

    reader = Reader(args.path)
    tensors = list(walk(reader.root))

    if args.dump:
        matches = [(n, a) for n, a in tensors if args.dump in n]
        if not matches:
            print(f"no tensor matching {args.dump!r}")
            print("available:", ", ".join(n for n, _ in tensors[:20]), "...")
            return
        for name, array in matches:
            print(f"\n{name}  shape={array.shape}  dtype={array.dtype}")
            with np.printoptions(precision=4, suppress=True, threshold=200):
                print(array)
        return

    print(f"file:      {args.path}")
    print(f"root type: {type(reader.root).__name__}")
    print(f"tensors:   {len(tensors)}")
    print()

    if args.stats:
        header = f"{'name':52s} {'shape':20s} {'mean':>10s} {'std':>10s} {'min':>10s} {'max':>10s}"
        print(header)
        print("-" * len(header))

    suspicious = []
    for name, array in tensors:
        if args.stats:
            values = array.astype(np.float64).ravel()
            print(
                f"{name:52s} {str(array.shape):20s} "
                f"{values.mean():10.4f} {values.std():10.4f} "
                f"{values.min():10.4f} {values.max():10.4f}"
            )
            if not np.isfinite(values).all():
                suspicious.append(f"{name}: contains NaN or inf")
            elif values.std() == 0 and values.size > 1:
                suspicious.append(f"{name}: all values identical")
        else:
            print(f"{name:52s} {str(array.shape):20s} {array.dtype}")

    total = sum(a.size for _, a in tensors)
    print()
    print(f"total parameters: {total:,}")

    if suspicious:
        print()
        print("warnings:")
        for line in suspicious:
            print(f"  {line}")


if __name__ == "__main__":
    main()