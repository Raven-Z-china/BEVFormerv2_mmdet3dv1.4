import lmdb
import torch
import zlib
import msgpack
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

class OptimizedFeatureStorage:
    """
    Minimal feature storage for composite data (two tensors + one float).
    - Write: single write each time
    - Read: batch read (any size)
    """

    def __init__(self, db_path: str, map_size: int = 1 * 1024**3, compression_level: int = 0, device: str = 'cuda'):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.compression_level = compression_level
        self.device = device

        # LMDB environment (no metasync, sync for speed)
        self.env = lmdb.open(
            str(db_path), map_size=map_size, max_dbs=3,
            writemap=True, metasync=False, sync=False, map_async=True, lock=False
        )
        self.data_db = self.env.open_db(b'data')
        self.meta_db = self.env.open_db(b'meta')
        self.index_db = self.env.open_db(b'index')

        # Load index into memory
        self.index = {}
        with self.env.begin(db=self.index_db) as txn:
            cursor = txn.cursor()
            for k, v in cursor:
                self.index[k.decode()] = msgpack.loads(v)

    # ----------------------------------------------------------------------
    # Index helpers
    # ----------------------------------------------------------------------
    def _update_index(self, key: str, info: dict):
        self.index[key] = info
        with self.env.begin(db=self.index_db, write=True) as txn:
            txn.put(key.encode(), msgpack.dumps(info))

    # ----------------------------------------------------------------------
    # Compression / decompression
    # ----------------------------------------------------------------------
    def _compress_tensor(self, tensor: torch.Tensor) -> bytes:
        np_arr = tensor.detach().cpu().numpy()
        data = msgpack.dumps({
            'shape': np_arr.shape,
            'dtype': str(np_arr.dtype).encode(),
            'data': np_arr.tobytes()
        })
        return zlib.compress(data, level=self.compression_level) if self.compression_level > 0 else data

    def _decompress_tensor(self, data: bytes) -> torch.Tensor:
        if self.compression_level > 0:
            data = zlib.decompress(data)
        info = msgpack.loads(data)
        np_arr = np.frombuffer(info['data'], dtype=info['dtype']).reshape(info['shape'])
        return torch.from_numpy(np_arr).to(self.device)

    def _compress_composite(self, t1: torch.Tensor, t2: torch.Tensor, f: float) -> bytes:
        c1 = self._compress_tensor(t1)
        c2 = self._compress_tensor(t2)
        return msgpack.dumps({'t1': c1, 't2': c2, 'f': f})

    def _decompress_composite(self, data: bytes) -> Tuple[torch.Tensor, torch.Tensor, float]:
        packed = msgpack.loads(data)
        return (self._decompress_tensor(packed['t1']),
                self._decompress_tensor(packed['t2']),
                packed['f'])

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------
    def write(self, key: str, tensor1: torch.Tensor, tensor2: torch.Tensor, float_val: float, metadata: dict = None):
        """Write composite feature (two tensors + float)."""
        compressed = self._compress_composite(tensor1, tensor2, float_val)
        with self.env.begin(write=True, db=self.data_db) as txn:
            txn.put(key.encode(), compressed)
        if metadata:
            with self.env.begin(write=True, db=self.meta_db) as txn:
                txn.put(key.encode(), msgpack.dumps(metadata))
        self._update_index(key, {
            't1_shape': tensor1.shape, 't1_dtype': str(tensor1.dtype),
            't2_shape': tensor2.shape, 't2_dtype': str(tensor2.dtype),
            'float': float_val
        })

    def read_batch(self, keys: List[str]) -> List[Tuple[torch.Tensor, torch.Tensor, float]]:
        """Read a batch of composite features."""
        result = []
        with self.env.begin(db=self.data_db) as txn:
            for key in keys:
                data = txn.get(key.encode())
                if data is None:
                    raise KeyError(f"Key not found: {key}")
                result.append(self._decompress_composite(data))
        return result

    def read(self, key: str) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Read a single composite feature."""
        return self.read_batch([key])[0]

    def exists(self, key: str) -> bool:
        return key in self.index

    def get_info(self, key: str) -> Optional[dict]:
        return self.index.get(key)

    def delete(self, key: str):
        with self.env.begin(write=True, db=self.data_db) as txn:
            txn.delete(key.encode())
        with self.env.begin(write=True, db=self.meta_db) as txn:
            txn.delete(key.encode())
        with self.env.begin(write=True, db=self.index_db) as txn:
            txn.delete(key.encode())
        self.index.pop(key, None)

    def close(self):
        self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    with OptimizedFeatureStorage('./History_BEV.lmdb') as store:
        # Write
        t1 = torch.randn(3, 224, 224)
        t2 = torch.randn(64, 56, 56)
        store.write('sample', t1, t2, 3.14, metadata={'desc': 'test'})

        # Read
        t1_out, t2_out, f_out = store.read('sample')
        print((t1_out-t1).sum(), (t2_out-t2).sum(), f_out-3.14)
