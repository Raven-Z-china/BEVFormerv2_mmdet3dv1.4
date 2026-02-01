import lmdb
import torch
import zlib
import msgpack
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from threading import Lock
from dataclasses import dataclass
import time
from functools import lru_cache

@dataclass
class FeatureInfo:
    """特征元信息"""
    shape: tuple
    dtype: str
    compressed_size: int
    timestamp: float
    access_count: int = 0

class OptimizedFeatureStorage:
    """
    针对读写模式优化的特征存储：
    - 写入：单次写入一个
    - 读取：批量读取2或8个
    """
    
    def __init__(self, 
                 db_path: str,
                 map_size: int = 1 * 1024**3,  # 1.5TB
                 compression_level: int = 3,
                 cache_size: int = 100,           # LRU缓存大小
                 prefetch_buffer: int = 8,       # 预读取缓冲
                 use_fp16: bool = False,
                 device: str = 'cuda'):
        """
        Args:
            db_path: 数据库路径
            map_size: 数据库大小（字节）
            compression_level: 压缩级别（1-9，1最快，9最高）
            cache_size: 内存缓存的特征数量
            prefetch_buffer: 预读取缓冲大小
            device: 默认设备
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.compression_level = compression_level
        self.device = device
        self.prefetch_buffer = prefetch_buffer
        self.use_fp16 = use_fp16
        
        # 创建LMDB环境
        self.env = lmdb.open(
            str(db_path),
            map_size=map_size,
            max_dbs=3,
            writemap=True,
            metasync=False,  # 提高写入性能
            sync=False,      # 异步写入，提高性能
            map_async=True,  # 异步内存映射
            lock=False       # 如果单进程可关闭锁
        )
        
        # 创建子数据库
        self.data_db = self.env.open_db(b'data')
        self.meta_db = self.env.open_db(b'meta')
        self.index_db = self.env.open_db(b'index')
        
        # 内存缓存（LRU）
        self.cache = {}
        self.cache_order = []  # 最近访问顺序
        self.cache_size = cache_size
        self.cache_lock = Lock()
        
        # 统计信息
        self.stats = {
            'write_count': 0,
            'read_count': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_write_time': 0.0,
            'total_read_time': 0.0
        }
        
        # 预读取管理
        self.prefetch_queue = []
        self.prefetch_lock = Lock()
        
        # 索引表（快速查找）
        self._load_index()
    
    def _load_index(self):
        """加载索引到内存"""
        self.index = {}  # key -> FeatureInfo
        with self.env.begin(db=self.index_db) as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                info = msgpack.loads(value)
                self.index[key.decode()] = FeatureInfo(**info)
    
    def _update_index(self, key: str, shape: tuple, dtype: str, compressed_size: int):
        """更新索引"""
        info = FeatureInfo(
            shape=shape,
            dtype=dtype,
            compressed_size=compressed_size,
            timestamp=time.time()
        )
        self.index[key] = info
        
        # 保存到数据库
        with self.env.begin(db=self.index_db, write=True) as txn:
            txn.put(key.encode(), msgpack.dumps(info.__dict__))
    
    def _compress_tensor(self, tensor: torch.Tensor) -> bytes:
        """高效压缩张量"""
        # 转换为numpy并确保内存连续
        np_array = tensor.detach().cpu().numpy()
        
        # 使用半精度浮点数存储（节省50%空间）
        if self.use_fp16 and np_array.dtype == np.float32:
            np_array = np_array.astype(np.float16)  # 压缩为半精度
            dtype_flag = b'float16'
        else:
            dtype_flag = str(np_array.dtype).encode()
        
        # 序列化
        data = msgpack.dumps({
            'shape': np_array.shape,
            'dtype': dtype_flag,
            'data': np_array.tobytes()
        })
        
        # 压缩
        if self.compression_level > 0:
            data = zlib.compress(data, level=self.compression_level)
        
        return data
    
    def _decompress_tensor(self, data: bytes, device: Optional[str] = None) -> torch.Tensor:
        """快速解压张量"""
        start_time = time.time()
        
        # 解压
        if self.compression_level > 0:
            data = zlib.decompress(data)
        
        info = msgpack.loads(data)
        
        # 重建numpy数组
        if self.use_fp16 and info['dtype'] == b'float16':
            np_array = np.frombuffer(info['data'], dtype=np.float16)
            np_array = np_array.reshape(info['shape'])
            # 转回float32
            tensor = torch.from_numpy(np_array.astype(np.float32))
        else:
            np_array = np.frombuffer(info['data'], dtype=info['dtype'])
            np_array = np_array.reshape(info['shape'])
            tensor = torch.from_numpy(np_array)
        
        # 移动到设备
        if device is None:
            device = self.device
        
        if device.startswith('cuda'):
            tensor = tensor.pin_memory()  # 固定内存，加速GPU传输
            tensor = tensor.to(device, non_blocking=True)  # 异步传输
        else:
            tensor = tensor.to(device)
        
        return tensor
    
    def _add_to_cache(self, key: str, tensor: torch.Tensor):
        """添加到LRU缓存"""
        with self.cache_lock:
            if key in self.cache:
                # 更新访问顺序
                self.cache_order.remove(key)
            else:
                # 如果缓存满了，移除最久未使用的
                if len(self.cache) >= self.cache_size:
                    lru_key = self.cache_order.pop(0)
                    del self.cache[lru_key]
            
            # 添加新缓存
            self.cache[key] = tensor
            self.cache_order.append(key)
    
    def _get_from_cache(self, key: str) -> Optional[torch.Tensor]:
        """从缓存获取"""
        with self.cache_lock:
            if key in self.cache:
                # 更新访问顺序
                self.cache_order.remove(key)
                self.cache_order.append(key)
                self.stats['cache_hits'] += 1
                return self.cache[key]
            self.stats['cache_misses'] += 1
            return None
    
    def write_single(self, key: str, tensor: torch.Tensor, metadata: Optional[Dict] = None):
        """
        高效写入单个特征
        
        Args:
            key: 特征标识符
            tensor: 特征张量
            metadata: 额外元数据
        """
        start_time = time.time()
        
        # 压缩数据
        compressed_data = self._compress_tensor(tensor)
        
        # 开始事务
        with self.env.begin(write=True, db=self.data_db) as txn:
            # 写入特征数据
            txn.put(key.encode(), compressed_data)
            
        # 如果有元数据，写入元数据库
        if metadata:
            with self.env.begin(write=True, db=self.meta_db) as meta_txn:
                meta_txn.put(key.encode(), msgpack.dumps(metadata))
        
        # 更新索引
        self._update_index(
            key=key,
            shape=tensor.shape,
            dtype=str(tensor.dtype),
            compressed_size=len(compressed_data)
        )
        
        # 添加到缓存
        self._add_to_cache(key, tensor)
        
        # 更新统计
        write_time = time.time() - start_time
        self.stats['write_count'] += 1
        self.stats['total_write_time'] += write_time
    
    def read_batch(self, keys: List[str], batch_size: int = 8, prefetch: bool = True) -> List[torch.Tensor]:
        """
        高效批量读取（2或8个）
        
        Args:
            keys: 要读取的键列表
            batch_size: 批量大小（2或8）
            prefetch: 是否预读取
        """
        start_time = time.time()
        
        # 检查缓存
        tensors = []
        remaining_keys = []
        
        for key in keys[:batch_size]:  # 确保不超过batch_size
            cached = self._get_from_cache(key)
            if cached is not None:
                tensors.append(cached)
            else:
                remaining_keys.append(key)
        
        # 从数据库读取剩余的特征
        if remaining_keys:
            db_tensors = self._read_from_db_batch(remaining_keys)
            tensors.extend(db_tensors)
            
            # 添加到缓存
            for key, tensor in zip(remaining_keys, db_tensors):
                self._add_to_cache(key, tensor)
        
        # 更新统计
        read_time = time.time() - start_time
        self.stats['read_count'] += 1
        self.stats['total_read_time'] += read_time
        
        # 预读取下一批
        if prefetch and len(keys) > batch_size:
            self._prefetch_keys(keys[batch_size:batch_size + self.prefetch_buffer])
        
        return tensors
    
    def _read_from_db_batch(self, keys: List[str]) -> List[torch.Tensor]:
        """从数据库批量读取"""
        tensors = []
        
        # 使用单个事务读取所有键
        with self.env.begin(db=self.data_db) as txn:
            for key in keys:
                data = txn.get(key.encode())
                if data:
                    tensor = self._decompress_tensor(data, self.device)
                    tensors.append(tensor)
                else:
                    raise KeyError(f"Key not found: {key}")
        
        return tensors
    
    def _prefetch_keys(self, keys: List[str]):
        """预读取键列表"""
        # 在实际应用中，可以在这里实现异步预读取
        # 这里简化实现：直接读取并缓存
        with self.prefetch_lock:
            for key in keys:
                if key not in self.cache and key in self.index:
                    # 从数据库读取并缓存
                    with self.env.begin(db=self.data_db) as txn:
                        data = txn.get(key.encode())
                        if data:
                            tensor = self._decompress_tensor(data, 'cpu')  # 先读到CPU
                            self._add_to_cache(key, tensor)
    
    def read_single(self, key: str) -> torch.Tensor:
        """读取单个特征"""
        return self.read_batch([key], batch_size=1)[0]
    
    def get_info(self, key: str) -> Optional[FeatureInfo]:
        """获取特征信息"""
        return self.index.get(key)
    
    def exists(self, key: str) -> bool:
        """检查特征是否存在"""
        return key in self.index
    
    def delete(self, key: str):
        """删除特征"""
        with self.env.begin(write=True, db=self.data_db) as txn:
            txn.delete(key.encode())
        
        with self.env.begin(write=True, db=self.meta_db) as txn:
            txn.delete(key.encode())
        
        with self.env.begin(write=True, db=self.index_db) as txn:
            txn.delete(key.encode())
        
        # 从缓存和索引中移除
        with self.cache_lock:
            if key in self.cache:
                del self.cache[key]
                self.cache_order.remove(key)
        
        if key in self.index:
            del self.index[key]
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        stats = self.stats.copy()
        
        # 计算平均值
        if stats['write_count'] > 0:
            stats['avg_write_time'] = stats['total_write_time'] / stats['write_count']
        if stats['read_count'] > 0:
            stats['avg_read_time'] = stats['total_read_time'] / stats['read_count']
        
        # 缓存命中率
        total_access = stats['cache_hits'] + stats['cache_misses']
        if total_access > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / total_access
        
        # 数据库大小
        if self.db_path.exists():
            data_size = sum(f.stat().st_size for f in self.db_path.rglob('*') if f.is_file())
            stats['db_size_gb'] = data_size / (1024**3)
            stats['total_features'] = len(self.index)
        
        return stats
    
    def optimize_compression(self, target_compression: int = 3):
        """
        优化压缩级别，平衡读写速度
        
        Args:
            target_compression: 目标压缩级别（1-9）
                - 1: 最快，压缩率低
                - 3: 推荐（平衡）
                - 6: 较好压缩，速度适中
                - 9: 最高压缩，速度慢
        """
        if target_compression != self.compression_level:
            old_level = self.compression_level
            self.compression_level = target_compression
            
            # 根据压缩级别调整其他参数
            if target_compression >= 6:
                # 高压缩，增加缓存以补偿读取速度
                self.cache_size = min(200, self.cache_size * 2)
            elif target_compression <= 2:
                # 低压缩，可减小缓存
                self.cache_size = max(50, self.cache_size // 2)
            
            print(f"压缩级别从 {old_level} 调整为 {target_compression}")
    
    def close(self):
        """关闭数据库"""
        self.env.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()



# ================== 示例 ==================
def real_usage_example():
    """实际使用示例"""
    print("=== 实际使用示例 ===")
    
    # 1. 初始化
    storage = OptimizedFeatureStorage(
        db_path='./real_features.lmdb',
        compression_level=3,
        cache_size=1,
    )
    
    try:
        # 2. 写入单个特征（例如来自模型提取）
        feature1 = torch.randn(256, 200, 200,dtype=torch.float32)
        storage.write_single('image_001', feature1, 
                            {'timestamp': ''})
                            
        feature2 = torch.randn(256, 200, 200,dtype=torch.float32)
        storage.write_single('image_002', feature1, 
                            {'timestamp': ''})
        
        print(f"已写入 2 个特征")
        
        # 3. 批量读取（例如训练时取一个batch）
        batch_keys = ['image_001']
        batch_tensors = storage.read_batch(batch_keys, batch_size=1)
        print((feature1-batch_tensors[0].to(feature1)).abs().max(),feature1.device,feature1.shape)
        
        print(f"批量读取 {len(batch_tensors)} 个特征")
        print(f"特征形状: {batch_tensors[0].shape}")
        
        # 4. 读取8个特征（更大的batch）
        # 先写入更多特征
        for i in range(3, 11):
            tensor = torch.randn(256, 14, 14)
            storage.write_single(f'image_{i:03d}', tensor)
        
        # 读取8个
        batch_keys = [f'image_{i:03d}' for i in range(1, 9)]
        batch_tensors = storage.read_batch(batch_keys, batch_size=8)
        
        print(f"批量读取 {len(batch_tensors)} 个特征")
        
        # 5. 检查信息
        info = storage.get_info('image_001')
        print(f"特征信息: {info}")
        
        # 6. 获取统计
        stats = storage.get_stats()
        print(f"\n统计信息:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
                
    finally:
        storage.close()


# ================== 主程序 ==================

if __name__ == "__main__":
    real_usage_example()
