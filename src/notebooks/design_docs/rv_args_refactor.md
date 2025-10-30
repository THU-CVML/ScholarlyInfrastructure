# RV Args重构方案

## 现状问题

1. RandomVariable类与optuna的BaseDistribution强耦合
2. 采样逻辑直接依赖optuna的Trial类
3. 搜索空间生成函数与optuna深度绑定

## 重构目标

1. 解耦分布定义与具体优化框架
2. 提供通用的采样接口
3. 支持用户自定义分布实现
4. 保持对optuna.distributions的直接支持
5. 保持现有API的向后兼容性

## 技术方案

### 1. 抽象分布接口

创建一个基础的Distribution抽象类：

```python
from dataclasses import dataclass
from optuna.distributions import BaseDistribution

class Distribution(ABC):
    @abstractmethod
    def sample(self, rng=None):
        """从分布中采样一个值"""
        pass
    
    @abstractmethod
    def to_internal_repr(self):
        """转换为框架特定的内部表示"""
        pass
    
    @classmethod
    @abstractmethod
    def from_internal_repr(cls, internal_repr):
        """从框架特定的内部表示创建分布"""
        pass
```

基于dataclass实现常用分布：

```python
@dataclass
class UniformDistribution(Distribution):
    low: float
    high: float
    
    def sample(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(self.low, self.high)
    
    def to_internal_repr(self):
        return {
            "type": "uniform",
            "low": self.low,
            "high": self.high
        }
    
    @classmethod
    def from_internal_repr(cls, internal_repr):
        return cls(internal_repr["low"], 
                  internal_repr["high"])
```

### 2. 框架适配器设计

为每个支持的优化框架创建专门的适配器：

```python
class OptunaAdapter:
    @staticmethod
    def to_optuna_dist(dist):
        """将通用分布转换为optuna分布"""
        if isinstance(dist, BaseDistribution):
            return dist  # 如果已经是optuna分布，直接返回
        
        """将通用分布转换为optuna分布"""
        pass
    
    @staticmethod
    def from_optuna_dist(optuna_dist):
        """将optuna分布转换为通用分布"""
        pass
    
    def suggest(self, trial, name: str, dist):
        """使用optuna trial采样"""
        if isinstance(dist, BaseDistribution):
            return trial._suggest(name, dist)
        optuna_dist = OptunaAdapter.to_optuna_dist(dist)
        return trial._suggest(name, optuna_dist)
```

### 3. RandomVariable类改造

```python
from typing import Union

@dataclass
class RandomVariable(PythonField):
    description: str = "MISSING description."
    # 支持optuna的BaseDistribution和我们自己的Distribution
    distribution: Union[BaseDistribution, Distribution] = "MISSING distribution."
```

### 4. 错误处理

```python
class DistributionError(Exception):
    """分布相关的错误基类"""
    pass

class FrameworkAdapterError(Exception):
    """框架适配器相关的错误"""
    pass

class ValidationError(Exception):
    """参数验证错误"""
    pass
```

### 5. 向后兼容性

为了确保向后兼容性，我们会：

1. 保持现有的API签名不变
2. 在内部进行适配：

```python
def get_optuna_search_space(cls, frozen_rvs:set = None):
    search_space = {}
    adapter = OptunaAdapter()
    for field in fields(cls):
        field_name = field.name
        if frozen_rvs is not None and field_name in frozen_rvs:
            continue
        rv = field.metadata.get(rv_dataclass_metadata_key, None)
        if rv is None:
            raise ValidationError("Class needs to use ~RandomVariable fields")
        # 通过适配器转换分布
        search_space[field_name] = adapter.to_optuna_dist(rv.distribution)
    return search_space
```

### 6. 测试策略

为确保重构的正确性，我们需要：

1. 为现有功能编写完整的单元测试
2. 测试与optuna的兼容性：
   - 直接使用optuna分布
   - 使用自定义分布
3. 测试错误处理
4. 测试向后兼容性
5. 性能测试

### 7. 文档更新

需要更新的文档：

1. API参考文档
2. 使用教程
3. 迁移指南
4. 示例代码

### 8. 发布计划

1. 在0.x版本中添加新功能
2. 标记废弃的API
3. 在1.0版本中完成迁移