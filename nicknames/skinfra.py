# 这是一个别名模块 (例如 skinfra.py 或 大师荟萃之楼.py)
#
# Python 3.7+
import sys
import warnings

# 1. 导入你的真实模块
import scholarly_infrastructure 

# 2. (可选但推荐) 警告用户他们正在使用别名
# warnings.warn(
#     f"您正在通过别名 '{__name__}' 导入本库。 "
#     f"推荐使用 'import scholarly_infrastructure' 来导入。",
#     DeprecationWarning,
#     stacklevel=2
# )

# 3. 关键的“魔术”：
# 将当前模块 (例如 'skinfra') 在系统模块缓存 (sys.modules) 中的条目
# 替换为“真实”模块 ('scholarly_infrastructure') 的条目。
sys.modules[__name__] = sys.modules['scholarly_infrastructure']