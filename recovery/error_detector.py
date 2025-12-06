"""
错误检测和自动恢复系统
监控agent执行过程中的常见错误模式并自动修复
"""
import re
import logging
import time
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import json

logger = logging.getLogger(__name__)

class ErrorType(Enum):
    """错误类型分类"""
    IMPORT_ERROR = "import_error"
    VARIABLE_ERROR = "variable_error"
    TYPE_ERROR = "type_error"
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT_ERROR = "timeout_error"
    RESOURCE_ERROR = "resource_error"
    NETWORK_ERROR = "network_error"
    MODAL_ERROR = "modal_error"
    UNKNOWN_ERROR = "unknown_error"

@dataclass
class ErrorPattern:
    """错误模式定义"""
    pattern: str
    error_type: ErrorType
    severity: str  # low, medium, high, critical
    auto_fixable: bool
    fix_strategy: Optional[str] = None
    fix_code: Optional[str] = None

class ErrorDetector:
    """错误检测器"""

    def __init__(self):
        self.error_patterns = self._initialize_error_patterns()
        self.error_history: Dict[str, List[Dict]] = {}  # agent_id -> error_list
        self.fix_strategies = self._initialize_fix_strategies()

    def _initialize_error_patterns(self) -> List[ErrorPattern]:
        """初始化错误模式"""
        return [
            # 导入错误
            ErrorPattern(
                r"NameError: name '(\w+)' is not defined",
                ErrorType.VARIABLE_ERROR,
                "medium",
                True,
                "variable_import",
                "import {variable}"
            ),
            ErrorPattern(
                r"ImportError: cannot import name '(\w+)' from '(\w+)'",
                ErrorType.IMPORT_ERROR,
                "medium",
                True,
                "alternative_import",
                "from {module} import {name}"
            ),
            ErrorPattern(
                r"ModuleNotFoundError: No module named '(\w+)'",
                ErrorType.IMPORT_ERROR,
                "medium",
                True,
                "install_package",
                "!pip install {package}"
            ),

            # 类型错误
            ErrorPattern(
                r"TypeError: '(\w+)' object is not (\w+)",
                ErrorType.TYPE_ERROR,
                "medium",
                True,
                "type_conversion",
                "converted_value = {conversion_function}(value)"
            ),
            ErrorPattern(
                r"TypeError:.*'float' object is not iterable",
                ErrorType.TYPE_ERROR,
                "medium",
                True,
                "float_iteration_fix",
                "if isinstance(value, float): value = [value]"
            ),

            # 语法错误
            ErrorPattern(
                r"SyntaxError:.*",
                ErrorType.SYNTAX_ERROR,
                "high",
                True,
                "syntax_fix",
                "syntax_correction"
            ),

            # Modal相关错误
            ErrorPattern(
                r"modal\.Exception:.*",
                ErrorType.MODAL_ERROR,
                "high",
                True,
                "modal_retry",
                "retry_with_different_config"
            ),
            ErrorPattern(
                r"TimeoutError:.*",
                ErrorType.TIMEOUT_ERROR,
                "medium",
                True,
                "timeout_handling",
                "increase_timeout_or_chunk"
            ),

            # 资源错误
            ErrorPattern(
                r"MemoryError:.*",
                ErrorType.RESOURCE_ERROR,
                "high",
                True,
                "memory_optimization",
                "reduce_memory_usage"
            ),
            ErrorPattern(
                r"CUDA out of memory",
                ErrorType.RESOURCE_ERROR,
                "high",
                True,
                "gpu_memory_fix",
                "reduce_batch_size_or_use_cpu"
            ),

            # 网络错误
            ErrorPattern(
                r"ConnectionError:.*",
                ErrorType.NETWORK_ERROR,
                "medium",
                True,
                "retry_with_backoff",
                "exponential_backoff_retry"
            ),
            ErrorPattern(
                r"HTTPError:.*",
                ErrorType.NETWORK_ERROR,
                "medium",
                True,
                "http_error_handling",
                "handle_http_errors"
            )
        ]

    def _initialize_fix_strategies(self) -> Dict[str, str]:
        """初始化修复策略"""
        return {
            "variable_import": """
# 自动导入缺失的变量
try:
    {variable}
except NameError:
    if '{variable}' == 'TfidfVectorizer':
        from sklearn.feature_extraction.text import TfidfVectorizer
    elif '{variable}' == 'np':
        import numpy as np
    elif '{variable}' == 'pd':
        import pandas as pd
    # 添加更多常见变量...
""",
            "alternative_import": """
# 尝试替代导入
try:
    from {module} import {name}
except ImportError:
    try:
        import {module}
        {name} = getattr({module}, '{name}')
    except:
        # 使用替代包
        if '{name}' == 'TfidfVectorizer':
            from sklearn.feature_extraction.text import TfidfVectorizer
""",
            "install_package": """
# 安装缺失的包
!pip install {package}
import {package}
""",
            "type_conversion": """
# 类型转换修复
if isinstance(value, float) and expected_type == "iterable":
    value = [value]
elif isinstance(value, str) and expected_type == "int":
    value = int(value)
""",
            "float_iteration_fix": """
# 修复float迭代错误
if isinstance(data, float):
    data = [data]  # 将float转换为单元素列表
elif isinstance(data, (int, str)):
    data = [data]
""",
            "syntax_fix": """
# 语法修复 - 需要人工检查或更复杂的修复逻辑
print("检测到语法错误，请检查代码语法")
""",
            "modal_retry": """
# Modal错误重试策略
import time
time.sleep(5)  # 等待5秒后重试
# 可能需要调整Modal配置或使用不同的GPU
""",
            "timeout_handling": """
# 超时处理
import signal
import time

class TimeoutHandler:
    def __init__(self, timeout_seconds=300):
        self.timeout_seconds = timeout_seconds

    def __enter__(self):
        signal.signal(signal.SIGALRM, self._handle_timeout)
        signal.alarm(self.timeout_seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)

    def _handle_timeout(self, signum, frame):
        raise TimeoutError("操作超时")

# 使用方式
try:
    with TimeoutHandler(600):  # 增加超时时间
        # 你的代码
        pass
except TimeoutError:
    print("操作超时，尝试分块处理")
""",
            "memory_optimization": """
# 内存优化
import gc

def process_in_chunks(data, chunk_size=1000):
    \"\"\"分块处理大数据\"\"\"
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        yield chunk
        gc.collect()  # 强制垃圾回收

# 使用方式
for chunk in process_in_chunks(large_data):
    result = process_chunk(chunk)
""",
            "gpu_memory_fix": """
# GPU内存修复
import torch

if torch.cuda.is_available():
    torch.cuda.empty_cache()  # 清空GPU缓存

    # 减少batch size
    batch_size = max(1, batch_size // 2)

    # 或者切换到CPU
    device = "cpu"
"""
        }

    def detect_error(self, agent_id: str, error_output: str) -> Optional[ErrorPattern]:
        """检测错误类型"""
        for pattern in self.error_patterns:
            if re.search(pattern.pattern, error_output, re.IGNORECASE):
                # 记录错误历史
                if agent_id not in self.error_history:
                    self.error_history[agent_id] = []

                self.error_history[agent_id].append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error_type": pattern.error_type.value,
                    "severity": pattern.severity,
                    "pattern": pattern.pattern,
                    "output": error_output[:500]  # 保存前500字符
                })

                logger.warning(f"检测到错误 - Agent: {agent_id}, 类型: {pattern.error_type.value}")
                return pattern

        # 未知错误
        if agent_id not in self.error_history:
            self.error_history[agent_id] = []

        self.error_history[agent_id].append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_type": ErrorType.UNKNOWN_ERROR.value,
            "severity": "medium",
            "pattern": "unknown",
            "output": error_output[:500]
        })

        return None

    def get_fix_strategy(self, pattern: ErrorPattern, error_output: str) -> Optional[str]:
        """获取修复策略"""
        if not pattern.auto_fixable or not pattern.fix_strategy:
            return None

        strategy = self.fix_strategies.get(pattern.fix_strategy)
        if not strategy:
            return None

        # 解析错误信息并替换模板变量
        try:
            match = re.search(pattern.pattern, error_output, re.IGNORECASE)
            if match:
                groups = match.groups()
                if pattern.error_type == ErrorType.VARIABLE_ERROR and groups:
                    variable_name = groups[0]
                    strategy = strategy.replace("{variable}", variable_name)
                elif pattern.error_type == ErrorType.IMPORT_ERROR and len(groups) >= 2:
                    module_name, import_name = groups[0], groups[1]
                    strategy = strategy.replace("{module}", module_name).replace("{name}", import_name)
                elif pattern.error_type == ErrorType.IMPORT_ERROR and groups:
                    package_name = groups[0]
                    strategy = strategy.replace("{package}", package_name)

        except Exception as e:
            logger.warning(f"解析错误信息失败: {e}")

        return strategy

    def should_attempt_recovery(self, agent_id: str) -> bool:
        """判断是否应该尝试恢复"""
        if agent_id not in self.error_history:
            return True

        recent_errors = self.error_history[agent_id][-5:]  # 最近5个错误
        critical_errors = [e for e in recent_errors if e["severity"] == "critical"]

        # 如果有3个以上严重错误，不再尝试恢复
        if len(critical_errors) >= 3:
            return False

        # 如果同一错误类型重复出现3次以上，不再尝试恢复
        error_types = [e["error_type"] for e in recent_errors]
        for error_type in set(error_types):
            if error_types.count(error_type) >= 3:
                return False

        return True

    def get_error_summary(self, agent_id: str) -> Dict:
        """获取错误摘要"""
        if agent_id not in self.error_history:
            return {"total_errors": 0, "error_types": {}, "recent_errors": []}

        errors = self.error_history[agent_id]
        error_types = {}
        for error in errors:
            error_type = error["error_type"]
            error_types[error_type] = error_types.get(error_type, 0) + 1

        return {
            "total_errors": len(errors),
            "error_types": error_types,
            "recent_errors": errors[-10:],  # 最近10个错误
            "last_error": errors[-1] if errors else None
        }

class AutoRecoveryEngine:
    """自动恢复引擎"""

    def __init__(self, error_detector: ErrorDetector):
        self.error_detector = error_detector
        self.recovery_attempts: Dict[str, int] = {}  # agent_id -> attempts
        self.max_recovery_attempts = 3

    def attempt_recovery(self, agent_id: str, error_output: str, current_code: str = None) -> Tuple[bool, str]:
        """尝试自动恢复"""
        if not self.error_detector.should_attempt_recovery(agent_id):
            logger.info(f"Agent {agent_id} 不再尝试自动恢复")
            return False, "错误过于频繁，停止自动恢复"

        # 检测错误类型
        pattern = self.error_detector.detect_error(agent_id, error_output)
        if not pattern:
            logger.warning(f"无法识别Agent {agent_id} 的错误类型")
            return False, "无法识别的错误类型"

        if not pattern.auto_fixable:
            logger.info(f"Agent {agent_id} 的错误类型 {pattern.error_type.value} 不可自动修复")
            return False, f"错误类型 {pattern.error_type.value} 不可自动修复"

        # 获取修复策略
        fix_strategy = self.error_detector.get_fix_strategy(pattern, error_output)
        if not fix_strategy:
            return False, "无可用修复策略"

        # 记录恢复尝试
        self.recovery_attempts[agent_id] = self.recovery_attempts.get(agent_id, 0) + 1
        if self.recovery_attempts[agent_id] > self.max_recovery_attempts:
            logger.error(f"Agent {agent_id} 恢复尝试次数过多")
            return False, "恢复尝试次数过多"

        logger.info(f"为Agent {agent_id} 尝试自动恢复 (第{self.recovery_attempts[agent_id]}次)")
        return True, fix_strategy

    def generate_recovery_code(self, original_code: str, fix_strategy: str, error_pattern: ErrorPattern) -> str:
        """生成恢复后的代码"""
        # 简单的代码修复策略
        recovery_code = f"""
# 自动错误恢复代码 - {datetime.now(timezone.utc).isoformat()}
{fix_strategy}

# 原始代码
{original_code}
"""
        return recovery_code

# 全局实例
error_detector = ErrorDetector()
recovery_engine = AutoRecoveryEngine(error_detector)