"""
Local API Logger - 轻量级 LLM API 调用日志记录工具

一个极简的 Python 库，用于本地记录 LLM API 调用，包括：
- 请求和响应数据
- Token 使用统计
- 调用时长和成功率
- 按模型和用户分类的统计

特点：
- 零配置，开箱即用
- 无需数据库，使用 JSONL 格式存储
- 线程安全
- 支持流式和非流式响应
- 提供装饰器和包装器，易于集成

基本用法：
    ```python
    from local_api_logger import log_completion, print_stats_summary

    # 记录一次调用
    log_completion(
        model="claude-3-opus",
        request_data={"messages": [...]},
        response_data=response,  # 服务商返回的完整响应（包含 usage）
        user="john"
    )

    # 查看统计
    print_stats_summary()
    ```
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__license__ = "MIT"

# 导出核心类
from .logger import APILogger, log_call, set_log_dir
from .tracker import APITracker, track_request, log_completion, wrap_requests_call
from .viewer import LogViewer, get_stats_summary, print_stats_summary, print_recent_calls, export_to_csv
from .utils import (
    estimate_tokens,
    format_token_count,
    calculate_cost,
    truncate_string,
    extract_model_name,
    safe_get
)

# 定义公开的 API
__all__ = [
    # 版本信息
    "__version__",

    # 核心类
    "APILogger",
    "APITracker",
    "LogViewer",

    # Logger 函数
    "log_call",
    "set_log_dir",

    # Tracker 函数
    "track_request",
    "log_completion",
    "wrap_requests_call",

    # Viewer 函数
    "get_stats_summary",
    "print_stats_summary",
    "print_recent_calls",
    "export_to_csv",

    # 工具函数
    "estimate_tokens",
    "format_token_count",
    "calculate_cost",
    "truncate_string",
    "extract_model_name",
    "safe_get",
]
