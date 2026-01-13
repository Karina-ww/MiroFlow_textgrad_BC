"""
日志查看和分析工具
提供统计、查询、导出等功能
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict


class LogViewer:
    """日志查看器"""

    def __init__(self, log_dir: str = "api_logs"):
        """
        初始化日志查看器

        Args:
            log_dir: 日志目录
        """
        self.log_dir = Path(log_dir)

    def get_stats_summary(self, model: Optional[str] = None, month: Optional[str] = None) -> Dict[str, Any]:
        """
        获取统计摘要

        Args:
            model: 模型名称（可选，为空则统计所有模型）
            month: 月份，格式 YYYY-MM（可选）

        Returns:
            统计摘要字典
        """
        stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "zero_output_calls": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "by_model": defaultdict(lambda: {
                "calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "zero_output_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }),
            "by_user": defaultdict(lambda: {
                "calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "zero_output_calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            })
        }

        stats_path = self.log_dir / "stats"
        if not stats_path.exists():
            return stats

        # 遍历所有统计文件
        for model_dir in stats_path.iterdir():
            if not model_dir.is_dir():
                continue

            current_model = model_dir.name

            # 如果指定了 model，跳过其他模型
            if model and current_model != model:
                continue

            for stats_file in model_dir.glob("*.jsonl"):
                # 如果指定了月份，过滤文件名
                if month and month not in stats_file.stem:
                    continue

                # 从文件名提取用户名
                user = stats_file.stem.rsplit('_', 2)[0]

                # 读取统计数据
                with open(stats_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())

                            prompt_tokens = entry.get("prompt_tokens", 0)
                            completion_tokens = entry.get("completion_tokens", 0)
                            total_tokens = entry.get("total_tokens", 0)
                            success = entry.get("success", True)

                            # 统计所有调用（包括失败的）
                            stats["total_calls"] += 1
                            stats["by_model"][current_model]["calls"] += 1
                            stats["by_user"][user]["calls"] += 1

                            # 统计成功/失败
                            if success:
                                stats["successful_calls"] += 1
                                stats["by_model"][current_model]["successful_calls"] += 1
                                stats["by_user"][user]["successful_calls"] += 1
                            else:
                                stats["failed_calls"] += 1
                                stats["by_model"][current_model]["failed_calls"] += 1
                                stats["by_user"][user]["failed_calls"] += 1

                            # 统计输出token为0的调用
                            if completion_tokens == 0:
                                stats["zero_output_calls"] += 1
                                stats["by_model"][current_model]["zero_output_calls"] += 1
                                stats["by_user"][user]["zero_output_calls"] += 1

                            # 累计 token 数量
                            stats["total_prompt_tokens"] += prompt_tokens
                            stats["total_completion_tokens"] += completion_tokens
                            stats["total_tokens"] += total_tokens

                            stats["by_model"][current_model]["prompt_tokens"] += prompt_tokens
                            stats["by_model"][current_model]["completion_tokens"] += completion_tokens
                            stats["by_model"][current_model]["total_tokens"] += total_tokens

                            stats["by_user"][user]["prompt_tokens"] += prompt_tokens
                            stats["by_user"][user]["completion_tokens"] += completion_tokens
                            stats["by_user"][user]["total_tokens"] += total_tokens

                        except Exception:
                            continue

        # 转换 defaultdict 为普通 dict
        stats["by_model"] = dict(stats["by_model"])
        stats["by_user"] = dict(stats["by_user"])

        return stats

    def get_recent_calls(self, model: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取最近的 API 调用记录

        Args:
            model: 模型名称（可选）
            limit: 返回数量限制

        Returns:
            调用记录列表
        """
        calls = []
        calls_path = self.log_dir / "calls"

        if not calls_path.exists():
            return calls

        # 收集所有日志文件
        log_files = []
        for model_dir in calls_path.iterdir():
            if not model_dir.is_dir():
                continue

            # 如果指定了 model，跳过其他模型
            if model and model_dir.name != model:
                continue

            for month_dir in model_dir.iterdir():
                if not month_dir.is_dir():
                    continue

                for log_file in month_dir.glob("*.jsonl"):
                    log_files.append(log_file)

        # 按修改时间排序，最新的在前
        log_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)

        # 读取最近的记录
        for log_file in log_files:
            if len(calls) >= limit:
                break

            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

                # 从文件末尾开始读取
                for line in reversed(lines):
                    if len(calls) >= limit:
                        break

                    try:
                        entry = json.loads(line.strip())
                        calls.append(entry)
                    except Exception:
                        continue

        return calls[:limit]

    def print_stats_summary(self, model: Optional[str] = None, month: Optional[str] = None):
        """
        打印统计摘要（格式化输出）

        Args:
            model: 模型名称（可选）
            month: 月份（可选）
        """
        stats = self.get_stats_summary(model, month)

        print("=" * 80)
        print("API 调用统计摘要")
        if model:
            print(f"模型: {model}")
        if month:
            print(f"月份: {month}")
        print("=" * 80)

        # 计算成功率
        success_rate = 0
        if stats['total_calls'] > 0:
            success_rate = (stats['successful_calls'] / stats['total_calls']) * 100

        print(f"\n总调用次数: {stats['total_calls']:,}")
        print(f"  ✓ 成功: {stats['successful_calls']:,}")
        print(f"  ✗ 失败: {stats['failed_calls']:,}")
        print(f"  ⚠ 输出token为0: {stats['zero_output_calls']:,}")
        print(f"  成功率: {success_rate:.1f}%")

        print(f"\n总输入 Tokens: {stats['total_prompt_tokens']:,}")
        print(f"总输出 Tokens: {stats['total_completion_tokens']:,}")
        print(f"总 Tokens: {stats['total_tokens']:,}")

        if stats["by_model"]:
            print("\n按模型统计:")
            print("-" * 80)
            for model_name, model_stats in stats["by_model"].items():
                model_success_rate = 0
                if model_stats['calls'] > 0:
                    model_success_rate = (model_stats['successful_calls'] / model_stats['calls']) * 100

                print(f"\n{model_name}:")
                print(f"  调用次数: {model_stats['calls']:,} (成功: {model_stats['successful_calls']:,}, 失败: {model_stats['failed_calls']:,})")
                print(f"  成功率: {model_success_rate:.1f}%")
                print(f"  输出token为0: {model_stats['zero_output_calls']:,}")
                print(f"  输入 Tokens: {model_stats['prompt_tokens']:,}")
                print(f"  输出 Tokens: {model_stats['completion_tokens']:,}")
                print(f"  总 Tokens: {model_stats['total_tokens']:,}")

        if stats["by_user"]:
            print("\n按用户统计:")
            print("-" * 80)
            for user_name, user_stats in stats["by_user"].items():
                user_success_rate = 0
                if user_stats['calls'] > 0:
                    user_success_rate = (user_stats['successful_calls'] / user_stats['calls']) * 100

                print(f"\n{user_name}:")
                print(f"  调用次数: {user_stats['calls']:,} (成功: {user_stats['successful_calls']:,}, 失败: {user_stats['failed_calls']:,})")
                print(f"  成功率: {user_success_rate:.1f}%")
                print(f"  输出token为0: {user_stats['zero_output_calls']:,}")
                print(f"  输入 Tokens: {user_stats['prompt_tokens']:,}")
                print(f"  输出 Tokens: {user_stats['completion_tokens']:,}")
                print(f"  总 Tokens: {user_stats['total_tokens']:,}")

        print("\n" + "=" * 80)

    def print_recent_calls(self, model: Optional[str] = None, limit: int = 5):
        """
        打印最近的调用记录

        Args:
            model: 模型名称（可选）
            limit: 显示数量
        """
        calls = self.get_recent_calls(model, limit)

        print("=" * 80)
        print(f"最近 {limit} 次 API 调用")
        if model:
            print(f"模型: {model}")
        print("=" * 80)

        for i, call in enumerate(calls, 1):
            print(f"\n--- 调用 #{i} ---")
            print(f"时间: {call.get('timestamp', 'N/A')}")
            print(f"模型: {call.get('model', 'N/A')}")
            print(f"用户: {call.get('user', 'N/A')}")
            print(f"成功: {'是' if call.get('success', False) else '否'}")

            if call.get('error'):
                print(f"错误: {call['error']}")

            print(f"输入 Tokens: {call.get('prompt_tokens', 0):,}")
            print(f"输出 Tokens: {call.get('completion_tokens', 0):,}")
            print(f"总 Tokens: {call.get('total_tokens', 0):,}")

            if call.get('duration_ms') is not None:
                print(f"耗时: {call['duration_ms']:.2f} ms")

        print("\n" + "=" * 80)

    def export_to_csv(
        self,
        output_file: str,
        model: Optional[str] = None,
        month: Optional[str] = None
    ):
        """
        导出统计数据到 CSV

        Args:
            output_file: 输出文件路径
            model: 模型名称（可选）
            month: 月份（可选）
        """
        import csv

        stats_path = self.log_dir / "stats"
        if not stats_path.exists():
            print("没有找到统计数据")
            return

        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                "时间戳", "模型", "用户",
                "输入Tokens", "输出Tokens", "总Tokens", "耗时(ms)", "成功"
            ])

            # 遍历统计文件
            for model_dir in stats_path.iterdir():
                if not model_dir.is_dir():
                    continue

                current_model = model_dir.name
                if model and current_model != model:
                    continue

                for stats_file in model_dir.glob("*.jsonl"):
                    if month and month not in stats_file.stem:
                        continue

                    user = stats_file.stem.rsplit('_', 2)[0]

                    with open(stats_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            try:
                                entry = json.loads(line.strip())
                                writer.writerow([
                                    entry.get("timestamp", ""),
                                    current_model,
                                    user,
                                    entry.get("prompt_tokens", 0),
                                    entry.get("completion_tokens", 0),
                                    entry.get("total_tokens", 0),
                                    entry.get("duration_ms", ""),
                                    "是" if entry.get("success", True) else "否"
                                ])
                            except Exception:
                                continue

        print(f"数据已导出到: {output_file}")


# 全局默认查看器
_default_viewer = LogViewer()


def get_stats_summary(model: Optional[str] = None, month: Optional[str] = None) -> Dict[str, Any]:
    """使用默认查看器获取统计摘要"""
    return _default_viewer.get_stats_summary(model, month)


def print_stats_summary(model: Optional[str] = None, month: Optional[str] = None):
    """使用默认查看器打印统计摘要"""
    _default_viewer.print_stats_summary(model, month)


def print_recent_calls(model: Optional[str] = None, limit: int = 5):
    """使用默认查看器打印最近的调用"""
    _default_viewer.print_recent_calls(model, limit)


def export_to_csv(output_file: str, model: Optional[str] = None, month: Optional[str] = None):
    """使用默认查看器导出数据"""
    _default_viewer.export_to_csv(output_file, model, month)
