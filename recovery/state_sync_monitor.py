"""
状态同步监控器
监控前后端状态一致性，确保资源同步清理
"""
import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import requests
import subprocess
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class StateSnapshot:
    """状态快照"""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    frontend_agents: List[Dict] = field(default_factory=list)
    backend_agents: List[Dict] = field(default_factory=list)
    modal_sandboxes: List[Dict] = field(default_factory=list)
    system_resources: Dict = field(default_factory=dict)

class StateSyncMonitor:
    """状态同步监控器"""

    def __init__(self, api_base_url: str = "http://localhost:8001",
                 frontend_url: str = "http://localhost:5173"):
        self.api_base_url = api_base_url
        self.frontend_url = frontend_url
        self.monitoring_active = False
        self.monitoring_thread: Optional[threading.Thread] = None
        self.state_history: List[StateSnapshot] = []
        self.max_history_size = 100
        self.check_interval = 30  # 30秒检查间隔
        self.alert_threshold = 2  # 连续2次不一致才告警

    def start_monitoring(self):
        """启动状态同步监控"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("状态同步监控已启动")

    def stop_monitoring(self):
        """停止状态同步监控"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("状态同步监控已停止")

    def _monitoring_loop(self):
        """监控循环"""
        consecutive_inconsistencies = 0

        while self.monitoring_active:
            try:
                snapshot = self._collect_state_snapshot()
                self.state_history.append(snapshot)

                # 保持历史记录大小
                if len(self.state_history) > self.max_history_size:
                    self.state_history.pop(0)

                # 检查状态一致性
                is_consistent = self._check_state_consistency(snapshot)

                if not is_consistent:
                    consecutive_inconsistencies += 1
                    logger.warning(f"状态不一致检测 (第{consecutive_inconsistencies}次)")

                    if consecutive_inconsistencies >= self.alert_threshold:
                        self._handle_state_inconsistency(snapshot)
                        consecutive_inconsistencies = 0  # 重置计数
                else:
                    consecutive_inconsistencies = 0

                time.sleep(self.check_interval)

            except Exception as e:
                logger.error(f"状态监控循环错误: {e}")
                time.sleep(5)

    def _collect_state_snapshot(self) -> StateSnapshot:
        """收集状态快照"""
        snapshot = StateSnapshot()

        try:
            # 获取前端状态
            snapshot.frontend_agents = self._get_frontend_agents()
        except Exception as e:
            logger.warning(f"获取前端状态失败: {e}")

        try:
            # 获取后端状态
            snapshot.backend_agents = self._get_backend_agents()
        except Exception as e:
            logger.warning(f"获取后端状态失败: {e}")

        try:
            # 获取Modal sandbox状态
            snapshot.modal_sandboxes = self._get_modal_sandboxes()
        except Exception as e:
            logger.warning(f"获取Modal状态失败: {e}")

        try:
            # 获取系统资源状态
            snapshot.system_resources = self._get_system_resources()
        except Exception as e:
            logger.warning(f"获取系统资源状态失败: {e}")

        return snapshot

    def _get_frontend_agents(self) -> List[Dict]:
        """获取前端显示的agent状态"""
        # 这里应该通过前端API或DOM解析获取
        # 暂时返回模拟数据
        try:
            # 尝试访问前端状态API（如果存在）
            response = requests.get(f"{self.frontend_url}/api/state", timeout=5)
            if response.status_code == 200:
                return response.json().get("agents", [])
        except:
            pass

        # 如果没有专门的API，返回空列表，需要其他方式获取
        return []

    def _get_backend_agents(self) -> List[Dict]:
        """获取后端记录的agent状态"""
        try:
            response = requests.get(f"{self.api_base_url}/api/orchestrator/resumable", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return data.get("resumable_experiments", [])
        except Exception as e:
            logger.warning(f"获取后端agent状态失败: {e}")
            return []

    def _get_modal_sandboxes(self) -> List[Dict]:
        """获取Modal sandbox状态"""
        try:
            result = subprocess.run(
                ["modal", "list", "--json"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                sandboxes = json.loads(result.stdout)
                return [{"id": s.get("id"), "status": s.get("status"), "created": s.get("created")}
                       for s in sandboxes]
        except Exception as e:
            logger.warning(f"获取Modal sandbox状态失败: {e}")

        return []

    def _get_system_resources(self) -> Dict:
        """获取系统资源状态"""
        try:
            import psutil
            return {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "process_count": len(psutil.pids())
            }
        except ImportError:
            # psutil不可用时的fallback
            return {}
        except Exception as e:
            logger.warning(f"获取系统资源失败: {e}")
            return {}

    def _check_state_consistency(self, snapshot: StateSnapshot) -> bool:
        """检查状态一致性"""
        # 1. 检查前后端agent数量一致性
        frontend_count = len(snapshot.frontend_agents)
        backend_count = len(snapshot.backend_agents)

        if frontend_count != backend_count:
            logger.warning(f"前后端agent数量不一致: 前端={frontend_count}, 后端={backend_count}")
            return False

        # 2. 检查Modal sandbox与后端agent的一致性
        modal_count = len(snapshot.modal_sandboxes)
        running_backend_agents = len([a for a in snapshot.backend_agents if a.get("status") == "running"])

        # Modal sandbox数量不应超过运行中的backend agents
        if modal_count > running_backend_agents:
            logger.warning(f"Modal sandbox数量异常: Modal={modal_count}, 运行中agents={running_backend_agents}")
            return False

        # 3. 检查agent ID一致性
        frontend_ids = {a.get("agent_id") for a in snapshot.frontend_agents}
        backend_ids = {a.get("agent_id") for a in snapshot.backend_agents}

        if frontend_ids != backend_ids:
            logger.warning(f"前后端agent ID不一致: 前端={frontend_ids}, 后端={backend_ids}")
            return False

        return True

    def _handle_state_inconsistency(self, snapshot: StateSnapshot):
        """处理状态不一致"""
        logger.error("检测到严重状态不一致，开始自动修复...")

        try:
            # 1. 强制同步清理Modal资源
            self._cleanup_orphaned_modal_resources(snapshot)

            # 2. 通知后端重置状态
            self._reset_backend_state()

            # 3. 记录不一致事件
            self._log_inconsistency_event(snapshot)

        except Exception as e:
            logger.error(f"状态不一致修复失败: {e}")

    def _cleanup_orphaned_modal_resources(self, snapshot: StateSnapshot):
        """清理孤立的Modal资源"""
        backend_agent_ids = {a.get("agent_id") for a in snapshot.backend_agents}
        modal_sandbox_ids = {s.get("id") for s in snapshot.modal_sandboxes}

        # 找出可能孤立的sandbox（没有对应agent的sandbox）
        orphaned_sandboxes = []
        for sandbox in snapshot.modal_sandboxes:
            sandbox_id = sandbox.get("id")
            if sandbox_id and sandbox_id not in backend_agent_ids:
                orphaned_sandboxes.append(sandbox_id)

        if orphaned_sandboxes:
            logger.warning(f"发现孤立的Modal sandboxes: {orphaned_sandboxes}")

            for sandbox_id in orphaned_sandboxes:
                try:
                    subprocess.run(
                        ["modal", "app", "stop", sandbox_id],
                        capture_output=True,
                        timeout=30
                    )
                    logger.info(f"清理孤立Modal sandbox: {sandbox_id}")
                except Exception as e:
                    logger.error(f"清理Modal sandbox {sandbox_id} 失败: {e}")

    def _reset_backend_state(self):
        """重置后端状态"""
        try:
            # 调用后端API重置状态
            response = requests.post(f"{self.api_base_url}/api/orchestrator/reset-state", timeout=10)
            if response.status_code == 200:
                logger.info("后端状态重置成功")
            else:
                logger.warning(f"后端状态重置失败: {response.status_code}")
        except Exception as e:
            logger.error(f"重置后端状态失败: {e}")

    def _log_inconsistency_event(self, snapshot: StateSnapshot):
        """记录不一致事件"""
        event = {
            "timestamp": snapshot.timestamp.isoformat(),
            "type": "state_inconsistency",
            "frontend_agents": snapshot.frontend_agents,
            "backend_agents": snapshot.backend_agents,
            "modal_sandboxes": snapshot.modal_sandboxes,
            "system_resources": snapshot.system_resources
        }

        # 保存到日志文件
        log_file = Path(".orchestrator_state") / "inconsistency_events.jsonl"
        log_file.parent.mkdir(exist_ok=True)

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

        logger.info(f"状态不一致事件已记录到 {log_file}")

    def get_system_health_report(self) -> Dict:
        """获取系统健康报告"""
        if not self.state_history:
            return {"status": "no_data", "message": "暂无监控数据"}

        latest_snapshot = self.state_history[-1]

        # 计算健康指标
        health_score = 100
        issues = []

        # 检查资源使用率
        resources = latest_snapshot.system_resources
        if resources.get("cpu_percent", 0) > 80:
            health_score -= 20
            issues.append("CPU使用率过高")

        if resources.get("memory_percent", 0) > 85:
            health_score -= 20
            issues.append("内存使用率过高")

        # 检查状态一致性
        is_consistent = self._check_state_consistency(latest_snapshot)
        if not is_consistent:
            health_score -= 30
            issues.append("状态不一致")

        # 检查孤立资源
        modal_count = len(latest_snapshot.modal_sandboxes)
        backend_running = len([a for a in latest_snapshot.backend_agents if a.get("status") == "running"])
        if modal_count > backend_running:
            health_score -= 25
            issues.append("存在孤立的Modal资源")

        # 确定健康状态
        if health_score >= 80:
            status = "healthy"
        elif health_score >= 60:
            status = "warning"
        else:
            status = "critical"

        return {
            "status": status,
            "health_score": max(0, health_score),
            "issues": issues,
            "timestamp": latest_snapshot.timestamp.isoformat(),
            "frontend_agents": len(latest_snapshot.frontend_agents),
            "backend_agents": len(latest_snapshot.backend_agents),
            "modal_sandboxes": modal_count,
            "system_resources": resources
        }

    def force_state_sync(self) -> Dict:
        """强制状态同步"""
        try:
            snapshot = self._collect_state_snapshot()
            is_consistent = self._check_state_consistency(snapshot)

            if not is_consistent:
                self._handle_state_inconsistency(snapshot)
                return {"success": True, "message": "状态同步完成", "was_consistent": False}
            else:
                return {"success": True, "message": "状态已一致", "was_consistent": True}

        except Exception as e:
            return {"success": False, "message": f"状态同步失败: {e}"}

# 全局实例
state_sync_monitor = StateSyncMonitor()