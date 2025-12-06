"""
Agent生命周期管理器
负责agent的创建、监控、清理和恢复
"""
import asyncio
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import json
import subprocess
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    """Agent状态枚举"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    IDLE = "idle"
    ERROR = "error"
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    KILLED = "killed"

@dataclass
class AgentProcess:
    """Agent进程信息"""
    agent_id: str
    experiment_id: str
    process: Optional[subprocess.Popen] = None
    modal_sandbox_id: Optional[str] = None
    status: AgentStatus = AgentStatus.INITIALIZING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    error_count: int = 0
    consecutive_errors: int = 0
    max_consecutive_errors: int = 3
    timeout_seconds: int = 600  # 10分钟超时
    checkpoint_interval: int = 60  # 1分钟检查点间隔
    resource_usage: Dict = field(default_factory=dict)

    def is_timed_out(self) -> bool:
        """检查是否超时"""
        return (datetime.now(timezone.utc) - self.last_activity).total_seconds() > self.timeout_seconds

    def should_create_checkpoint(self) -> bool:
        """检查是否应该创建检查点"""
        return (datetime.now(timezone.utc) - self.last_activity).total_seconds() > self.checkpoint_interval

class AgentLifecycleManager:
    """Agent生命周期管理器"""

    def __init__(self, state_dir: Path = None):
        self.state_dir = state_dir or Path(".orchestrator_state")
        self.agents: Dict[str, AgentProcess] = {}
        self.modal_sandbox_tracker: Dict[str, Set[str]] = {}  # experiment_id -> sandbox_ids
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        self.cleanup_interval = 30  # 30秒清理间隔

    def register_agent(self, agent_id: str, experiment_id: str, process: subprocess.Popen = None,
                      modal_sandbox_id: str = None) -> AgentProcess:
        """注册新的agent"""
        agent = AgentProcess(
            agent_id=agent_id,
            experiment_id=experiment_id,
            process=process,
            modal_sandbox_id=modal_sandbox_id
        )

        self.agents[agent_id] = agent

        # 跟踪Modal sandbox
        if modal_sandbox_id:
            if experiment_id not in self.modal_sandbox_tracker:
                self.modal_sandbox_tracker[experiment_id] = set()
            self.modal_sandbox_tracker[experiment_id].add(modal_sandbox_id)

        logger.info(f"注册agent {agent_id} for experiment {experiment_id}")
        return agent

    def update_agent_status(self, agent_id: str, status: AgentStatus, error_message: str = None):
        """更新agent状态"""
        if agent_id not in self.agents:
            logger.warning(f"尝试更新未注册的agent状态: {agent_id}")
            return

        agent = self.agents[agent_id]
        old_status = agent.status
        agent.status = status
        agent.last_activity = datetime.now(timezone.utc)

        # 错误计数
        if status == AgentStatus.ERROR:
            agent.consecutive_errors += 1
            agent.error_count += 1
            logger.error(f"Agent {agent_id} 错误 ({agent.consecutive_errors}/{agent.max_consecutive_errors}): {error_message}")
        else:
            agent.consecutive_errors = 0

        logger.info(f"Agent {agent_id} 状态更新: {old_status.value} -> {status.value}")

    def get_agent(self, agent_id: str) -> Optional[AgentProcess]:
        """获取agent信息"""
        return self.agents.get(agent_id)

    def get_experiment_agents(self, experiment_id: str) -> List[AgentProcess]:
        """获取实验的所有agents"""
        return [agent for agent in self.agents.values() if agent.experiment_id == experiment_id]

    def get_active_modal_sandboxes(self, experiment_id: str) -> Set[str]:
        """获取实验的活跃Modal sandboxes"""
        return self.modal_sandbox_tracker.get(experiment_id, set())

    def start_monitoring(self):
        """启动监控线程"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("Agent生命周期监控已启动")

    def stop_monitoring(self):
        """停止监控线程"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Agent生命周期监控已停止")

    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                self._check_agent_health()
                self._cleanup_dead_resources()
                time.sleep(self.cleanup_interval)
            except Exception as e:
                logger.error(f"监控循环错误: {e}")
                time.sleep(5)

    def _check_agent_health(self):
        """检查agent健康状态"""
        current_time = datetime.now(timezone.utc)

        for agent_id, agent in list(self.agents.items()):
            try:
                # 检查超时
                if agent.is_timed_out() and agent.status == AgentStatus.RUNNING:
                    logger.warning(f"Agent {agent_id} 超时，标记为TIMEOUT")
                    self.update_agent_status(agent_id, AgentStatus.TIMEOUT, "执行超时")
                    self._kill_agent(agent_id)
                    continue

                # 检查连续错误
                if agent.consecutive_errors >= agent.max_consecutive_errors:
                    logger.error(f"Agent {agent_id} 连续错误过多，强制终止")
                    self.update_agent_status(agent_id, AgentStatus.KILLED, "连续错误过多")
                    self._kill_agent(agent_id)
                    continue

                # 检查进程状态
                if agent.process and agent.process.poll() is not None:
                    exit_code = agent.process.poll()
                    if exit_code != 0 and agent.status == AgentStatus.RUNNING:
                        logger.error(f"Agent {agent_id} 进程异常退出，退出码: {exit_code}")
                        self.update_agent_status(agent_id, AgentStatus.ERROR, f"进程退出码: {exit_code}")
                    elif exit_code == 0:
                        self.update_agent_status(agent_id, AgentStatus.COMPLETED)

                # 检查Modal sandbox状态
                if agent.modal_sandbox_id:
                    self._check_modal_sandbox_health(agent)

            except Exception as e:
                logger.error(f"检查agent {agent_id} 健康状态时出错: {e}")

    def _check_modal_sandbox_health(self, agent: AgentProcess):
        """检查Modal sandbox健康状态"""
        try:
            # 这里应该调用Modal API检查sandbox状态
            # 暂时用模拟实现
            import subprocess
            result = subprocess.run(
                ["modal", "list", "--json"],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                sandboxes = json.loads(result.stdout)
                sandbox_found = False
                for sandbox in sandboxes:
                    if sandbox.get("id") == agent.modal_sandbox_id:
                        sandbox_found = True
                        if sandbox.get("status") == "FAILED":
                            self.update_agent_status(agent.agent_id, AgentStatus.ERROR, "Modal sandbox失败")
                        break

                if not sandbox_found and agent.status == AgentStatus.RUNNING:
                    self.update_agent_status(agent.agent_id, AgentStatus.ERROR, "Modal sandbox消失")

        except Exception as e:
            logger.warning(f"检查Modal sandbox状态失败: {e}")

    def _cleanup_dead_resources(self):
        """清理死亡资源"""
        for agent_id, agent in list(self.agents.items()):
            if agent.status in [AgentStatus.COMPLETED, AgentStatus.KILLED, AgentStatus.TIMEOUT]:
                # 清理进程
                if agent.process and agent.process.poll() is None:
                    try:
                        agent.process.terminate()
                        agent.process.wait(timeout=5)
                    except:
                        try:
                            agent.process.kill()
                            agent.process.wait(timeout=5)
                        except:
                            pass

                # 清理Modal sandbox
                if agent.modal_sandbox_id:
                    self._cleanup_modal_sandbox(agent.modal_sandbox_id)

                # 从活跃列表中移除
                if agent_id in self.agents:
                    del self.agents[agent_id]

                logger.info(f"清理agent {agent_id} 资源完成")

    def _cleanup_modal_sandbox(self, sandbox_id: str):
        """清理Modal sandbox"""
        try:
            subprocess.run(
                ["modal", "app", "stop", sandbox_id],
                capture_output=True,
                timeout=30
            )
            logger.info(f"清理Modal sandbox {sandbox_id}")
        except Exception as e:
            logger.warning(f"清理Modal sandbox {sandbox_id} 失败: {e}")

    def _kill_agent(self, agent_id: str):
        """终止agent"""
        if agent_id not in self.agents:
            return

        agent = self.agents[agent_id]

        # 终止进程
        if agent.process and agent.process.poll() is None:
            try:
                agent.process.terminate()
                agent.process.wait(timeout=5)
            except:
                try:
                    agent.process.kill()
                    agent.process.wait(timeout=5)
                except:
                    pass

        # 清理Modal sandbox
        if agent.modal_sandbox_id:
            self._cleanup_modal_sandbox(agent.modal_sandbox_id)
            # 从跟踪器中移除
            if agent.experiment_id in self.modal_sandbox_tracker:
                self.modal_sandbox_tracker[agent.experiment_id].discard(agent.modal_sandbox_id)

    def force_cleanup_experiment(self, experiment_id: str):
        """强制清理实验的所有资源"""
        agents = self.get_experiment_agents(experiment_id)
        for agent in agents:
            self._kill_agent(agent.agent_id)

        # 清理Modal sandbox跟踪
        if experiment_id in self.modal_sandbox_tracker:
            for sandbox_id in self.modal_sandbox_tracker[experiment_id]:
                self._cleanup_modal_sandbox(sandbox_id)
            del self.modal_sandbox_tracker[experiment_id]

        logger.info(f"强制清理实验 {experiment_id} 的所有资源")

    def get_system_status(self) -> Dict:
        """获取系统状态"""
        active_agents = len([a for a in self.agents.values() if a.status == AgentStatus.RUNNING])
        error_agents = len([a for a in self.agents.values() if a.status == AgentStatus.ERROR])
        completed_agents = len([a for a in self.agents.values() if a.status == AgentStatus.COMPLETED])

        total_sandboxes = sum(len(sandboxes) for sandboxes in self.modal_sandbox_tracker.values())

        return {
            "total_agents": len(self.agents),
            "active_agents": active_agents,
            "error_agents": error_agents,
            "completed_agents": completed_agents,
            "total_modal_sandboxes": total_sandboxes,
            "experiments": list(self.modal_sandbox_tracker.keys()),
            "monitoring_active": self.monitoring_active
        }

# 全局实例
lifecycle_manager = AgentLifecycleManager()