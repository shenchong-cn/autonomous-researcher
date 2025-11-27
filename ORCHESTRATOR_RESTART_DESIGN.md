# Orchestrator 重启恢复功能设计文档

## 1. 问题分析

### 1.1 当前问题
- **无状态持久化**：orchestrator的所有状态都保存在内存中，重启后全部丢失
- **实验结果丢失**：已完成的agent实验结果无法复用，需要重新运行
- **卡住检测困难**：无法判断orchestrator是正常工作中还是已经卡住
- **恢复机制缺失**：无法从中断点继续执行，只能从头开始

### 1.2 影响分析
- **资源浪费**：重复运行已完成的experiment，浪费API调用和时间
- **用户体验差**：长时间等待无结果，无法判断系统状态
- **可靠性低**：单点故障导致整个研究流程失败

## 2. 设计目标

### 2.1 核心功能
- ✅ **状态持久化**：定期保存orchestrator运行状态
- ✅ **断点续传**：从中断点恢复执行，复用已完成结果
- ✅ **智能重启**：检测卡住状态并自动重启
- ✅ **进度可视化**：显示当前进度和可恢复的检查点

### 2.2 非功能性需求
- **性能影响最小**：持久化操作不应显著影响正常运行
- **数据一致性**：确保保存的状态数据完整和一致
- **向后兼容**：不破坏现有API和功能
- **错误恢复**：处理各种异常情况下的状态恢复

## 3. 架构设计

### 3.1 整体架构
```
┌─────────────────────────────────────────────────────────┐
│                    Orchestrator                          │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────────────────┐  │
│  │   状态管理器      │◄──►│     持久化存储               │  │
│  │ StateManager    │    │   Persistence Layer        │  │
│  └─────────────────┘    └─────────────────────────────┘  │
│           │                           │                   │
│           ▼                           ▼                   │
│  ┌─────────────────┐    ┌─────────────────────────────┐  │
│  │   进度跟踪器      │    │     检查点管理器             │  │
│  ProgressTracker  │    │   CheckpointManager         │  │
│  └─────────────────┘    └─────────────────────────────┘  │
│           │                           │                   │
│           ▼                           ▼                   │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              重启恢复引擎                             │  │
│  │            RecoveryEngine                          │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### 3.2 核心组件

#### 3.2.1 StateManager（状态管理器）
```python
class StateManager:
    """管理orchestrator的运行状态"""

    def __init__(self):
        self.current_state = OrchestratorState()
        self.checkpoint_manager = CheckpointManager()
        self.progress_tracker = ProgressTracker()

    def save_state(self, context: Dict[str, Any]) -> bool:
        """保存当前状态"""
        pass

    def load_state(self, experiment_id: str) -> Optional[OrchestratorState]:
        """加载保存的状态"""
        pass

    def can_resume(self, experiment_id: str) -> bool:
        """检查是否可以恢复执行"""
        pass
```

#### 3.2.2 CheckpointManager（检查点管理器）
```python
class CheckpointManager:
    """管理检查点的创建和恢复"""

    def create_checkpoint(self, state: OrchestratorState) -> str:
        """创建检查点，返回检查点ID"""
        pass

    def restore_from_checkpoint(self, checkpoint_id: str) -> OrchestratorState:
        """从检查点恢复状态"""
        pass

    def cleanup_old_checkpoints(self, max_age_days: int = 7):
        """清理过期检查点"""
        pass
```

#### 3.2.3 RecoveryEngine（恢复引擎）
```python
class RecoveryEngine:
    """处理重启恢复逻辑"""

    def detect_hang(self, last_activity: datetime, timeout_minutes: int = 10) -> bool:
        """检测是否卡住"""
        pass

    def graceful_restart(self, experiment_id: str) -> bool:
        """优雅重启orchestrator"""
        pass

    def resume_execution(self, state: OrchestratorState) -> bool:
        """恢复执行"""
        pass
```

## 4. 数据模型

### 4.1 OrchestratorState（状态数据结构）
```python
@dataclass
class OrchestratorState:
    """Orchestrator运行状态的完整描述"""

    # 基本信息
    experiment_id: str
    research_task: str
    model: str
    created_at: datetime
    updated_at: datetime

    # 执行状态
    current_step: int
    max_steps: int
    status: OrchestratorStatus  # RUNNING, PAUSED, COMPLETED, FAILED, HUNG

    # 实验结果
    completed_experiments: List[ExperimentResult]
    active_experiments: List[str]  # 正在运行的experiment ID

    # 对话历史
    message_history: List[Dict[str, Any]]

    # 配置信息
    config: OrchestratorConfig

    # 元数据
    metadata: Dict[str, Any]

@dataclass
class ExperimentResult:
    """单个实验的结果"""
    experiment_id: str
    hypothesis: str
    status: str  # COMPLETED, FAILED, RUNNING
    exit_code: int
    stdout: str
    stderr: str
    started_at: datetime
    completed_at: Optional[datetime]
    gpu: Optional[str]
    duration_seconds: float

enum OrchestratorStatus:
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    HUNG = "hung"
```

### 4.2 持久化存储格式
```json
{
    "experiment_id": "exp_20251127_001",
    "research_task": "AI systems can analyze Word documents larger than 100 pages...",
    "model": "claude-opus-4-5",
    "created_at": "2025-11-27T08:30:00Z",
    "updated_at": "2025-11-27T09:06:42Z",
    "current_step": 5,
    "max_steps": 9,
    "status": "hung",
    "completed_experiments": [
        {
            "experiment_id": "agent_1",
            "hypothesis": "Document structure analysis maintains performance...",
            "status": "completed",
            "exit_code": 0,
            "stdout": "experiment output...",
            "stderr": "",
            "started_at": "2025-11-27T08:35:00Z",
            "completed_at": "2025-11-27T09:00:00Z",
            "gpu": null,
            "duration_seconds": 1500.0
        }
    ],
    "active_experiments": [],
    "message_history": [...],
    "config": {
        "num_agents": 3,
        "max_rounds": 3,
        "max_parallel": 2,
        "gpu": null
    },
    "metadata": {
        "version": "1.0",
        "checkpoint_reason": "pre_final_report"
    }
}
```

## 5. 实现方案

### 5.1 状态持久化策略

#### 5.1.1 触发时机
- **定期保存**：每完成一个step后自动保存
- **关键节点**：启动、完成experiment、生成最终报告前
- **异常处理**：捕获异常时保存当前状态
- **手动触发**：用户主动保存检查点

#### 5.1.2 存储位置
```python
# 文件存储结构
PROJECT_ROOT/
├── .orchestrator_state/
│   ├── experiments/
│   │   ├── exp_20251127_001/
│   │   │   ├── state.json          # 主状态文件
│   │   │   ├── checkpoints/        # 检查点目录
│   │   │   │   ├── step_3.json
│   │   │   │   ├── pre_final.json
│   │   │   │   └── ...
│   │   │   ├── experiments/        # 实验结果
│   │   │   │   ├── agent_1.json
│   │   │   │   ├── agent_2.json
│   │   │   │   └── agent_3.json
│   │   │   └── logs/              # 日志文件
│   │   │       ├── orchestrator.log
│   │   │       └── ...
│   │   └── exp_20251127_002/
│   │       └── ...
│   └── config.json                # 全局配置
```

### 5.2 重启恢复流程

#### 5.2.1 检测卡住
```python
def detect_hang(state: OrchestratorState) -> bool:
    """检测orchestrator是否卡住"""
    now = datetime.now()

    # 检查最后活动时间
    if (now - state.updated_at).total_seconds() > HANG_TIMEOUT_SECONDS:
        return True

    # 检查是否有长时间运行的experiment
    for exp in state.active_experiments:
        exp_start = get_experiment_start_time(exp)
        if (now - exp_start).total_seconds() > EXPERIMENT_TIMEOUT_SECONDS:
            return True

    return False
```

#### 5.2.2 恢复执行
```python
def resume_orchestrator(experiment_id: str) -> bool:
    """恢复orchestrator执行"""

    # 1. 加载保存的状态
    state = state_manager.load_state(experiment_id)
    if not state:
        logger.error(f"无法找到实验状态: {experiment_id}")
        return False

    # 2. 验证状态完整性
    if not validate_state(state):
        logger.error("状态数据不完整或损坏")
        return False

    # 3. 重建执行环境
    rebuild_execution_context(state)

    # 4. 从断点继续执行
    return continue_from_checkpoint(state)
```

### 5.3 API接口设计

#### 5.3.1 状态查询API
```http
GET /api/orchestrator/state/{experiment_id}
Response: {
    "experiment_id": "exp_20251127_001",
    "status": "hung",
    "progress": {
        "current_step": 5,
        "max_steps": 9,
        "completed_experiments": 3,
        "total_experiments": 3
    },
    "can_resume": true,
    "last_activity": "2025-11-27T09:06:42Z",
    "available_checkpoints": [
        {"id": "step_3", "description": "完成第3步后"},
        {"id": "pre_final", "description": "生成最终报告前"}
    ]
}
```

#### 5.3.2 重启恢复API
```http
POST /api/orchestrator/resume
Request: {
    "experiment_id": "exp_20251127_001",
    "checkpoint_id": "pre_final",  // 可选，默认使用最新
    "force": false  // 是否强制重启
}
Response: {
    "success": true,
    "message": "已从检查点恢复执行",
    "resumed_from_step": 5,
    "estimated_remaining_time": "5-10分钟"
}
```

#### 5.3.3 手动保存检查点API
```http
POST /api/orchestrator/checkpoint
Request: {
    "experiment_id": "exp_20251127_001",
    "description": "手动保存检查点"
}
Response: {
    "success": true,
    "checkpoint_id": "manual_20251127_0930",
    "created_at": "2025-11-27T09:30:00Z"
}
```

## 6. 前端集成

### 6.1 UI组件设计

#### 6.1.1 状态指示器
```typescript
interface OrchestratorStatusProps {
  experimentId: string;
  status: 'running' | 'hung' | 'paused' | 'completed';
  progress: number;
  canResume: boolean;
  onResume: () => void;
  onSaveCheckpoint: () => void;
}

const OrchestratorStatus: React.FC<OrchestratorStatusProps> = ({
  status,
  progress,
  canResume,
  onResume,
  onSaveCheckpoint
}) => {
  return (
    <div className="orchestrator-status">
      <div className="status-header">
        <StatusIndicator status={status} />
        <ProgressBar progress={progress} />
      </div>

      {status === 'hung' && canResume && (
        <div className="recovery-actions">
          <p>检测到系统可能卡住，您可以：</p>
          <button onClick={onResume}>恢复执行</button>
          <button onClick={onSaveCheckpoint}>保存检查点</button>
        </div>
      )}

      <CheckpointList experimentId={experimentId} />
    </div>
  );
};
```

#### 6.1.2 检查点管理器
```typescript
const CheckpointManager: React.FC<{experimentId: string}> = ({ experimentId }) => {
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);

  const handleResume = async (checkpointId: string) => {
    try {
      await api.resumeOrchestrator(experimentId, checkpointId);
      // 刷新状态
    } catch (error) {
      showError('恢复失败: ' + error.message);
    }
  };

  return (
    <div className="checkpoint-manager">
      <h4>可用检查点</h4>
      <ul>
        {checkpoints.map(cp => (
          <li key={cp.id}>
            <span>{cp.description}</span>
            <small>{cp.createdAt}</small>
            <button onClick={() => handleResume(cp.id)}>
              从此恢复
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
};
```

### 6.2 实时状态更新
```typescript
// 使用WebSocket或Server-Sent Events实时更新状态
const useOrchestratorStatus = (experimentId: string) => {
  const [status, setStatus] = useState<OrchestratorStatus | null>(null);

  useEffect(() => {
    const eventSource = new EventSource(
      `/api/orchestrator/status/${experimentId}/stream`
    );

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setStatus(data);
    };

    return () => eventSource.close();
  }, [experimentId]);

  return status;
};
```

## 7. 错误处理和边界情况

### 7.1 状态文件损坏
```python
def handle_corrupted_state(experiment_id: str) -> bool:
    """处理损坏的状态文件"""

    # 1. 尝试从备份恢复
    backup_path = get_backup_path(experiment_id)
    if os.path.exists(backup_path):
        try:
            return restore_from_backup(backup_path)
        except Exception as e:
            logger.error(f"备份恢复失败: {e}")

    # 2. 尝试重建部分状态
    try:
        return reconstruct_partial_state(experiment_id)
    except Exception as e:
        logger.error(f"状态重建失败: {e}")

    # 3. 如果都失败，标记为不可恢复
    mark_as_unrecoverable(experiment_id)
    return False
```

### 7.2 版本兼容性
```python
def migrate_state_format(state_data: dict, from_version: str, to_version: str) -> dict:
    """迁移状态数据格式"""

    if from_version == "1.0" and to_version == "1.1":
        # 添加新字段
        state_data["metadata"]["recovery_attempts"] = 0
        state_data["metadata"]["last_checkpoint_reason"] = None

    elif from_version == "1.1" and to_version == "1.2":
        # 修改字段结构
        state_data["config"]["timeout_settings"] = {
            "hang_timeout": 600,
            "experiment_timeout": 1800
        }

    return state_data
```

## 8. 性能优化

### 8.1 异步持久化
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class AsyncStateManager:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.pending_saves = set()

    async def save_state_async(self, state: OrchestratorState) -> str:
        """异步保存状态，不阻塞主流程"""

        save_id = generate_save_id()
        self.pending_saves.add(save_id)

        # 在后台线程中执行保存
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            self._save_state_sync,
            state,
            save_id
        )

        self.pending_saves.discard(save_id)
        return save_id

    def _save_state_sync(self, state: OrchestratorState, save_id: str):
        """同步保存状态的实际实现"""
        try:
            # 执行实际的文件保存操作
            save_to_file(state, save_id)
        except Exception as e:
            logger.error(f"状态保存失败: {e}")
            # 可以考虑重试机制
```

### 8.2 增量保存
```python
def save_incremental_state(current_state: OrchestratorState,
                          previous_state: OrchestratorState) -> bool:
    """只保存变化的部分"""

    # 计算差异
    changes = calculate_state_diff(current_state, previous_state)

    if not changes:
        return True  # 没有变化，不需要保存

    # 保存增量变化
    return save_state_changes(changes)
```

## 9. 测试策略

### 9.1 单元测试
```python
class TestStateManager:
    def test_save_and_load_state(self):
        """测试状态保存和加载"""
        state = create_test_state()
        save_id = state_manager.save_state(state)
        loaded_state = state_manager.load_state(save_id)

        assert loaded_state.experiment_id == state.experiment_id
        assert loaded_state.current_step == state.current_step

    def test_detect_hang(self):
        """测试卡住检测"""
        old_state = create_old_state(minutes=15)
        assert recovery_engine.detect_hang(old_state) == True

        recent_state = create_recent_state(minutes=5)
        assert recovery_engine.detect_hang(recent_state) == False

    def test_resume_execution(self):
        """测试恢复执行"""
        # 模拟中断场景
        state = create_interrupted_state()

        # 恢复执行
        success = recovery_engine.resume_execution(state)
        assert success == True

        # 验证恢复后的状态
        current_state = state_manager.get_current_state()
        assert current_state.status == OrchestratorStatus.RUNNING
```

### 9.2 集成测试
```python
class TestOrchestratorRecovery:
    def test_full_recovery_scenario(self):
        """测试完整的恢复场景"""

        # 1. 启动orchestrator
        experiment_id = start_orchestrator(test_task)

        # 2. 等待完成几个experiment
        wait_for_experiments(experiment_id, count=2)

        # 3. 模拟卡住（终止进程）
        terminate_orchestrator_gracefully(experiment_id)

        # 4. 检测卡住状态
        status = get_orchestrator_status(experiment_id)
        assert status.status == "hung"

        # 5. 恢复执行
        success = resume_orchestrator(experiment_id)
        assert success == True

        # 6. 验证完成
        wait_for_completion(experiment_id, timeout=300)
        final_status = get_orchestrator_status(experiment_id)
        assert final_status.status == "completed"
```

## 10. 部署和配置

### 10.1 配置文件
```json
{
    "orchestrator": {
        "recovery": {
            "enabled": true,
            "hang_timeout_seconds": 600,
            "experiment_timeout_seconds": 1800,
            "max_recovery_attempts": 3,
            "auto_cleanup_days": 7
        },
        "persistence": {
            "storage_path": ".orchestrator_state",
            "backup_enabled": true,
            "compression_enabled": true,
            "max_state_file_size_mb": 100
        },
        "checkpoints": {
            "auto_save_interval_steps": 1,
            "max_checkpoints_per_experiment": 10,
            "save_before_final_report": true
        }
    }
}
```

### 10.2 环境变量
```bash
# 启用/禁用恢复功能
ORCHESTRATOR_RECOVERY_ENABLED=true

# 状态存储路径
ORCHESTRATOR_STATE_PATH=/app/data/orchestrator_state

# 卡住检测超时（秒）
ORCHESTRATOR_HANG_TIMEOUT=600

# 实验超时（秒）
ORCHESTRATOR_EXPERIMENT_TIMEOUT=1800
```

## 11. 监控和日志

### 11.1 关键指标
- **状态保存成功率**：保存操作的成功/失败比例
- **恢复成功率**：恢复操作的成功/失败比例
- **检测准确率**：卡住检测的准确度
- **平均恢复时间**：从中断到恢复的平均时间

### 11.2 日志格式
```json
{
    "timestamp": "2025-11-27T09:30:00Z",
    "level": "INFO",
    "component": "StateManager",
    "experiment_id": "exp_20251127_001",
    "action": "save_state",
    "details": {
        "step": 5,
        "completed_experiments": 3,
        "file_size_bytes": 2048,
        "duration_ms": 150
    }
}
```

## 12. 实施计划

### Phase 1: 核心功能（2-3天）
- [ ] 实现StateManager基础功能
- [ ] 添加状态持久化
- [ ] 实现基本的恢复逻辑

### Phase 2: API和前端（2天）
- [ ] 添加恢复相关API
- [ ] 实现前端状态显示
- [ ] 添加手动恢复按钮

### Phase 3: 高级功能（2-3天）
- [ ] 实现智能卡住检测
- [ ] 添加自动恢复
- [ ] 性能优化和错误处理

### Phase 4: 测试和部署（1-2天）
- [ ] 完整测试覆盖
- [ ] 文档更新
- [ ] 生产环境部署

## 13. 风险评估

### 13.1 技术风险
- **数据一致性**：状态保存过程中的并发问题
- **性能影响**：持久化操作对主流程的影响
- **存储空间**：状态文件可能占用大量磁盘空间

### 13.2 缓解措施
- **原子操作**：使用原子写入确保数据一致性
- **异步处理**：后台处理持久化操作
- **定期清理**：自动清理过期和多余的状态文件

## 14. 总结

这个设计提供了一个完整的orchestrator重启恢复解决方案，主要特点：

1. **完整的状态管理**：保存和恢复orchestrator的所有关键状态
2. **智能检测机制**：自动检测系统卡住并触发恢复
3. **用户友好界面**：提供直观的状态显示和恢复操作
4. **高性能实现**：异步处理和增量保存最小化性能影响
5. **可靠性保证**：完善的错误处理和恢复机制

通过这个功能，用户不再需要担心orchestrator卡住或意外中断的问题，大大提升了系统的可靠性和用户体验。