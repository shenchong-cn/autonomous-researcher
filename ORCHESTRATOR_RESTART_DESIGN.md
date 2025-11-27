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

### 6.1 恢复功能入口设计

#### 6.1.1 主界面集成恢复 (LabNotebook.tsx)

恢复功能主要集成在现有的 `LabNotebook` 组件中，在初始状态区域添加恢复选项：

```typescript
// 在 LabNotebook.tsx 的初始状态区域添加恢复功能
{orchestrator.timeline.length === 0 && !isRunning && (
  <div className="min-h-[60vh] flex flex-col justify-center items-center space-y-12">

    {/* 恢复功能区域 - 优先显示 */}
    {resumableExperiments.length > 0 && (
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="w-full max-w-xl space-y-6"
      >
        <div className="bg-gradient-to-r from-blue-500/10 to-purple-500/10 border border-blue-500/20 rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <div className="w-8 h-8 rounded-full bg-blue-500/20 flex items-center justify-center">
              <svg className="w-4 h-4 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </div>
            <div>
              <h3 className="text-white font-medium">可恢复的实验</h3>
              <p className="text-[#86868b] text-sm">发现 {resumableExperiments.length} 个中断的实验</p>
            </div>
          </div>

          {/* 可恢复实验列表 */}
          <div className="space-y-3">
            {resumableExperiments.map((exp) => (
              <div key={exp.experiment_id} className="bg-black/40 rounded-lg p-4 border border-white/10">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-white text-sm font-medium truncate">
                    {exp.research_task.substring(0, 60)}...
                  </span>
                  <StatusBadge status={exp.status} />
                </div>

                <div className="flex items-center gap-4 text-xs text-[#86868b] mb-3">
                  <span>进度: {exp.current_step}/{exp.max_steps}</span>
                  <span>中断于: {formatTime(exp.updated_at)}</span>
                  <span>已完成: {exp.completed_experiments.length} 个实验</span>
                </div>

                <div className="flex gap-2">
                  <button
                    onClick={() => handleResume(exp.experiment_id)}
                    className="flex-1 px-3 py-2 bg-blue-500 text-white rounded-lg text-xs font-medium hover:bg-blue-600 transition-colors"
                  >
                    恢复执行
                  </button>
                  <button
                    onClick={() => handleViewDetails(exp.experiment_id)}
                    className="px-3 py-2 bg-white/10 text-white rounded-lg text-xs font-medium hover:bg-white/20 transition-colors"
                  >
                    查看详情
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      </motion.div>
    )}

    {/* 原有的 "Research Objective" 输入区域 */}
    <div className="space-y-6 text-center max-w-lg">
      {/* ... 现有代码 ... */}
    </div>
  </div>
)}
```

#### 6.1.2 浮动恢复按钮

在页面右上角（API Keys 按钮旁边）添加恢复按钮：

```typescript
// 在 LabNotebook.tsx 的固定按钮区域添加
{hasResumableExperiments && (
  <button
    onClick={() => setShowRecoveryPanel(true)}
    className="fixed top-16 right-4 z-50 flex items-center gap-2 px-3 py-2 rounded-lg bg-blue-500/10 border border-blue-500/30 text-blue-400 hover:text-blue-300 hover:border-blue-500/50 transition-all duration-300"
  >
    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
    </svg>
    <span className="text-xs font-medium">恢复实验</span>
    {resumableCount > 0 && (
      <span className="ml-1 px-1.5 py-0.5 bg-blue-500 text-white text-xs rounded-full">
        {resumableCount}
      </span>
    )}
  </button>
)}
```

### 6.2 核心组件设计

#### 6.2.1 恢复面板组件 (RecoveryPanel.tsx)

```typescript
interface RecoveryPanelProps {
  open: boolean;
  experiments: ResumableExperiment[];
  onResume: (experimentId: string, checkpointId?: string) => void;
  onClose: () => void;
}

export function RecoveryPanel({ open, experiments, onResume, onClose }: RecoveryPanelProps) {
  const [selectedExperiment, setSelectedExperiment] = useState<string | null>(null);
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string | null>(null);

  const selectedExp = experiments.find(exp => exp.experiment_id === selectedExperiment);

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4"
          onClick={onClose}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            className="bg-[#1d1d1f] border border-white/10 rounded-2xl max-w-4xl w-full max-h-[80vh] overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b border-white/10">
              <div>
                <h2 className="text-xl font-medium text-white">恢复中断的实验</h2>
                <p className="text-sm text-[#86868b] mt-1">选择要恢复的实验和检查点</p>
              </div>
              <button
                onClick={onClose}
                className="p-2 hover:bg-white/10 rounded-lg transition-colors"
              >
                <X className="w-5 h-5 text-[#86868b]" />
              </button>
            </div>

            {/* Content */}
            <div className="flex h-[600px]">
              {/* 实验列表 */}
              <div className="w-1/2 border-r border-white/10 overflow-y-auto p-6">
                <h3 className="text-sm font-medium text-white mb-4">可恢复的实验</h3>
                <div className="space-y-3">
                  {experiments.map((exp) => (
                    <motion.div
                      key={exp.experiment_id}
                      whileHover={{ scale: 1.02 }}
                      onClick={() => {
                        setSelectedExperiment(exp.experiment_id);
                        setSelectedCheckpoint(null);
                      }}
                      className={`p-4 rounded-lg border cursor-pointer transition-all ${
                        selectedExperiment === exp.experiment_id
                          ? 'bg-blue-500/10 border-blue-500/30'
                          : 'bg-white/5 border-white/10 hover:bg-white/10'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-white text-sm font-medium truncate">
                          {exp.research_task.substring(0, 50)}...
                        </span>
                        <StatusBadge status={exp.status} />
                      </div>

                      <div className="text-xs text-[#86868b] space-y-1">
                        <div>实验ID: {exp.experiment_id}</div>
                        <div>进度: {exp.current_step}/{exp.max_steps}</div>
                        <div>中断时间: {formatTime(exp.updated_at)}</div>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>

              {/* 检查点详情 */}
              <div className="w-1/2 overflow-y-auto p-6">
                {selectedExp ? (
                  <div className="space-y-6">
                    <div>
                      <h3 className="text-sm font-medium text-white mb-4">实验详情</h3>
                      <div className="bg-black/40 rounded-lg p-4 space-y-3">
                        <div className="text-sm text-white">
                          {selectedExp.research_task}
                        </div>
                        <div className="grid grid-cols-2 gap-4 text-xs text-[#86868b]">
                          <div>模式: {selectedExp.config.num_agents} Agent</div>
                          <div>模型: {selectedExp.model}</div>
                          <div>已完成: {selectedExp.completed_experiments.length}</div>
                          <div>总轮数: {selectedExp.config.max_rounds}</div>
                        </div>
                      </div>
                    </div>

                    <div>
                      <h4 className="text-sm font-medium text-white mb-3">可用检查点</h4>
                      <div className="space-y-2">
                        {selectedExp.available_checkpoints?.map((cp) => (
                          <motion.div
                            key={cp.id}
                            whileHover={{ scale: 1.01 }}
                            onClick={() => setSelectedCheckpoint(cp.id)}
                            className={`p-3 rounded-lg border cursor-pointer text-sm ${
                              selectedCheckpoint === cp.id
                                ? 'bg-blue-500/10 border-blue-500/30'
                                : 'bg-white/5 border-white/10 hover:bg-white/10'
                            }`}
                          >
                            <div className="flex items-center justify-between">
                              <span className="text-white">{cp.description}</span>
                              <span className="text-[#86868b] text-xs">
                                {formatTime(cp.created_at)}
                              </span>
                            </div>
                          </motion.div>
                        ))}
                      </div>
                    </div>

                    <div className="flex gap-3 pt-4">
                      <button
                        onClick={() => {
                          if (selectedExperiment) {
                            onResume(selectedExperiment, selectedCheckpoint || undefined);
                            onClose();
                          }
                        }}
                        disabled={!selectedExperiment}
                        className="flex-1 px-4 py-3 bg-blue-500 text-white rounded-lg font-medium hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                      >
                        恢复执行
                      </button>
                      <button
                        onClick={onClose}
                        className="px-4 py-3 bg-white/10 text-white rounded-lg font-medium hover:bg-white/20 transition-all"
                      >
                        取消
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-full text-[#86868b]">
                    选择一个实验查看详情
                  </div>
                )}
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
```

#### 6.2.2 状态管理 Hook (useRecovery.ts)

```typescript
export function useRecovery() {
  const [resumableExperiments, setResumableExperiments] = useState<ResumableExperiment[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [recoveryStatus, setRecoveryStatus] = useState<RecoveryStatus>('idle');

  useEffect(() => {
    checkForResumableExperiments();
  }, []);

  const checkForResumableExperiments = async () => {
    try {
      setIsLoading(true);
      const experiments = await api.getResumableExperiments();
      setResumableExperiments(experiments);
    } catch (error) {
      console.error('Failed to check for resumable experiments:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const resumeExperiment = async (experimentId: string, checkpointId?: string) => {
    try {
      setRecoveryStatus('resuming');
      await api.resumeExperiment(experimentId, checkpointId);
      setRecoveryStatus('success');

      // 刷新实验状态
      await checkForResumableExperiments();

      // 触发主实验状态更新
      window.location.reload(); // 或通过状态管理更新
    } catch (error) {
      setRecoveryStatus('error');
      console.error('Failed to resume experiment:', error);
    }
  };

  return {
    resumableExperiments,
    isLoading,
    recoveryStatus,
    resumeExperiment,
    checkForResumableExperiments
  };
}
```

### 6.3 API 接口扩展

#### 6.3.1 类型定义

```typescript
// 在 api.ts 中添加恢复相关接口
export interface ResumableExperiment {
  experiment_id: string;
  research_task: string;
  model: string;
  created_at: string;
  updated_at: string;
  current_step: number;
  max_steps: number;
  status: 'hung' | 'paused' | 'failed';
  completed_experiments: ExperimentResult[];
  available_checkpoints: CheckpointInfo[];
  config: OrchestratorConfig;
}

export interface CheckpointInfo {
  id: string;
  description: string;
  created_at: string;
  step: number;
}

export interface ResumeRequest {
  experiment_id: string;
  checkpoint_id?: string;
  force?: boolean;
}

export interface RecoveryStatus {
  success: boolean;
  message: string;
  resumed_from_step?: number;
  estimated_remaining_time?: string;
}
```

#### 6.3.2 API 函数

```typescript
// 恢复相关 API 函数
export async function getResumableExperiments(): Promise<ResumableExperiment[]> {
  const response = await fetch(`${API_BASE_URL}/api/orchestrator/resumable`);
  if (!response.ok) throw new Error('Failed to fetch resumable experiments');
  return response.json();
}

export async function getExperimentState(experimentId: string): Promise<OrchestratorState> {
  const response = await fetch(`${API_BASE_URL}/api/orchestrator/state/${experimentId}`);
  if (!response.ok) throw new Error('Failed to fetch experiment state');
  return response.json();
}

export async function resumeExperiment(request: ResumeRequest): Promise<RecoveryStatus> {
  const response = await fetch(`${API_BASE_URL}/api/orchestrator/resume`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request)
  });
  if (!response.ok) throw new Error('Failed to resume experiment');
  return response.json();
}

export async function createCheckpoint(experimentId: string, description: string): Promise<CheckpointInfo> {
  const response = await fetch(`${API_BASE_URL}/api/orchestrator/checkpoint`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ experiment_id: experimentId, description })
  });
  if (!response.ok) throw new Error('Failed to create checkpoint');
  return response.json();
}

export async function deleteCheckpoint(experimentId: string, checkpointId: string): Promise<boolean> {
  const response = await fetch(`${API_BASE_URL}/api/orchestrator/checkpoint/${checkpointId}`, {
    method: 'DELETE'
  });
  return response.ok;
}
```

### 6.4 用户体验流程

#### 6.4.1 自动检测恢复流程

```typescript
// 用户打开页面时的自动检测流程
useEffect(() => {
  const initializeRecovery = async () => {
    // 1. 检查是否有可恢复的实验
    const resumable = await api.getResumableExperiments();

    if (resumable.length > 0) {
      // 2. 显示恢复选项
      setResumableExperiments(resumable);

      // 3. 如果有卡住的实验，显示提示
      const hungExperiments = resumable.filter(exp => exp.status === 'hung');
      if (hungExperiments.length > 0) {
        showNotification({
          type: 'warning',
          title: '检测到卡住的实验',
          message: `发现 ${hungExperiments.length} 个可能卡住的实验，可以选择恢复执行。`,
          action: { label: '查看恢复选项', onClick: () => setShowRecoveryPanel(true) }
        });
      }
    }
  };

  initializeRecovery();
}, []);
```

#### 6.4.2 卡住状态实时检测

```typescript
// 使用 Server-Sent Events 实时监控实验状态
const useOrchestratorStatus = (experimentId: string) => {
  const [status, setStatus] = useState<OrchestratorStatus | null>(null);

  useEffect(() => {
    if (!experimentId) return;

    const eventSource = new EventSource(
      `${API_BASE_URL}/api/orchestrator/status/${experimentId}/stream`
    );

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setStatus(data);

      // 检测到卡住状态时的处理
      if (data.status === 'hung' && data.can_resume) {
        showHungDetectionDialog(data);
      }
    };

    return () => eventSource.close();
  }, [experimentId]);

  return status;
};

const showHungDetectionDialog = (status: OrchestratorStatus) => {
  showModal({
    title: '实验可能卡住',
    content: (
      <div className="space-y-4">
        <p>实验已 {formatTime(status.last_activity)} 没有活动，可能已经卡住。</p>
        <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-3">
          <p className="text-sm text-yellow-300">
            您可以选择等待更长时间，或者恢复实验到最近的检查点。
          </p>
        </div>
      </div>
    ),
    actions: [
      { label: '继续等待', variant: 'secondary', onClick: () => {} },
      { label: '恢复实验', variant: 'primary', onClick: () => handleResume(status.experiment_id) }
    ]
  });
};
```

### 6.5 视觉设计规范

#### 6.5.1 颜色系统

```css
/* 恢复功能专用颜色 */
:root {
  --recovery-primary: #3b82f6;        /* 蓝色 - 主要操作 */
  --recovery-secondary: #8b5cf6;      /* 紫色 - 次要操作 */
  --recovery-success: #10b981;        /* 绿色 - 成功状态 */
  --recovery-warning: #f59e0b;        /* 黄色 - 警告状态 */
  --recovery-error: #ef4444;          /* 红色 - 错误状态 */
  --recovery-bg: rgba(59, 130, 246, 0.1);  /* 背景蓝色 */
  --recovery-border: rgba(59, 130, 246, 0.2); /* 边框蓝色 */
}
```

#### 6.5.2 动画规范

```typescript
// 统一的动画配置
export const recoveryAnimations = {
  // 面板出现动画
  panelAppear: {
    initial: { opacity: 0, scale: 0.95, y: 20 },
    animate: { opacity: 1, scale: 1, y: 0 },
    exit: { opacity: 0, scale: 0.95, y: 20 },
    transition: { duration: 0.2, ease: "easeOut" }
  },

  // 卡片悬停动画
  cardHover: {
    whileHover: { scale: 1.02 },
    transition: { duration: 0.15, ease: "easeOut" }
  },

  // 状态指示器动画
  pulseIndicator: {
    animate: { scale: [1, 1.1, 1], opacity: [1, 0.7, 1] },
    transition: { duration: 2, repeat: Infinity }
  }
};
```

### 6.6 响应式设计

```typescript
// 响应式布局适配
export const RecoveryPanel = ({ open, experiments, onResume, onClose }: RecoveryPanelProps) => {
  return (
    <AnimatePresence>
      {open && (
        <motion.div className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4">
          <motion.div className="bg-[#1d1d1f] border border-white/10 rounded-2xl w-full max-h-[90vh] overflow-hidden
                         /* 响应式宽度 */
                         max-w-4xl lg:max-w-4xl md:max-w-2xl sm:max-w-full
                         /* 响应式高度 */
                         h-[80vh] lg:h-[80vh] md:h-[70vh] sm:h-[60vh]">

            {/* 移动端适配：单列布局 */}
            <div className="flex flex-col lg:flex-row md:flex-col sm:flex-col h-[600px] lg:h-[600px] md:h-[500px] sm:h-[400px]">
              {/* 实验列表 */}
              <div className="flex-1 lg:flex-1 md:flex-1 sm:flex-1 overflow-y-auto p-6
                         border-r lg:border-r border-white/10 md:border-r-0 sm:border-r-0">
                {/* ... */}
              </div>

              {/* 检查点详情 */}
              <div className="flex-1 lg:flex-1 md:flex-1 sm:flex-1 overflow-y-auto p-6">
                {/* ... */}
              </div>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};
```

### 6.7 实时状态更新

#### 6.7.1 Server-Sent Events 实现

```typescript
// 使用Server-Sent Events实时更新状态
const useOrchestratorStatus = (experimentId: string) => {
  const [status, setStatus] = useState<OrchestratorStatus | null>(null);

  useEffect(() => {
    if (!experimentId) return;

    const eventSource = new EventSource(
      `${API_BASE_URL}/api/orchestrator/status/${experimentId}/stream`
    );

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setStatus(data);
    };

    eventSource.onerror = (error) => {
      console.error('SSE connection error:', error);
    };

    return () => eventSource.close();
  }, [experimentId]);

  return status;
};
```

#### 6.7.2 恢复进度监控

```typescript
// 监控恢复过程的进度
const useRecoveryProgress = (experimentId: string) => {
  const [progress, setProgress] = useState<RecoveryProgress>({
    stage: 'idle',
    percentage: 0,
    message: ''
  });

  useEffect(() => {
    if (!experimentId) return;

    const eventSource = new EventSource(
      `${API_BASE_URL}/api/orchestrator/recovery/${experimentId}/progress`
    );

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setProgress(data);
    };

    return () => eventSource.close();
  }, [experimentId]);

  return progress;
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

### 7.3 前端错误处理

#### 7.3.1 网络错误处理

```typescript
// API 错误处理 Hook
const useApiErrorHandling = () => {
  const [error, setError] = useState<string | null>(null);

  const handleApiError = (error: Error, context: string) => {
    console.error(`${context}:`, error);

    if (error.message.includes('Failed to fetch')) {
      setError('网络连接失败，请检查网络连接后重试');
    } else if (error.message.includes('401')) {
      setError('认证失败，请检查 API 密钥配置');
    } else if (error.message.includes('404')) {
      setError('请求的资源不存在，可能实验已过期');
    } else if (error.message.includes('500')) {
      setError('服务器内部错误，请稍后重试');
    } else {
      setError(error.message);
    }

    // 5秒后自动清除错误
    setTimeout(() => setError(null), 5000);
  };

  return { error, handleApiError, clearError: () => setError(null) };
};
```

#### 7.3.2 恢复失败处理

```typescript
// 恢复失败的 UI 反馈
const RecoveryFailureHandler = ({
  error,
  onRetry,
  onAlternative
}: {
  error: string;
  onRetry: () => void;
  onAlternative: () => void;
}) => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-red-500/10 border border-red-500/30 rounded-lg p-4 mb-4"
    >
      <div className="flex items-start gap-3">
        <div className="w-5 h-5 rounded-full bg-red-500/20 flex items-center justify-center flex-shrink-0 mt-0.5">
          <svg className="w-3 h-3 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </div>
        <div className="flex-1">
          <h4 className="text-red-300 font-medium text-sm mb-1">恢复失败</h4>
          <p className="text-red-400/70 text-xs mb-3">{error}</p>
          <div className="flex gap-2">
            <button
              onClick={onRetry}
              className="px-3 py-1.5 bg-red-500/20 text-red-300 rounded text-xs font-medium hover:bg-red-500/30 transition-colors"
            >
              重试
            </button>
            <button
              onClick={onAlternative}
              className="px-3 py-1.5 bg-white/10 text-white rounded text-xs font-medium hover:bg-white/20 transition-colors"
            >
              开始新实验
            </button>
          </div>
        </div>
      </div>
    </motion.div>
  );
};
```

#### 7.3.3 状态同步问题处理

```typescript
// 处理前后端状态不一致的问题
const useStateSync = (experimentId: string) => {
  const [syncStatus, setSyncStatus] = useState<'synced' | 'syncing' | 'error'>('synced');
  const [lastSyncTime, setLastSyncTime] = useState<Date>(new Date());

  const syncState = async () => {
    setSyncStatus('syncing');
    try {
      // 获取服务器最新状态
      const serverState = await api.getExperimentState(experimentId);

      // 与本地状态对比
      const localState = getLocalState(experimentId);

      if (hasStateConflict(serverState, localState)) {
        // 处理状态冲突
        await resolveStateConflict(serverState, localState);
      }

      updateLocalState(serverState);
      setLastSyncTime(new Date());
      setSyncStatus('synced');
    } catch (error) {
      setSyncStatus('error');
      console.error('状态同步失败:', error);
    }
  };

  return { syncStatus, lastSyncTime, syncState };
};
```

#### 7.3.4 用户操作确认

```typescript
// 危险操作的二次确认
const ConfirmationDialog = ({
  open,
  title,
  message,
  confirmText = '确认',
  cancelText = '取消',
  onConfirm,
  onCancel,
  type = 'warning'
}: ConfirmationDialogProps) => {
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4"
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="bg-[#1d1d1f] border border-white/10 rounded-xl max-w-md w-full p-6"
          >
            <div className="flex items-center gap-3 mb-4">
              <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
                type === 'warning' ? 'bg-yellow-500/20' : 'bg-red-500/20'
              }`}>
                {type === 'warning' ? (
                  <svg className="w-4 h-4 text-yellow-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
                  </svg>
                ) : (
                  <svg className="w-4 h-4 text-red-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                )}
              </div>
              <h3 className="text-white font-medium">{title}</h3>
            </div>

            <p className="text-[#86868b] text-sm mb-6">{message}</p>

            <div className="flex gap-3">
              <button
                onClick={onCancel}
                className="flex-1 px-4 py-2 bg-white/10 text-white rounded-lg font-medium hover:bg-white/20 transition-colors"
              >
                {cancelText}
              </button>
              <button
                onClick={onConfirm}
                className={`flex-1 px-4 py-2 rounded-lg font-medium transition-colors ${
                  type === 'warning'
                    ? 'bg-yellow-500 text-black hover:bg-yellow-600'
                    : 'bg-red-500 text-white hover:bg-red-600'
                }`}
              >
                {confirmText}
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};
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