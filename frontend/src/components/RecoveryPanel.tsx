import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X } from 'lucide-react';
import {
    ResumableExperiment,
    CheckpointInfo,
    RecoveryStatus
} from '../lib/api';

interface RecoveryPanelProps {
    open: boolean;
    experiments: ResumableExperiment[];
    onResume: (experimentId: string, checkpointId?: string) => void;
    onDelete: (experimentId: string) => void;
    onClose: () => void;
    recoveryStatus: RecoveryStatus;
    error: string | null;
    clearError: () => void;
}

export function RecoveryPanel({
    open,
    experiments,
    onResume,
    onDelete,
    onClose,
    recoveryStatus,
    error,
    clearError
}: RecoveryPanelProps) {
    const [selectedExperiment, setSelectedExperiment] = useState<string | null>(null);
    const [selectedCheckpoint, setSelectedCheckpoint] = useState<string | null>(null);

    const selectedExp = experiments.find(exp => exp.experiment_id === selectedExperiment);

    const formatTime = (timeString: string) => {
        try {
            const date = new Date(timeString);
            return date.toLocaleString('zh-CN', {
                month: 'short',
                day: 'numeric',
                hour: '2-digit',
                minute: '2-digit'
            });
        } catch {
            return timeString;
        }
    };

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'hung': return 'text-yellow-400';
            case 'paused': return 'text-blue-400';
            case 'failed': return 'text-red-400';
            default: return 'text-gray-400';
        }
    };

    const getStatusText = (status: string) => {
        switch (status) {
            case 'hung': return '卡住';
            case 'paused': return '暂停';
            case 'failed': return '失败';
            default: return status;
        }
    };

    const handleResume = () => {
        if (selectedExperiment) {
            onResume(selectedExperiment, selectedCheckpoint || undefined);
            onClose();
        }
    };

    const handleDelete = () => {
        if (selectedExperiment && window.confirm('确定要删除这个实验的状态吗？此操作不可撤销。')) {
            onDelete(selectedExperiment);
            onClose();
        }
    };

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

                        {/* Error display */}
                        {error && (
                            <div className="mx-6 mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
                                <div className="flex items-center justify-between">
                                    <p className="text-red-400 text-sm">{error}</p>
                                    <button
                                        onClick={clearError}
                                        className="text-red-400 hover:text-red-300 text-sm"
                                    >
                                        清除
                                    </button>
                                </div>
                            </div>
                        )}

                        {/* Content */}
                        <div className="flex h-[600px]">
                            {/* Experiments list */}
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
                                                <span className={`text-xs ${getStatusColor(exp.status)}`}>
                                                    {getStatusText(exp.status)}
                                                </span>
                                            </div>

                                            <div className="text-xs text-[#86868b] space-y-1">
                                                <div>实验ID: {exp.experiment_id}</div>
                                                <div>进度: {exp.current_step}/{exp.max_steps}</div>
                                                <div>中断时间: {formatTime(exp.updated_at)}</div>
                                                <div>已完成: {exp.completed_experiments.length} 个实验</div>
                                                <div>模型: {exp.model}</div>
                                            </div>
                                        </motion.div>
                                    ))}
                                </div>
                            </div>

                            {/* Checkpoint details */}
                            <div className="w-1/2 overflow-y-auto p-6">
                                {selectedExp ? (
                                    <div className="space-y-6">
                                        {/* Experiment details */}
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
                                                    <div>最大并行: {selectedExp.config.max_parallel}</div>
                                                    <div>GPU: {selectedExp.config.gpu || 'CPU'}</div>
                                                </div>
                                            </div>
                                        </div>

                                        {/* Checkpoints */}
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

                                        {/* Actions */}
                                        <div className="flex gap-3 pt-4">
                                            <button
                                                onClick={handleResume}
                                                disabled={!selectedExperiment || recoveryStatus === 'loading'}
                                                className="flex-1 px-4 py-3 bg-blue-500 text-white rounded-lg font-medium hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                                            >
                                                {recoveryStatus === 'loading' ? '恢复中...' : '恢复执行'}
                                            </button>
                                            <button
                                                onClick={handleDelete}
                                                disabled={!selectedExperiment || recoveryStatus === 'loading'}
                                                className="px-4 py-3 bg-red-500/20 text-red-400 rounded-lg font-medium hover:bg-red-500/30 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                                            >
                                                删除状态
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