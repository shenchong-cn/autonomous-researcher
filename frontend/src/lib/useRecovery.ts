import { useState, useEffect } from 'react';
import {
    getResumableExperiments,
    getExperimentStatus,
    resumeExperiment,
    createCheckpoint,
    deleteCheckpoint,
    getCheckpoints,
    triggerAutoRecovery,
    deleteExperimentState,
    ResumableExperiment,
    ExperimentStatus,
    CheckpointInfo,
    ResumeRequest,
    CheckpointRequest,
    CheckpointDeleteRequest
} from './api';

export type RecoveryStatus = 'idle' | 'loading' | 'success' | 'error';

export function useRecovery() {
    const [resumableExperiments, setResumableExperiments] = useState<ResumableExperiment[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [recoveryStatus, setRecoveryStatus] = useState<RecoveryStatus>('idle');
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        checkForResumableExperiments();
    }, []);

    const checkForResumableExperiments = async () => {
        try {
            setIsLoading(true);
            setRecoveryStatus('loading');
            setError(null);

            const response = await getResumableExperiments();
            setResumableExperiments(response.resumable_experiments || []);
            setRecoveryStatus('success');
        } catch (error) {
            console.error('Failed to check for resumable experiments:', error);
            setError(error instanceof Error ? error.message : 'Unknown error');
            setRecoveryStatus('error');
        } finally {
            setIsLoading(false);
        }
    };

    const resumeExperimentHandler = async (experimentId: string, checkpointId?: string) => {
        try {
            setRecoveryStatus('loading');
            setError(null);

            const request: ResumeRequest = {
                experiment_id: experimentId,
                checkpoint_id: checkpointId,
                force: false
            };

            const response = await resumeExperiment(request);

            if (response.success) {
                setRecoveryStatus('success');
                // Refresh the list after successful resume
                await checkForResumableExperiments();
                return response;
            } else {
                throw new Error(response.message);
            }
        } catch (error) {
            console.error('Failed to resume experiment:', error);
            setError(error instanceof Error ? error.message : 'Unknown error');
            setRecoveryStatus('error');
            throw error;
        }
    };

    const createCheckpointHandler = async (experimentId: string, description?: string) => {
        try {
            setRecoveryStatus('loading');
            setError(null);

            const request: CheckpointRequest = {
                experiment_id: experimentId,
                description
            };

            const response = await createCheckpoint(request);

            if (response.success) {
                setRecoveryStatus('success');
                return response;
            } else {
                throw new Error('Failed to create checkpoint');
            }
        } catch (error) {
            console.error('Failed to create checkpoint:', error);
            setError(error instanceof Error ? error.message : 'Unknown error');
            setRecoveryStatus('error');
            throw error;
        }
    };

    const deleteCheckpointHandler = async (experimentId: string, checkpointId: string) => {
        try {
            setRecoveryStatus('loading');
            setError(null);

            const request: CheckpointDeleteRequest = {
                experiment_id: experimentId,
                checkpoint_id: checkpointId
            };

            const response = await deleteCheckpoint(request);

            if (response.success) {
                setRecoveryStatus('success');
                return response;
            } else {
                throw new Error(response.message);
            }
        } catch (error) {
            console.error('Failed to delete checkpoint:', error);
            setError(error instanceof Error ? error.message : 'Unknown error');
            setRecoveryStatus('error');
            throw error;
        }
    };

    const getExperimentStatusHandler = async (experimentId: string): Promise<ExperimentStatus | null> => {
        try {
            const status = await getExperimentStatus(experimentId);
            return status;
        } catch (error) {
            console.error('Failed to get experiment status:', error);
            return null;
        }
    };

    const getCheckpointsHandler = async (experimentId: string): Promise<CheckpointInfo[]> => {
        try {
            const response = await getCheckpoints(experimentId);
            return response.checkpoints;
        } catch (error) {
            console.error('Failed to get checkpoints:', error);
            return [];
        }
    };

    const triggerAutoRecoveryHandler = async () => {
        try {
            setRecoveryStatus('loading');
            setError(null);

            const response = await triggerAutoRecovery();

            if (response.success) {
                setRecoveryStatus('success');
                // Refresh the list after auto recovery
                await checkForResumableExperiments();
                return response;
            } else {
                throw new Error('Auto recovery failed');
            }
        } catch (error) {
            console.error('Failed to trigger auto recovery:', error);
            setError(error instanceof Error ? error.message : 'Unknown error');
            setRecoveryStatus('error');
            throw error;
        }
    };

    const deleteExperimentStateHandler = async (experimentId: string) => {
        try {
            setRecoveryStatus('loading');
            setError(null);

            const response = await deleteExperimentState(experimentId);

            if (response.success) {
                setRecoveryStatus('success');
                // Refresh the list after deletion
                await checkForResumableExperiments();
                return response;
            } else {
                throw new Error(response.message);
            }
        } catch (error) {
            console.error('Failed to delete experiment state:', error);
            setError(error instanceof Error ? error.message : 'Unknown error');
            setRecoveryStatus('error');
            throw error;
        }
    };

    const clearError = () => {
        setError(null);
        setRecoveryStatus('idle');
    };

    return {
        resumableExperiments,
        isLoading,
        recoveryStatus,
        error,
        hasResumableExperiments: Array.isArray(resumableExperiments) && resumableExperiments.length > 0,
        resumableCount: Array.isArray(resumableExperiments) ? resumableExperiments.length : 0,

        // Actions
        checkForResumableExperiments,
        resumeExperiment: resumeExperimentHandler,
        createCheckpoint: createCheckpointHandler,
        deleteCheckpoint: deleteCheckpointHandler,
        getExperimentStatus: getExperimentStatusHandler,
        getCheckpoints: getCheckpointsHandler,
        triggerAutoRecovery: triggerAutoRecoveryHandler,
        deleteExperimentState: deleteExperimentStateHandler,
        clearError
    };
}

export function useExperimentStatus(experimentId: string) {
    const [status, setStatus] = useState<ExperimentStatus | null>(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const refreshStatus = async () => {
        if (!experimentId) return;

        try {
            setIsLoading(true);
            setError(null);

            const experimentStatus = await getExperimentStatus(experimentId);
            setStatus(experimentStatus);
        } catch (error) {
            console.error('Failed to refresh experiment status:', error);
            setError(error instanceof Error ? error.message : 'Unknown error');
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        refreshStatus();
    }, [experimentId]);

    return {
        status,
        isLoading,
        error,
        refreshStatus
    };
}