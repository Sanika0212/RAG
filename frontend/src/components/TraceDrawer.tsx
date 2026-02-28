'use client';

import { motion, AnimatePresence } from 'framer-motion';
import {
  Search,
  Gauge,
  AlertTriangle,
  Wrench,
  Sparkles,
  CheckCircle,
  XCircle,
  ChevronDown,
  ChevronUp,
  Activity,
  Zap,
  Eye
} from 'lucide-react';

interface TraceStep {
  timestamp: number;
  state: string;
  message: string;
  data: Record<string, unknown>;
}

interface TraceDrawerProps {
  trace: TraceStep[];
  isOpen: boolean;
  onToggle: () => void;
  isProcessing?: boolean;
  currentState?: string;
  correctionTriggered?: boolean;
}

const stateConfig: Record<string, {
  icon: React.ReactNode;
  color: string;
  bgColor: string;
  label: string;
}> = {
  retrieve: {
    icon: <Search className="w-4 h-4" />,
    color: '#3B82F6',
    bgColor: 'rgba(59, 130, 246, 0.15)',
    label: 'Retrieving',
  },
  estimate_confidence: {
    icon: <Gauge className="w-4 h-4" />,
    color: '#8B5CF6',
    bgColor: 'rgba(139, 92, 246, 0.15)',
    label: 'Estimating Confidence',
  },
  route: {
    icon: <Activity className="w-4 h-4" />,
    color: '#F59E0B',
    bgColor: 'rgba(245, 158, 11, 0.15)',
    label: 'Routing',
  },
  diagnose: {
    icon: <AlertTriangle className="w-4 h-4" />,
    color: '#EF4444',
    bgColor: 'rgba(239, 68, 68, 0.15)',
    label: 'Diagnosing',
  },
  correct: {
    icon: <Wrench className="w-4 h-4" />,
    color: '#EC4899',
    bgColor: 'rgba(236, 72, 153, 0.15)',
    label: 'Self-Correcting',
  },
  generate: {
    icon: <Sparkles className="w-4 h-4" />,
    color: '#10B981',
    bgColor: 'rgba(16, 185, 129, 0.15)',
    label: 'Generating',
  },
  generate_hedged: {
    icon: <Sparkles className="w-4 h-4" />,
    color: '#F59E0B',
    bgColor: 'rgba(245, 158, 11, 0.15)',
    label: 'Generating (Hedged)',
  },
  validate: {
    icon: <CheckCircle className="w-4 h-4" />,
    color: '#06B6D4',
    bgColor: 'rgba(6, 182, 212, 0.15)',
    label: 'Validating',
  },
  abstain: {
    icon: <XCircle className="w-4 h-4" />,
    color: '#6B7280',
    bgColor: 'rgba(107, 114, 128, 0.15)',
    label: 'Abstaining',
  },
};

function ConfidenceDial({ score, band }: { score: number; band: string }) {
  const getColor = () => {
    switch (band) {
      case 'high': return '#10B981';
      case 'medium': return '#F59E0B';
      case 'low': return '#EF4444';
      default: return '#6B7280';
    }
  };

  // Rotation from 0 (left) to 180 (right), based on score
  const rotation = -90 + (score * 180);

  return (
    <div className="relative w-24 h-12 overflow-hidden">
      {/* Background arc */}
      <div
        className="absolute w-24 h-24 rounded-full border-4"
        style={{
          borderColor: 'rgba(255,255,255,0.1)',
          borderTopColor: 'transparent',
          borderLeftColor: 'transparent',
          transform: 'rotate(225deg)',
          bottom: 0,
        }}
      />
      {/* Filled arc */}
      <motion.div
        className="absolute w-24 h-24 rounded-full border-4"
        style={{
          borderColor: 'transparent',
          borderBottomColor: getColor(),
          borderRightColor: getColor(),
          bottom: 0,
        }}
        initial={{ transform: 'rotate(225deg)' }}
        animate={{ transform: `rotate(${225 + (score * 180)}deg)` }}
        transition={{ duration: 1, ease: 'easeOut' }}
      />
      {/* Center label */}
      <div className="absolute inset-x-0 bottom-0 text-center">
        <span className="text-lg font-bold" style={{ color: getColor() }}>
          {Math.round(score * 100)}%
        </span>
        <span className="block text-[10px] uppercase tracking-wider text-gray-500">
          {band}
        </span>
      </div>
    </div>
  );
}

export default function TraceDrawer({
  trace,
  isOpen,
  onToggle,
  isProcessing,
  currentState,
  correctionTriggered
}: TraceDrawerProps) {
  const currentConfig = currentState ? stateConfig[currentState] : null;

  // Find confidence from trace
  const confidenceStep = trace.find(s => s.state === 'estimate_confidence');
  const confidenceData = confidenceStep?.data as {
    score?: number;
    band?: string;
    components?: Record<string, number>;
  } | undefined;

  return (
    <motion.div
      className={`
        glass-strong border-l border-[rgba(255,255,255,0.08)]
        flex flex-col h-full overflow-hidden
        ${correctionTriggered ? 'animate-correction-shake border-yellow-500/50' : ''}
      `}
      initial={{ width: 0, opacity: 0 }}
      animate={{
        width: isOpen ? 380 : 0,
        opacity: isOpen ? 1 : 0
      }}
      transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
    >
      {/* Header */}
      <div className="p-4 border-b border-[rgba(255,255,255,0.08)] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Eye className="w-5 h-5 text-[#B026FF]" />
          <span className="font-semibold text-sm">Reasoning Trace</span>
        </div>
        <button
          onClick={onToggle}
          className="p-1.5 rounded-lg hover:bg-[rgba(255,255,255,0.05)] transition-colors"
        >
          <ChevronDown className="w-4 h-4" />
        </button>
      </div>

      {/* Current State Indicator */}
      {isProcessing && currentConfig && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mx-4 mt-4 p-3 rounded-xl"
          style={{ backgroundColor: currentConfig.bgColor }}
        >
          <div className="flex items-center gap-3">
            <div
              className="p-2 rounded-lg animate-pulse"
              style={{ backgroundColor: `${currentConfig.color}33` }}
            >
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                style={{ color: currentConfig.color }}
              >
                {currentConfig.icon}
              </motion.div>
            </div>
            <div>
              <p className="text-sm font-medium" style={{ color: currentConfig.color }}>
                {currentConfig.label}...
              </p>
              <p className="text-xs text-gray-400">Processing your query</p>
            </div>
          </div>
        </motion.div>
      )}

      {/* Confidence Dial */}
      {confidenceData?.score !== undefined && (
        <div className="px-4 py-4 border-b border-[rgba(255,255,255,0.08)]">
          <p className="text-xs text-gray-500 uppercase tracking-wider mb-3">
            Confidence Score
          </p>
          <div className="flex items-center justify-center">
            <ConfidenceDial
              score={confidenceData.score}
              band={confidenceData.band || 'medium'}
            />
          </div>
          {/* Components breakdown */}
          {confidenceData.components && (
            <div className="mt-4 grid grid-cols-2 gap-2">
              {Object.entries(confidenceData.components).map(([key, value]) => (
                <div key={key} className="text-xs">
                  <span className="text-gray-500 capitalize">
                    {key.replace(/_/g, ' ')}:
                  </span>{' '}
                  <span className="text-gray-300">
                    {typeof value === 'number' ? value.toFixed(2) : value}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Timeline */}
      <div className="flex-1 overflow-y-auto p-4">
        <div className="space-y-1">
          <AnimatePresence>
            {trace.map((step, index) => {
              const config = stateConfig[step.state] || {
                icon: <Zap className="w-4 h-4" />,
                color: '#6B7280',
                bgColor: 'rgba(107, 114, 128, 0.15)',
                label: step.state,
              };

              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05 }}
                  className="relative"
                >
                  {/* Connector line */}
                  {index < trace.length - 1 && (
                    <div
                      className="absolute left-[15px] top-8 w-0.5 h-full"
                      style={{
                        background: `linear-gradient(to bottom, ${config.color}40, transparent)`
                      }}
                    />
                  )}

                  <div className="flex items-start gap-3 p-2 rounded-lg hover:bg-[rgba(255,255,255,0.03)] transition-colors">
                    {/* Icon */}
                    <div
                      className="p-1.5 rounded-lg shrink-0"
                      style={{
                        backgroundColor: config.bgColor,
                        color: config.color
                      }}
                    >
                      {config.icon}
                    </div>

                    {/* Content */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center justify-between gap-2">
                        <span
                          className="text-sm font-medium capitalize"
                          style={{ color: config.color }}
                        >
                          {step.state.replace(/_/g, ' ')}
                        </span>
                        <span className="text-[10px] text-gray-500 shrink-0">
                          {step.timestamp.toFixed(2)}s
                        </span>
                      </div>
                      <p className="text-xs text-gray-400 mt-0.5 truncate">
                        {step.message}
                      </p>

                      {/* Expandable data */}
                      {Object.keys(step.data).length > 0 && (
                        <details className="mt-2">
                          <summary className="text-[10px] text-gray-500 cursor-pointer hover:text-gray-300 transition-colors">
                            View details
                          </summary>
                          <pre className="mt-2 p-2 bg-black/30 rounded-lg text-[10px] text-gray-400 overflow-x-auto">
                            {JSON.stringify(step.data, null, 2)}
                          </pre>
                        </details>
                      )}
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </AnimatePresence>
        </div>

        {trace.length === 0 && !isProcessing && (
          <div className="flex flex-col items-center justify-center h-full text-gray-500 text-sm">
            <Activity className="w-8 h-8 mb-2 opacity-50" />
            <p>No trace data yet</p>
            <p className="text-xs mt-1">Ask a question to see the reasoning</p>
          </div>
        )}
      </div>
    </motion.div>
  );
}
