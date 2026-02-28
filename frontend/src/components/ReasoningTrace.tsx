'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

type TracePhase =
  | 'idle'
  | 'vectorizing'
  | 'searching'
  | 'confidence'
  | 'self-healing'
  | 'generating'
  | 'complete';

interface ReasoningTraceProps {
  phase: TracePhase;
  confidenceLevel?: 'high' | 'medium' | 'low';
  failureMode?: string;
  onComplete?: () => void;
}

// Particle for the vectorizing effect
function VectorizingParticles() {
  return (
    <div className="relative h-8 w-64 mx-auto">
      {/* Initial line */}
      <motion.div
        className="absolute inset-y-0 left-0 right-0 flex items-center justify-center"
        initial={{ scaleX: 1, opacity: 1 }}
        animate={{ scaleX: 0, opacity: 0 }}
        transition={{ duration: 0.5, ease: 'easeInOut' }}
      >
        <div className="h-0.5 w-full bg-gradient-to-r from-transparent via-cyan-400 to-transparent" />
      </motion.div>

      {/* Fracturing particles */}
      {[...Array(24)].map((_, i) => (
        <motion.div
          key={i}
          className="absolute w-1.5 h-1.5 rounded-full bg-cyan-400"
          style={{ left: `${(i / 24) * 100}%`, top: '50%' }}
          initial={{ y: '-50%', opacity: 0, scale: 0 }}
          animate={{
            y: ['-50%', `${(Math.random() - 0.5) * 40}px`],
            x: [`0px`, `${(Math.random() - 0.5) * 30}px`],
            opacity: [0, 1, 1, 0],
            scale: [0, 1.5, 1, 0.5],
          }}
          transition={{
            duration: 1.5,
            delay: 0.3 + i * 0.02,
            ease: 'easeOut',
          }}
        />
      ))}
    </div>
  );
}

// Orbiting search visualization
function SearchOrbits() {
  return (
    <div className="relative w-32 h-32 mx-auto">
      {/* Vector orbit */}
      <motion.div
        className="absolute inset-0"
        animate={{ rotate: 360 }}
        transition={{ duration: 3, repeat: Infinity, ease: 'linear' }}
      >
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-3 h-3 rounded-full bg-cyan-400 shadow-[0_0_20px_rgba(0,240,255,0.8)]" />
        <div
          className="absolute inset-2 rounded-full border border-cyan-400/30"
          style={{ borderStyle: 'dashed' }}
        />
      </motion.div>

      {/* Keyword orbit */}
      <motion.div
        className="absolute inset-4"
        animate={{ rotate: -360 }}
        transition={{ duration: 2.5, repeat: Infinity, ease: 'linear' }}
      >
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-2.5 h-2.5 rounded-full bg-purple-400 shadow-[0_0_15px_rgba(176,38,255,0.8)]" />
        <div
          className="absolute inset-2 rounded-full border border-purple-400/30"
          style={{ borderStyle: 'dashed' }}
        />
      </motion.div>

      {/* HyDE orbit */}
      <motion.div
        className="absolute inset-8"
        animate={{ rotate: 360 }}
        transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
      >
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-2 h-2 rounded-full bg-blue-400 shadow-[0_0_12px_rgba(59,130,246,0.8)]" />
        <div
          className="absolute inset-1 rounded-full border border-blue-400/30"
          style={{ borderStyle: 'dashed' }}
        />
      </motion.div>

      {/* Center core - converging */}
      <motion.div
        className="absolute inset-0 flex items-center justify-center"
        animate={{
          scale: [1, 1.2, 1],
        }}
        transition={{ duration: 1, repeat: Infinity }}
      >
        <div className="w-4 h-4 rounded-full bg-gradient-to-br from-cyan-400 to-purple-500 shadow-[0_0_30px_rgba(0,240,255,0.5)]" />
      </motion.div>
    </div>
  );
}

// Confidence dial
function ConfidenceDial({ level }: { level: 'high' | 'medium' | 'low' }) {
  const targetAngle = level === 'high' ? 150 : level === 'medium' ? 90 : 30;
  const color = level === 'high' ? '#00F0FF' : level === 'medium' ? '#F59E0B' : '#EF4444';

  return (
    <div className="relative w-40 h-20 mx-auto overflow-hidden">
      {/* Dial background */}
      <div className="absolute bottom-0 left-1/2 -translate-x-1/2 w-36 h-36">
        <svg viewBox="0 0 100 50" className="w-full">
          {/* Background arc */}
          <path
            d="M 10 50 A 40 40 0 0 1 90 50"
            fill="none"
            stroke="rgba(255,255,255,0.1)"
            strokeWidth="8"
            strokeLinecap="round"
          />
          {/* Tick marks */}
          {[0, 30, 60, 90, 120, 150, 180].map((angle) => (
            <line
              key={angle}
              x1={50 + 35 * Math.cos((angle - 180) * (Math.PI / 180))}
              y1={50 + 35 * Math.sin((angle - 180) * (Math.PI / 180))}
              x2={50 + 42 * Math.cos((angle - 180) * (Math.PI / 180))}
              y2={50 + 42 * Math.sin((angle - 180) * (Math.PI / 180))}
              stroke="rgba(255,255,255,0.3)"
              strokeWidth="1"
            />
          ))}
        </svg>
      </div>

      {/* Needle */}
      <motion.div
        className="absolute bottom-0 left-1/2 origin-bottom"
        style={{ height: '60px', width: '2px', marginLeft: '-1px' }}
        initial={{ rotate: -90 }}
        animate={{ rotate: targetAngle - 90 }}
        transition={{ duration: 1, ease: 'easeOut', delay: 0.5 }}
      >
        <div
          className="w-full h-full rounded-full"
          style={{
            background: `linear-gradient(to top, ${color}, transparent)`,
            boxShadow: `0 0 10px ${color}`,
          }}
        />
      </motion.div>

      {/* Center point */}
      <div
        className="absolute bottom-0 left-1/2 -translate-x-1/2 w-3 h-3 rounded-full"
        style={{ backgroundColor: color, boxShadow: `0 0 15px ${color}` }}
      />

      {/* Label */}
      <motion.div
        className="absolute -bottom-6 left-1/2 -translate-x-1/2 text-xs font-bold uppercase tracking-widest"
        style={{ color }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 1.2 }}
      >
        {level}
      </motion.div>
    </div>
  );
}

// CRT Flicker effect for self-healing
function CRTFlicker({ children }: { children: React.ReactNode }) {
  return (
    <motion.div
      animate={{
        opacity: [1, 0.8, 1, 0.9, 1],
        x: [0, -2, 0, 2, 0],
      }}
      transition={{ duration: 0.3, repeat: 2 }}
      style={{
        filter: 'url(#crt-aberration)',
      }}
    >
      <svg className="absolute w-0 h-0">
        <defs>
          <filter id="crt-aberration">
            <feOffset in="SourceGraphic" dx="2" dy="0" result="red">
              <animate
                attributeName="dx"
                values="2;-2;2"
                dur="0.1s"
                repeatCount="indefinite"
              />
            </feOffset>
            <feOffset in="SourceGraphic" dx="-2" dy="0" result="blue">
              <animate
                attributeName="dx"
                values="-2;2;-2"
                dur="0.1s"
                repeatCount="indefinite"
              />
            </feOffset>
            <feBlend in="red" in2="blue" mode="screen" />
          </filter>
        </defs>
      </svg>
      {children}
    </motion.div>
  );
}

export default function ReasoningTrace({
  phase,
  confidenceLevel = 'medium',
  failureMode,
  onComplete,
}: ReasoningTraceProps) {
  const [internalPhase, setInternalPhase] = useState<TracePhase>('idle');

  useEffect(() => {
    setInternalPhase(phase);
  }, [phase]);

  if (internalPhase === 'idle' || internalPhase === 'complete') {
    return null;
  }

  return (
    <motion.div
      initial={{ opacity: 0, y: 20, scale: 0.95 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: -20, scale: 0.95 }}
      className="fixed inset-x-0 top-1/3 z-50 flex items-center justify-center pointer-events-none"
    >
      <div className="glass-strong px-12 py-8 rounded-2xl border border-white/10 min-w-[400px]">
        <AnimatePresence mode="wait">
          {/* Phase 1: Vectorizing */}
          {internalPhase === 'vectorizing' && (
            <motion.div
              key="vectorizing"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="text-center space-y-6"
            >
              <VectorizingParticles />
              <p className="text-xs font-mono uppercase tracking-[0.3em] text-cyan-400">
                [ VECTORIZING PROMPT ]
              </p>
            </motion.div>
          )}

          {/* Phase 2: Searching */}
          {internalPhase === 'searching' && (
            <motion.div
              key="searching"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="text-center space-y-6"
            >
              <SearchOrbits />
              <p className="text-xs font-mono uppercase tracking-[0.3em] text-purple-400">
                [ RERANKING KNOWLEDGE NODES ]
              </p>
            </motion.div>
          )}

          {/* Phase 3: Confidence */}
          {internalPhase === 'confidence' && (
            <motion.div
              key="confidence"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="text-center space-y-8"
            >
              <ConfidenceDial level={confidenceLevel} />
              <motion.p
                className="text-xs font-mono uppercase tracking-[0.3em]"
                style={{
                  color:
                    confidenceLevel === 'high'
                      ? '#00F0FF'
                      : confidenceLevel === 'medium'
                      ? '#F59E0B'
                      : '#EF4444',
                }}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 1.5 }}
              >
                [ CONFIDENCE: {confidenceLevel.toUpperCase()} ]
              </motion.p>
            </motion.div>
          )}

          {/* Phase 4: Self-Healing */}
          {internalPhase === 'self-healing' && (
            <CRTFlicker>
              <motion.div
                key="self-healing"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="text-center space-y-4"
              >
                <motion.div
                  animate={{
                    boxShadow: [
                      '0 0 20px rgba(245,158,11,0.3)',
                      '0 0 40px rgba(245,158,11,0.6)',
                      '0 0 20px rgba(245,158,11,0.3)',
                    ],
                  }}
                  transition={{ duration: 0.5, repeat: Infinity }}
                  className="w-16 h-16 mx-auto rounded-full border-2 border-yellow-500 flex items-center justify-center"
                >
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
                    className="w-8 h-8 border-2 border-yellow-500 border-t-transparent rounded-full"
                  />
                </motion.div>
                <p className="text-xs font-mono uppercase tracking-[0.2em] text-yellow-500">
                  [ TRIGGERING SELF-HEALING LOOP ]
                </p>
                {failureMode && (
                  <p className="text-[10px] font-mono uppercase tracking-wider text-yellow-500/70">
                    {failureMode} DETECTED
                  </p>
                )}
              </motion.div>
            </CRTFlicker>
          )}

          {/* Phase 5: Generating */}
          {internalPhase === 'generating' && (
            <motion.div
              key="generating"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -50, scale: 0.8 }}
              className="text-center space-y-4"
            >
              <div className="flex items-center justify-center gap-1">
                {[0, 1, 2].map((i) => (
                  <motion.div
                    key={i}
                    className="w-2 h-2 rounded-full bg-green-400"
                    animate={{
                      scale: [1, 1.5, 1],
                      opacity: [0.5, 1, 0.5],
                    }}
                    transition={{
                      duration: 0.6,
                      repeat: Infinity,
                      delay: i * 0.2,
                    }}
                  />
                ))}
              </div>
              <p className="text-xs font-mono uppercase tracking-[0.3em] text-green-400">
                [ STREAMING RESPONSE ]
              </p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </motion.div>
  );
}
