'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Eye,
  EyeOff,
  Zap,
  Activity,
  AlertCircle,
  Clock,
  MessageSquare
} from 'lucide-react';
import Sidebar from '@/components/Sidebar';
import ChatInput from '@/components/ChatInput';
import TraceDrawer from '@/components/TraceDrawer';
import StreamingResponse from '@/components/StreamingResponse';
import ReasoningTrace from '@/components/ReasoningTrace';
import { ResponseWithCitations } from '@/components/CitationCard';
import WorkspaceSelector, { Workspace } from '@/components/WorkspaceSelector';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface Document {
  id: string;
  filename: string;
  title: string;
  doc_type: string;
  total_chunks: number;
  total_tokens: number;
  upload_date: string;
}

interface Citation {
  index: number;
  chunk_id: string;
  document_title: string;
  text_snippet: string;
  relevance_score: number;
}

interface TraceStep {
  timestamp: number;
  state: string;
  message: string;
  data: Record<string, unknown>;
}

interface QueryResult {
  query: string;
  response: string;
  citations: Citation[];
  confidence_score: number;
  confidence_band: string;
  correction_attempts: number;
  trace: TraceStep[];
  latency_ms: number;
}

interface HealthStatus {
  status: string;
  database: {
    status: string;
    pgvector_version: string;
    documents: number;
    chunks: number;
  };
  embedding_model: string;
  version: string;
}

type TracePhase = 'idle' | 'vectorizing' | 'searching' | 'confidence' | 'self-healing' | 'generating' | 'complete';

export default function Home() {
  // State
  const [documents, setDocuments] = useState<Document[]>([]);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [history, setHistory] = useState<QueryResult[]>([]);
  const [currentResponse, setCurrentResponse] = useState('');
  const [currentTrace, setCurrentTrace] = useState<TraceStep[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [showTrace, setShowTrace] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Reasoning trace animation state
  const [tracePhase, setTracePhase] = useState<TracePhase>('idle');
  const [confidenceLevel, setConfidenceLevel] = useState<'high' | 'medium' | 'low'>('medium');
  const [failureMode, setFailureMode] = useState<string | undefined>();

  // Workspace state
  const [currentWorkspace, setCurrentWorkspace] = useState<Workspace | null>(null);

  const chatEndRef = useRef<HTMLDivElement>(null);

  // Fetch health and documents on mount
  useEffect(() => {
    fetchHealth();
    fetchDocuments();
  }, []);

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [history, currentResponse]);

  const fetchHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      const data = await res.json();
      setHealth(data);
    } catch (err) {
      console.error('Failed to fetch health:', err);
    }
  };

  const fetchDocuments = async (workspaceId?: string | null) => {
    try {
      const url = workspaceId
        ? `${API_BASE}/documents?workspace_id=${workspaceId}`
        : `${API_BASE}/documents`;
      const res = await fetch(url);
      const data = await res.json();
      setDocuments(data.documents || []);
    } catch (err) {
      console.error('Failed to fetch documents:', err);
    }
  };

  const handleUpload = async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);

    // Add workspace_id to the URL if a workspace is selected
    const url = currentWorkspace
      ? `${API_BASE}/ingest?workspace_id=${currentWorkspace.id}`
      : `${API_BASE}/ingest`;

    const res = await fetch(url, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      throw new Error(`Upload failed: ${res.statusText}`);
    }

    const data = await res.json();
    if (!data.success) {
      throw new Error(data.error || 'Upload failed');
    }
  };

  const handleDelete = async (docId: string) => {
    try {
      const res = await fetch(`${API_BASE}/documents/${docId}`, {
        method: 'DELETE',
      });
      if (res.ok) {
        fetchDocuments();
      }
    } catch (err) {
      console.error('Failed to delete document:', err);
    }
  };

  const handleQuery = useCallback(async (query: string) => {
    setIsLoading(true);
    setCurrentResponse('');
    setCurrentTrace([]);
    setError(null);
    setFailureMode(undefined);

    // Start the cinematic trace animation
    setTracePhase('vectorizing');

    // Phase 1: Vectorizing (0.5s)
    await new Promise(r => setTimeout(r, 600));
    setTracePhase('searching');

    // Phase 2: Searching (1s)
    await new Promise(r => setTimeout(r, 1000));
    setTracePhase('confidence');

    try {
      // Use streaming endpoint
      const res = await fetch(`${API_BASE}/query/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query,
          include_trace: true,
          workspace_id: currentWorkspace?.id || null,
        }),
      });

      if (!res.ok) {
        throw new Error(`Query failed: ${res.statusText}`);
      }

      const reader = res.body?.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let fullResponse = '';
      let citations: Citation[] = [];
      let latency = 0;
      let confidence_score = 0;
      let confidence_band = 'medium';
      let correction_attempts = 0;
      let currentEventType = '';

      // Wait a bit for confidence animation
      await new Promise(r => setTimeout(r, 800));

      if (reader) {
        setIsStreaming(true);

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            if (line.startsWith('event: ')) {
              currentEventType = line.slice(7).trim();
            } else if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));

                if (currentEventType === 'confidence' || data.band) {
                  confidence_score = data.score || confidence_score;
                  confidence_band = data.band || confidence_band;
                  setConfidenceLevel(confidence_band as 'high' | 'medium' | 'low');

                  // If low confidence, show self-healing
                  if (confidence_band === 'low') {
                    setTracePhase('self-healing');
                    setFailureMode(data.failure_mode || 'RETRIEVAL ISSUE');
                    await new Promise(r => setTimeout(r, 1500));
                  }

                  setTracePhase('generating');
                  setCurrentTrace(prev => [...prev, {
                    timestamp: 0,
                    state: 'estimate_confidence',
                    message: `Confidence: ${confidence_band} (${Math.round((data.score || 0) * 100)}%)`,
                    data: data
                  }]);
                }

                if (currentEventType === 'retrieval') {
                  setCurrentTrace(prev => [...prev, {
                    timestamp: 0,
                    state: 'retrieve',
                    message: `Found ${data.results_count} results`,
                    data: data
                  }]);
                }

                if (currentEventType === 'generation' || data.type === 'chunk') {
                  if (data.type === 'chunk') {
                    fullResponse += data.chunk;
                    setCurrentResponse(fullResponse);
                  } else if (data.type === 'abstention') {
                    fullResponse = data.chunk;
                    setCurrentResponse(fullResponse);
                  }
                }

                if (currentEventType === 'validation') {
                  setCurrentTrace(prev => [...prev, {
                    timestamp: 0,
                    state: 'validate',
                    message: data.skipped
                      ? 'Validation skipped'
                      : `Validated ${data.grounded_claims}/${data.total_claims} claims`,
                    data: data
                  }]);
                }

                if (currentEventType === 'done') {
                  latency = data.latency_ms || 0;
                  citations = data.citations?.map((c: { index: number; document_title: string }, i: number) => ({
                    index: c.index,
                    chunk_id: `chunk-${i}`,
                    document_title: c.document_title,
                    text_snippet: '',
                    relevance_score: 0.9
                  })) || [];
                }

                if (currentEventType === 'error') {
                  setError(data.message);
                }
              } catch {
                // JSON parse error, skip
              }
            }
          }
        }
      }

      // Complete
      setTracePhase('complete');

      // Add to history
      setHistory(prev => [...prev, {
        query,
        response: fullResponse,
        citations,
        confidence_score,
        confidence_band,
        correction_attempts,
        trace: currentTrace,
        latency_ms: latency
      }]);

    } catch (err) {
      setError(err instanceof Error ? err.message : 'Query failed');
      setTracePhase('idle');
    } finally {
      setIsLoading(false);
      setIsStreaming(false);
      setCurrentResponse('');
      setTimeout(() => setTracePhase('idle'), 500);
    }
  }, [currentTrace]);

  const getConfidenceBadgeClass = (band: string) => {
    switch (band) {
      case 'high': return 'badge-high';
      case 'medium': return 'badge-medium';
      case 'low': return 'badge-low';
      default: return 'bg-gray-600';
    }
  };

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Cinematic Reasoning Trace Overlay */}
      <AnimatePresence>
        {tracePhase !== 'idle' && tracePhase !== 'complete' && (
          <ReasoningTrace
            phase={tracePhase}
            confidenceLevel={confidenceLevel}
            failureMode={failureMode}
          />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <Sidebar
        documents={documents}
        onUpload={handleUpload}
        onDelete={handleDelete}
        onRefresh={() => fetchDocuments(currentWorkspace?.id)}
        isQuerying={isLoading}
      />

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0 relative z-10">
        {/* Header */}
        <header className="shrink-0 px-6 py-4 border-b border-white/5 bg-[#0B0F19]/60 backdrop-blur-xl flex items-center justify-between">
          <div>
            <h1 className="text-xl font-bold gradient-text font-[family-name:var(--font-space-grotesk)]">
              SELF-HEALING RAG
            </h1>
            <p className="text-[10px] text-gray-500 mt-0.5 uppercase tracking-widest font-[family-name:var(--font-space-grotesk)]">
              Confidence-Calibrated Retrieval - Agentic Self-Correction
            </p>
          </div>

          <div className="flex items-center gap-4">
            {/* Workspace Selector */}
            <WorkspaceSelector
              currentWorkspace={currentWorkspace}
              onWorkspaceChange={(ws) => {
                setCurrentWorkspace(ws);
                fetchDocuments(ws?.id);
                // Clear chat history when switching workspaces
                setHistory([]);
              }}
              onRefresh={() => fetchDocuments(currentWorkspace?.id)}
            />

            {/* Health indicator */}
            {health && (
              <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/5 border border-white/5 backdrop-blur-sm">
                <div className={`w-1.5 h-1.5 rounded-full ${health.status === 'healthy' ? 'bg-[#10B981] shadow-[0_0_8px_rgba(16,185,129,0.8)] animate-pulse' : 'bg-red-500'}`} />
                <span className="text-[#10B981] font-mono font-semibold uppercase tracking-widest text-[9px]">
                  {health.database.documents} DOCS • {health.database.chunks} CHUNKS
                </span>
              </div>
            )}

            {/* Toggle trace */}
            <button
              onClick={() => setShowTrace(!showTrace)}
              className={`
                p-2 rounded-lg transition-all
                ${showTrace
                  ? 'bg-[rgba(176,38,255,0.15)] text-[#B026FF] shadow-[0_0_15px_rgba(176,38,255,0.3)]'
                  : 'hover:bg-[rgba(255,255,255,0.05)] text-gray-400'
                }
              `}
              title={showTrace ? 'Hide reasoning trace' : 'Show reasoning trace'}
            >
              {showTrace ? <Eye className="w-5 h-5" /> : <EyeOff className="w-5 h-5" />}
            </button>
          </div>
        </header>

        {/* Chat Area */}
        <div className="flex-1 flex overflow-hidden">
          {/* Messages */}
          <div className="flex-1 flex flex-col min-w-0">
            <div className="flex-1 overflow-y-auto px-6 py-4">
              {/* Welcome state */}
              {history.length === 0 && !isStreaming && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="h-full flex flex-col items-center justify-center text-center"
                >
                  <motion.div
                    className="p-5 rounded-2xl bg-[#0B0F19] border border-white/5 mb-6 relative"
                    animate={{
                      boxShadow: [
                        '0 0 20px rgba(0,240,255,0.15)',
                        '0 0 40px rgba(176,38,255,0.2)',
                        '0 0 20px rgba(0,240,255,0.15)',
                      ],
                    }}
                    transition={{ duration: 4, repeat: Infinity, ease: "easeInOut" }}
                  >
                    <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 to-purple-500/10 rounded-2xl pointer-events-none" />
                    <Zap className="w-10 h-10 text-[#00F0FF] relative z-10 drop-shadow-[0_0_15px_rgba(0,240,255,0.8)]" />
                  </motion.div>
                  <h2 className="text-2xl font-bold gradient-text mb-2 font-[family-name:var(--font-space-grotesk)]">
                    ASK YOUR KNOWLEDGE BASE
                  </h2>
                  <p className="text-gray-400 max-w-md text-sm">
                    Upload documents and ask questions. The engine will retrieve relevant
                    information, estimate confidence, and self-correct when needed.
                  </p>
                </motion.div>
              )}

              {/* Message history */}
              <div className="space-y-8">
                <AnimatePresence>
                  {history.map((result, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, y: 20 }}
                      animate={{ opacity: 1, y: 0 }}
                      className="space-y-4"
                    >
                      {/* User query */}
                      <div className="flex justify-end">
                        <div className="glass px-5 py-3 rounded-2xl rounded-tr-sm max-w-[70%] border border-[rgba(0,240,255,0.2)]">
                          <p className="text-white">{result.query}</p>
                        </div>
                      </div>

                      {/* AI Response */}
                      <motion.div
                        className="glass-strong p-6 rounded-2xl rounded-tl-sm"
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.2 }}
                      >
                        {/* Metrics */}
                        <div className="flex items-center gap-3 mb-4 flex-wrap">
                          <span className={`px-2.5 py-1 rounded-lg text-[10px] font-bold uppercase tracking-wider ${getConfidenceBadgeClass(result.confidence_band)}`}>
                            {result.confidence_band}
                          </span>
                          {result.correction_attempts > 0 && (
                            <span className="flex items-center gap-1 text-[10px] text-yellow-400 uppercase tracking-wider">
                              <Activity className="w-3 h-3" />
                              {result.correction_attempts} correction{result.correction_attempts > 1 ? 's' : ''}
                            </span>
                          )}
                          <span className="flex items-center gap-1 text-[10px] text-gray-500 uppercase tracking-wider">
                            <Clock className="w-3 h-3" />
                            {result.latency_ms}ms
                          </span>
                          <span className="flex items-center gap-1 text-[10px] text-gray-500 uppercase tracking-wider">
                            <MessageSquare className="w-3 h-3" />
                            {result.citations.length} citation{result.citations.length !== 1 ? 's' : ''}
                          </span>
                        </div>

                        {/* Response text with citations */}
                        <div className="prose-glass">
                          {result.citations.length > 0 ? (
                            <ResponseWithCitations text={result.response} citations={result.citations} />
                          ) : (
                            <p>{result.response}</p>
                          )}
                        </div>
                      </motion.div>
                    </motion.div>
                  ))}
                </AnimatePresence>

                {/* Streaming response */}
                {isStreaming && currentResponse && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="glass-strong p-6 rounded-2xl"
                  >
                    <StreamingResponse
                      text={currentResponse}
                      isStreaming={true}
                    />
                  </motion.div>
                )}

                {/* Error */}
                {error && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="flex items-center gap-3 p-4 rounded-xl bg-red-500/10 border border-red-500/30"
                  >
                    <AlertCircle className="w-5 h-5 text-red-400 shrink-0" />
                    <p className="text-red-300 text-sm">{error}</p>
                  </motion.div>
                )}
              </div>

              <div ref={chatEndRef} />
            </div>

            {/* Input */}
            <div className="shrink-0 p-4 border-t border-[rgba(255,255,255,0.08)]">
              <ChatInput
                onSubmit={handleQuery}
                isLoading={isLoading}
              />
            </div>
          </div>

          {/* Trace Drawer */}
          <TraceDrawer
            trace={currentTrace.length > 0 ? currentTrace : (history[history.length - 1]?.trace || [])}
            isOpen={showTrace}
            onToggle={() => setShowTrace(!showTrace)}
            isProcessing={isLoading}
            currentState={tracePhase === 'idle' ? undefined : tracePhase}
          />
        </div>
      </main>
    </div>
  );
}
