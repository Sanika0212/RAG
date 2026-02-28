'use client';

import { useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { FileText, ExternalLink, Copy, Check } from 'lucide-react';

interface Citation {
  index: number;
  chunk_id: string;
  document_title: string;
  text_snippet: string;
  relevance_score: number;
}

interface CitationCardProps {
  citation: Citation;
  children: React.ReactNode;
}

export default function CitationCard({ citation, children }: CitationCardProps) {
  const [isHovered, setIsHovered] = useState(false);
  const [copied, setCopied] = useState(false);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  const handleMouseEnter = () => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    setIsHovered(true);
  };

  const handleMouseLeave = () => {
    timeoutRef.current = setTimeout(() => {
      setIsHovered(false);
    }, 200);
  };

  const handleCopy = async () => {
    await navigator.clipboard.writeText(citation.text_snippet);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getScoreColor = (score: number) => {
    if (score >= 0.8) return '#10B981';
    if (score >= 0.6) return '#F59E0B';
    return '#EF4444';
  };

  return (
    <span
      className="relative inline"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
    >
      <span className="
        inline-flex items-center justify-center
        w-5 h-5 text-[10px] font-bold
        rounded-md cursor-pointer
        bg-gradient-to-r from-[#00F0FF] to-[#B026FF]
        text-[#0B0F19]
        hover:scale-110 transition-transform
      ">
        {children}
      </span>

      <AnimatePresence>
        {isHovered && (
          <motion.div
            initial={{ opacity: 0, y: 10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: 10, scale: 0.95 }}
            transition={{ duration: 0.2 }}
            className="
              absolute left-0 top-full mt-2 z-50
              w-80 p-4
              glass-strong rounded-xl
              shadow-xl
            "
            style={{
              boxShadow: '0 20px 50px rgba(0, 0, 0, 0.5), 0 0 30px rgba(0, 240, 255, 0.1)'
            }}
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
          >
            {/* Header */}
            <div className="flex items-start justify-between mb-3">
              <div className="flex items-center gap-2">
                <div className="p-1.5 rounded-lg bg-[rgba(0,240,255,0.15)]">
                  <FileText className="w-4 h-4 text-[#00F0FF]" />
                </div>
                <div>
                  <p className="text-sm font-medium text-white truncate max-w-[180px]">
                    {citation.document_title}
                  </p>
                  <p className="text-[10px] text-gray-500">
                    Chunk ID: {citation.chunk_id.slice(0, 8)}...
                  </p>
                </div>
              </div>

              {/* Relevance score */}
              <div
                className="px-2 py-1 rounded-lg text-xs font-bold"
                style={{
                  backgroundColor: `${getScoreColor(citation.relevance_score)}20`,
                  color: getScoreColor(citation.relevance_score)
                }}
              >
                {Math.round(citation.relevance_score * 100)}%
              </div>
            </div>

            {/* Text snippet */}
            <div className="relative">
              <p className="text-sm text-gray-300 leading-relaxed line-clamp-4">
                "{citation.text_snippet}"
              </p>

              {/* Gradient fade */}
              <div className="absolute bottom-0 inset-x-0 h-6 bg-gradient-to-t from-[rgba(13,17,23,0.9)] to-transparent pointer-events-none" />
            </div>

            {/* Actions */}
            <div className="flex items-center justify-end gap-2 mt-3 pt-3 border-t border-[rgba(255,255,255,0.08)]">
              <button
                onClick={handleCopy}
                className="p-1.5 rounded-lg hover:bg-[rgba(255,255,255,0.05)] transition-colors text-gray-400 hover:text-white"
                title="Copy snippet"
              >
                {copied ? (
                  <Check className="w-4 h-4 text-[#10B981]" />
                ) : (
                  <Copy className="w-4 h-4" />
                )}
              </button>
              <button
                className="p-1.5 rounded-lg hover:bg-[rgba(255,255,255,0.05)] transition-colors text-gray-400 hover:text-white"
                title="View chunk"
              >
                <ExternalLink className="w-4 h-4" />
              </button>
            </div>

            {/* Decorative gradient border */}
            <div
              className="absolute inset-0 rounded-xl pointer-events-none"
              style={{
                background: 'linear-gradient(135deg, rgba(0,240,255,0.2), rgba(176,38,255,0.2))',
                mask: 'linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)',
                maskComposite: 'exclude',
                WebkitMaskComposite: 'xor',
                padding: '1px',
              }}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </span>
  );
}

// Component to render response text with interactive citations
interface ResponseWithCitationsProps {
  text: string;
  citations: Citation[];
}

export function ResponseWithCitations({ text, citations }: ResponseWithCitationsProps) {
  // Parse text and replace [n] with interactive citation components
  const parts = text.split(/(\[\d+\])/g);

  return (
    <div className="prose-glass">
      {parts.map((part, index) => {
        const match = part.match(/\[(\d+)\]/);
        if (match) {
          const citationIndex = parseInt(match[1], 10);
          const citation = citations.find(c => c.index === citationIndex);
          if (citation) {
            return (
              <CitationCard key={index} citation={citation}>
                {citationIndex}
              </CitationCard>
            );
          }
        }
        return <span key={index}>{part}</span>;
      })}
    </div>
  );
}
