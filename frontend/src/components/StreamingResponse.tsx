'use client';

import { useEffect, useState, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import { ResponseWithCitations } from './CitationCard';

interface Citation {
  index: number;
  chunk_id: string;
  document_title: string;
  text_snippet: string;
  relevance_score: number;
}

interface StreamingResponseProps {
  text: string;
  isStreaming: boolean;
  citations?: Citation[];
}

// Individual word component with materialization effect
function MaterializingWord({ word, delay }: { word: string; delay: number }) {
  return (
    <motion.span
      initial={{ opacity: 0, y: 5, filter: 'blur(4px)' }}
      animate={{ opacity: 1, y: 0, filter: 'blur(0px)' }}
      transition={{
        duration: 0.3,
        delay,
        ease: [0.25, 0.46, 0.45, 0.94],
      }}
      className="inline"
    >
      {word}{' '}
    </motion.span>
  );
}

// Streaming text with word-by-word materialization
function MaterializingText({ text, isStreaming }: { text: string; isStreaming: boolean }) {
  const [displayedWordCount, setDisplayedWordCount] = useState(0);
  const words = useMemo(() => text.split(/(\s+)/), [text]);

  useEffect(() => {
    if (isStreaming) {
      // During streaming, show all words as they come
      setDisplayedWordCount(words.length);
    }
  }, [words.length, isStreaming]);

  // For non-streaming completed text, show all immediately
  if (!isStreaming) {
    return <span>{text}</span>;
  }

  return (
    <span className="inline">
      {words.slice(0, displayedWordCount).map((word, index) => {
        // Don't animate whitespace
        if (/^\s+$/.test(word)) {
          return <span key={index}>{word}</span>;
        }

        // Only animate the last few words for performance
        const isRecent = index > displayedWordCount - 8;

        if (isRecent) {
          return (
            <MaterializingWord
              key={`${index}-${word}`}
              word={word}
              delay={0}
            />
          );
        }

        return <span key={index}>{word} </span>;
      })}

      {/* Blinking cursor */}
      <motion.span
        className="inline-block w-[3px] h-[1.1em] ml-0.5 align-text-bottom rounded-sm"
        style={{
          background: 'linear-gradient(to bottom, #00F0FF, #B026FF)',
        }}
        animate={{
          opacity: [1, 1, 0, 0],
        }}
        transition={{
          duration: 1,
          repeat: Infinity,
          times: [0, 0.5, 0.5, 1],
        }}
      />
    </span>
  );
}

export default function StreamingResponse({
  text,
  isStreaming,
  citations = []
}: StreamingResponseProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="relative"
    >
      <div className="prose-glass leading-relaxed">
        {citations.length > 0 && !isStreaming ? (
          <ResponseWithCitations text={text} citations={citations} />
        ) : isStreaming ? (
          <MaterializingText text={text} isStreaming={isStreaming} />
        ) : (
          <ReactMarkdown
            components={{
              p: ({ children }) => <p className="mb-4 last:mb-0">{children}</p>,
              strong: ({ children }) => <strong className="text-white font-semibold">{children}</strong>,
              em: ({ children }) => <em className="text-gray-300">{children}</em>,
              code: ({ children }) => (
                <code className="px-1.5 py-0.5 rounded bg-[rgba(255,255,255,0.05)] text-[#00F0FF] text-sm font-mono">
                  {children}
                </code>
              ),
              ul: ({ children }) => <ul className="list-disc list-inside mb-4 space-y-1">{children}</ul>,
              ol: ({ children }) => <ol className="list-decimal list-inside mb-4 space-y-1">{children}</ol>,
              li: ({ children }) => <li className="text-gray-300">{children}</li>,
              h1: ({ children }) => <h1 className="text-2xl font-bold text-white mb-4 font-[var(--font-technical)]">{children}</h1>,
              h2: ({ children }) => <h2 className="text-xl font-bold text-white mb-3 font-[var(--font-technical)]">{children}</h2>,
              h3: ({ children }) => <h3 className="text-lg font-bold text-white mb-2 font-[var(--font-technical)]">{children}</h3>,
              blockquote: ({ children }) => (
                <blockquote className="border-l-2 border-[#00F0FF] pl-4 my-4 italic text-gray-400">
                  {children}
                </blockquote>
              ),
            }}
          >
            {text}
          </ReactMarkdown>
        )}
      </div>

      {/* Ambient glow effect during streaming */}
      <AnimatePresence>
        {isStreaming && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute -inset-4 -z-10 pointer-events-none"
            style={{
              background: 'radial-gradient(ellipse at bottom right, rgba(0,240,255,0.08) 0%, transparent 60%)',
            }}
          />
        )}
      </AnimatePresence>
    </motion.div>
  );
}
