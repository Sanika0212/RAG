'use client';

import { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Send, Loader2, Sparkles } from 'lucide-react';

interface ChatInputProps {
  onSubmit: (query: string) => void;
  isLoading: boolean;
  placeholder?: string;
}

export default function ChatInput({
  onSubmit,
  isLoading,
  placeholder = "Ask anything about your documents..."
}: ChatInputProps) {
  const [query, setQuery] = useState('');
  const [isFocused, setIsFocused] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  }, [query]);

  const handleSubmit = () => {
    if (query.trim() && !isLoading) {
      onSubmit(query.trim());
      setQuery('');
      if (textareaRef.current) {
        textareaRef.current.style.height = 'auto';
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  return (
    <motion.div
      className={`
        relative glass-strong rounded-2xl p-1
        transition-all duration-300
        ${isFocused ? 'glow-gradient' : ''}
      `}
      animate={{
        boxShadow: isFocused
          ? '0 0 30px rgba(0, 240, 255, 0.2), 0 0 60px rgba(176, 38, 255, 0.1)'
          : '0 0 0px transparent'
      }}
    >
      {/* Gradient border effect when focused */}
      {isFocused && (
        <motion.div
          className="absolute inset-0 rounded-2xl pointer-events-none"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          style={{
            background: 'linear-gradient(135deg, rgba(0,240,255,0.3), rgba(176,38,255,0.3))',
            mask: 'linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)',
            maskComposite: 'exclude',
            WebkitMaskComposite: 'xor',
            padding: '2px',
          }}
        />
      )}

      <div className="flex items-end gap-2 p-3">
        {/* Icon */}
        <div className="shrink-0 pb-1">
          <Sparkles className={`w-5 h-5 transition-colors ${isFocused ? 'text-[#00F0FF]' : 'text-gray-500'}`} />
        </div>

        {/* Textarea */}
        <textarea
          ref={textareaRef}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setIsFocused(false)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={isLoading}
          rows={1}
          className="
            flex-1 bg-transparent resize-none
            text-white placeholder-gray-500
            focus:outline-none
            min-h-[24px] max-h-[200px]
            py-1 text-sm leading-relaxed
          "
        />

        {/* Send button */}
        <motion.button
          onClick={handleSubmit}
          disabled={!query.trim() || isLoading}
          className={`
            shrink-0 p-3 rounded-xl
            transition-all duration-300
            ${query.trim() && !isLoading
              ? 'bg-gradient-to-r from-[#00F0FF] to-[#B026FF] text-[#0B0F19] cursor-pointer'
              : 'bg-[rgba(255,255,255,0.05)] text-gray-500 cursor-not-allowed'
            }
          `}
          whileHover={query.trim() && !isLoading ? { scale: 1.05 } : {}}
          whileTap={query.trim() && !isLoading ? { scale: 0.95 } : {}}
        >
          {isLoading ? (
            <Loader2 className="w-5 h-5 animate-spin" />
          ) : (
            <Send className="w-5 h-5" />
          )}
        </motion.button>
      </div>

      {/* Helper text */}
      <div className="px-4 pb-2 flex items-center justify-between text-[10px] text-gray-500">
        <span>Press Enter to send, Shift+Enter for new line</span>
        {query.length > 0 && (
          <span>{query.length} / 2000</span>
        )}
      </div>
    </motion.div>
  );
}
