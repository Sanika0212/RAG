'use client';

import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  FileText,
  Upload,
  ChevronLeft,
  ChevronRight,
  Trash2,
  Loader2,
  CheckCircle,
  AlertCircle,
  Database
} from 'lucide-react';

interface Document {
  id: string;
  filename: string;
  title: string;
  doc_type: string;
  total_chunks: number;
  total_tokens: number;
  upload_date: string;
}

interface SidebarProps {
  documents: Document[];
  onUpload: (file: File) => Promise<void>;
  onDelete?: (docId: string) => Promise<void>;
  onRefresh: () => void;
  activeDocId?: string;
  isQuerying?: boolean;
}

export default function Sidebar({
  documents,
  onUpload,
  onDelete,
  onRefresh,
  activeDocId,
  isQuerying
}: SidebarProps) {
  const [collapsed, setCollapsed] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback(async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files[0];
    if (file) {
      await handleUpload(file);
    }
  }, []);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      await handleUpload(file);
    }
  };

  const handleUpload = async (file: File) => {
    setUploading(true);
    setUploadProgress(0);

    // Simulate progress
    const interval = setInterval(() => {
      setUploadProgress(prev => Math.min(prev + 10, 90));
    }, 200);

    try {
      await onUpload(file);
      setUploadProgress(100);
      onRefresh();
    } finally {
      clearInterval(interval);
      setTimeout(() => {
        setUploading(false);
        setUploadProgress(0);
      }, 1000);
    }
  };

  const getDocIcon = (docType: string) => {
    return <FileText className="w-4 h-4" />;
  };

  return (
    <motion.aside
      initial={{ width: 280 }}
      animate={{ width: collapsed ? 60 : 320 }}
      transition={{ duration: 0.3, ease: [0.4, 0, 0.2, 1] }}
      className="h-screen bg-[#0B0F19]/80 backdrop-blur-xl border-r border-white/5 flex flex-col z-20"
    >
      {/* Header */}
      <div className="p-4 border-b border-white/5 bg-[#0B0F19]/40 flex items-center justify-between">
        <AnimatePresence mode="wait">
          {!collapsed && (
            <motion.div
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -10 }}
              className="flex items-center justify-between w-full pr-2"
            >
              <div className="flex items-center gap-2">
                <Database className="w-4 h-4 text-[#00F0FF]" />
                <span className="font-semibold text-sm tracking-wide">Knowledge Base</span>
              </div>

              {/* Live Status Tracker */}
              <div className="flex items-center gap-2">
                <div className={`w-1.5 h-1.5 rounded-full ${uploading ? 'bg-yellow-400 animate-pulse shadow-[0_0_8px_rgba(250,204,21,0.8)]' : 'bg-[#00F0FF] shadow-[0_0_8px_rgba(0,240,255,0.8)]'}`} />
                <span className={`text-[9px] uppercase tracking-widest font-mono ${uploading ? 'text-yellow-400' : 'text-[#00F0FF]'}`}>
                  {uploading ? 'VECTORIZING' : 'READY'}
                </span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <button
          onClick={() => setCollapsed(!collapsed)}
          className="p-1.5 rounded-lg hover:bg-[rgba(255,255,255,0.05)] transition-colors"
        >
          {collapsed ? (
            <ChevronRight className="w-4 h-4" />
          ) : (
            <ChevronLeft className="w-4 h-4" />
          )}
        </button>
      </div>

      {/* Upload Zone */}
      {!collapsed && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="p-4 bg-[#0B0F19]/20"
        >
          <div
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`
              drop-zone p-4 text-center cursor-pointer transition-all
              ${isDragging ? 'active border-[#00F0FF] bg-[rgba(0,240,255,0.1)]' : ''}
              ${uploading ? 'pointer-events-none' : ''}
            `}
          >
            <input
              type="file"
              onChange={handleFileSelect}
              accept=".pdf,.docx,.doc,.md,.txt"
              className="hidden"
              id="file-upload"
              disabled={uploading}
            />
            <label htmlFor="file-upload" className="cursor-pointer">
              {uploading ? (
                <div className="space-y-2">
                  <Loader2 className="w-6 h-6 mx-auto animate-spin text-[#00F0FF]" />
                  <div className="h-1 bg-[rgba(255,255,255,0.1)] rounded-full overflow-hidden">
                    <motion.div
                      className="h-full bg-gradient-to-r from-[#00F0FF] to-[#B026FF]"
                      initial={{ width: 0 }}
                      animate={{ width: `${uploadProgress}%` }}
                      transition={{ duration: 0.3 }}
                    />
                  </div>
                  <p className="text-xs text-gray-400">Uploading...</p>
                </div>
              ) : (
                <>
                  <Upload className="w-6 h-6 mx-auto mb-2 text-gray-400" />
                  <p className="text-xs text-gray-400">
                    Drop files or click to upload
                  </p>
                  <p className="text-xs text-gray-500 mt-1">
                    PDF, DOCX, MD, TXT
                  </p>
                </>
              )}
            </label>
          </div>
        </motion.div>
      )}

      {/* Document List */}
      <div className="flex-1 overflow-y-auto px-2 pb-4">
        <AnimatePresence>
          {documents.map((doc, index) => (
            <motion.div
              key={doc.id}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0, x: -20 }}
              transition={{ delay: index * 0.05 }}
              className={`
                group relative p-3 rounded-xl mb-2 cursor-pointer transition-all
                ${activeDocId === doc.id && isQuerying
                  ? 'glass-strong animate-pulse-glow'
                  : 'hover:bg-[rgba(255,255,255,0.05)]'
                }
              `}
            >
              <div className="flex items-start gap-3">
                <div className={`
                  p-2 rounded-lg
                  ${activeDocId === doc.id && isQuerying
                    ? 'bg-[rgba(0,240,255,0.2)] text-[#00F0FF]'
                    : 'bg-[rgba(255,255,255,0.05)] text-gray-400'
                  }
                `}>
                  {getDocIcon(doc.doc_type)}
                </div>

                {!collapsed && (
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">
                      {doc.filename}
                    </p>
                    <p className="text-xs text-gray-500 mt-0.5">
                      {doc.total_chunks} chunks
                    </p>
                  </div>
                )}
              </div>

              {/* Delete button */}
              {!collapsed && onDelete && (
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    onDelete(doc.id);
                  }}
                  className="absolute right-2 top-1/2 -translate-y-1/2 p-1.5 rounded-lg
                    opacity-0 group-hover:opacity-100 hover:bg-red-500/20 hover:text-red-400
                    transition-all"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              )}
            </motion.div>
          ))}
        </AnimatePresence>

        {documents.length === 0 && !collapsed && (
          <div className="text-center py-8 text-gray-500 text-sm">
            No documents yet.
            <br />
            Upload one to get started.
          </div>
        )}
      </div>

      {/* Footer */}
      {!collapsed && (
        <div className="p-4 border-t border-white/5 bg-[#0B0F19]/40 flex items-center gap-2">
          <div className="w-5 h-5 rounded-full bg-white/10 flex items-center justify-center text-[10px] font-bold">
            N
          </div>
          <div className="text-[10px] text-gray-500 uppercase tracking-widest font-mono">
            {documents.length} document{documents.length !== 1 ? 's' : ''} indexed
          </div>
        </div>
      )}
    </motion.aside>
  );
}
