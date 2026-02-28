'use client';

import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Plus,
  ChevronDown,
  Folder,
  Trash2,
  Edit2,
  Check,
  X,
  Sparkles,
  Database,
  BookOpen,
  Briefcase,
  Code,
  FileText,
  Globe,
  Heart,
  Lightbulb,
  MessageSquare,
  Music,
  Settings,
  Star,
  Users,
  Zap,
} from 'lucide-react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Available icons for workspaces
const ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  folder: Folder,
  database: Database,
  book: BookOpen,
  briefcase: Briefcase,
  code: Code,
  file: FileText,
  globe: Globe,
  heart: Heart,
  lightbulb: Lightbulb,
  message: MessageSquare,
  music: Music,
  settings: Settings,
  star: Star,
  users: Users,
  zap: Zap,
};

// Available colors
const COLORS = [
  '#00F0FF', // Cyan
  '#B026FF', // Purple
  '#10B981', // Green
  '#F59E0B', // Amber
  '#EF4444', // Red
  '#3B82F6', // Blue
  '#EC4899', // Pink
  '#8B5CF6', // Violet
];

export interface Workspace {
  id: string;
  name: string;
  description?: string;
  color: string;
  icon: string;
  document_count: number;
  created_at: string;
}

interface WorkspaceSelectorProps {
  currentWorkspace: Workspace | null;
  onWorkspaceChange: (workspace: Workspace | null) => void;
  onRefresh?: () => void;
}

export default function WorkspaceSelector({
  currentWorkspace,
  onWorkspaceChange,
  onRefresh,
}: WorkspaceSelectorProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [workspaces, setWorkspaces] = useState<Workspace[]>([]);
  const [isCreating, setIsCreating] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [newName, setNewName] = useState('');
  const [newColor, setNewColor] = useState(COLORS[0]);
  const [newIcon, setNewIcon] = useState('folder');
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Fetch workspaces on mount
  useEffect(() => {
    fetchWorkspaces();
  }, []);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
        setIsCreating(false);
      }
    }
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const fetchWorkspaces = async () => {
    try {
      const res = await fetch(`${API_BASE}/workspaces`);
      const data = await res.json();
      setWorkspaces(data.workspaces || []);
    } catch (err) {
      console.error('Failed to fetch workspaces:', err);
    }
  };

  const createWorkspace = async () => {
    if (!newName.trim()) return;

    setIsLoading(true);
    try {
      const res = await fetch(`${API_BASE}/workspaces`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: newName.trim(),
          color: newColor,
          icon: newIcon,
        }),
      });

      if (res.ok) {
        const workspace = await res.json();
        setWorkspaces(prev => [workspace, ...prev]);
        onWorkspaceChange(workspace);
        setNewName('');
        setIsCreating(false);
        onRefresh?.();
      }
    } catch (err) {
      console.error('Failed to create workspace:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const deleteWorkspace = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();

    try {
      const res = await fetch(`${API_BASE}/workspaces/${id}`, {
        method: 'DELETE',
      });

      if (res.ok) {
        setWorkspaces(prev => prev.filter(w => w.id !== id));
        if (currentWorkspace?.id === id) {
          onWorkspaceChange(null);
        }
        onRefresh?.();
      }
    } catch (err) {
      console.error('Failed to delete workspace:', err);
    }
  };

  const getIcon = (iconName: string) => {
    const IconComponent = ICONS[iconName] || Folder;
    return IconComponent;
  };

  const CurrentIcon = currentWorkspace ? getIcon(currentWorkspace.icon) : Sparkles;

  return (
    <div className="relative" ref={dropdownRef}>
      {/* Trigger Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-2 rounded-xl bg-white/5 border border-white/10 hover:border-white/20 transition-all group"
      >
        <div
          className="w-6 h-6 rounded-lg flex items-center justify-center"
          style={{ backgroundColor: `${currentWorkspace?.color || '#00F0FF'}20` }}
        >
          <CurrentIcon
            className="w-3.5 h-3.5"
            style={{ color: currentWorkspace?.color || '#00F0FF' }}
          />
        </div>
        <span className="text-sm font-medium max-w-[120px] truncate">
          {currentWorkspace?.name || 'All Documents'}
        </span>
        <ChevronDown
          className={`w-4 h-4 text-gray-400 transition-transform ${isOpen ? 'rotate-180' : ''}`}
        />
      </button>

      {/* Dropdown */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10, scale: 0.95 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -10, scale: 0.95 }}
            transition={{ duration: 0.15 }}
            className="absolute top-full left-0 mt-2 w-72 rounded-xl bg-[#0B0F19] border border-white/10 shadow-2xl overflow-hidden z-50"
          >
            {/* Header */}
            <div className="px-3 py-2 border-b border-white/5">
              <p className="text-[10px] text-gray-500 uppercase tracking-widest font-mono">
                Workspaces
              </p>
            </div>

            {/* All Documents Option */}
            <button
              onClick={() => {
                onWorkspaceChange(null);
                setIsOpen(false);
                onRefresh?.();
              }}
              className={`w-full flex items-center gap-3 px-3 py-2.5 hover:bg-white/5 transition-colors ${
                !currentWorkspace ? 'bg-white/5' : ''
              }`}
            >
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500/20 to-purple-500/20 flex items-center justify-center">
                <Sparkles className="w-4 h-4 text-[#00F0FF]" />
              </div>
              <div className="flex-1 text-left">
                <p className="text-sm font-medium">All Documents</p>
                <p className="text-[10px] text-gray-500">Search across everything</p>
              </div>
              {!currentWorkspace && (
                <Check className="w-4 h-4 text-[#00F0FF]" />
              )}
            </button>

            {/* Workspace List */}
            <div className="max-h-60 overflow-y-auto">
              {workspaces.map(workspace => {
                const Icon = getIcon(workspace.icon);
                return (
                  <button
                    key={workspace.id}
                    onClick={() => {
                      onWorkspaceChange(workspace);
                      setIsOpen(false);
                      onRefresh?.();
                    }}
                    className={`w-full flex items-center gap-3 px-3 py-2.5 hover:bg-white/5 transition-colors group ${
                      currentWorkspace?.id === workspace.id ? 'bg-white/5' : ''
                    }`}
                  >
                    <div
                      className="w-8 h-8 rounded-lg flex items-center justify-center"
                      style={{ backgroundColor: `${workspace.color}20` }}
                    >
                      <Icon className="w-4 h-4" style={{ color: workspace.color }} />
                    </div>
                    <div className="flex-1 text-left min-w-0">
                      <p className="text-sm font-medium truncate">{workspace.name}</p>
                      <p className="text-[10px] text-gray-500">
                        {workspace.document_count} document{workspace.document_count !== 1 ? 's' : ''}
                      </p>
                    </div>
                    {currentWorkspace?.id === workspace.id ? (
                      <Check className="w-4 h-4 text-[#00F0FF]" />
                    ) : (
                      <button
                        onClick={(e) => deleteWorkspace(workspace.id, e)}
                        className="p-1 rounded opacity-0 group-hover:opacity-100 hover:bg-red-500/20 hover:text-red-400 transition-all"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    )}
                  </button>
                );
              })}
            </div>

            {/* Create New Workspace */}
            <div className="border-t border-white/5">
              {isCreating ? (
                <div className="p-3 space-y-3">
                  <input
                    type="text"
                    value={newName}
                    onChange={(e) => setNewName(e.target.value)}
                    placeholder="Workspace name..."
                    className="w-full px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-sm focus:outline-none focus:border-[#00F0FF]/50"
                    autoFocus
                    onKeyDown={(e) => {
                      if (e.key === 'Enter') createWorkspace();
                      if (e.key === 'Escape') setIsCreating(false);
                    }}
                  />

                  {/* Color Picker */}
                  <div className="flex items-center gap-1.5">
                    {COLORS.map(color => (
                      <button
                        key={color}
                        onClick={() => setNewColor(color)}
                        className={`w-6 h-6 rounded-full transition-transform ${
                          newColor === color ? 'scale-110 ring-2 ring-white/30' : 'hover:scale-105'
                        }`}
                        style={{ backgroundColor: color }}
                      />
                    ))}
                  </div>

                  {/* Icon Picker */}
                  <div className="flex flex-wrap gap-1">
                    {Object.entries(ICONS).map(([name, Icon]) => (
                      <button
                        key={name}
                        onClick={() => setNewIcon(name)}
                        className={`p-1.5 rounded-lg transition-colors ${
                          newIcon === name ? 'bg-white/10' : 'hover:bg-white/5'
                        }`}
                      >
                        <Icon
                          className="w-4 h-4"
                          style={{ color: newIcon === name ? newColor : '#9CA3AF' }}
                        />
                      </button>
                    ))}
                  </div>

                  {/* Actions */}
                  <div className="flex items-center gap-2">
                    <button
                      onClick={createWorkspace}
                      disabled={!newName.trim() || isLoading}
                      className="flex-1 py-2 rounded-lg bg-[#00F0FF]/20 text-[#00F0FF] text-sm font-medium hover:bg-[#00F0FF]/30 transition-colors disabled:opacity-50"
                    >
                      {isLoading ? 'Creating...' : 'Create'}
                    </button>
                    <button
                      onClick={() => {
                        setIsCreating(false);
                        setNewName('');
                      }}
                      className="p-2 rounded-lg hover:bg-white/5 transition-colors"
                    >
                      <X className="w-4 h-4 text-gray-400" />
                    </button>
                  </div>
                </div>
              ) : (
                <button
                  onClick={() => setIsCreating(true)}
                  className="w-full flex items-center gap-2 px-3 py-3 text-sm text-gray-400 hover:text-white hover:bg-white/5 transition-colors"
                >
                  <Plus className="w-4 h-4" />
                  <span>New Workspace</span>
                </button>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
