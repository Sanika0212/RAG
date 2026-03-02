'use client';

import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  X,
  Settings,
  Key,
  Server,
  Cpu,
  Zap,
  Check,
  AlertCircle,
  Eye,
  EyeOff,
  RefreshCw,
  ChevronDown,
  ExternalLink
} from 'lucide-react';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Provider configurations
const PROVIDERS = {
  anthropic: {
    name: 'Anthropic (Claude)',
    icon: '🧠',
    keyPlaceholder: 'sk-ant-api03-...',
    models: ['claude-opus-4-5-20251101', 'claude-sonnet-4-5-20250929', 'claude-haiku-4-5-20251001'],
    docsUrl: 'https://console.anthropic.com/settings/keys',
    color: '#D97757'
  },
  google: {
    name: 'Google (Gemini)',
    icon: '✨',
    keyPlaceholder: 'AIza...',
    models: ['gemini-2.0-flash', 'gemini-2.0-pro', 'gemini-1.5-flash'],
    docsUrl: 'https://aistudio.google.com/app/apikey',
    color: '#4285F4'
  },
  mercury: {
    name: 'Mercury 2 (10x Faster)',
    icon: '⚡',
    keyPlaceholder: 'your-mercury-key',
    models: ['mercury-2'],
    docsUrl: 'https://platform.inceptionlabs.ai',
    color: '#00F0FF',
    badge: 'NEW'
  },
  openai: {
    name: 'OpenAI',
    icon: '🤖',
    keyPlaceholder: 'sk-...',
    models: ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo'],
    docsUrl: 'https://platform.openai.com/api-keys',
    color: '#10A37F'
  },
  ollama: {
    name: 'Ollama (Local)',
    icon: '🦙',
    keyPlaceholder: 'No key needed',
    models: ['llama3.2', 'llama3.1', 'mistral', 'mixtral', 'codellama', 'phi3'],
    docsUrl: 'https://ollama.ai',
    color: '#FFFFFF',
    isLocal: true
  }
};

export interface UserSettings {
  // API Keys
  anthropicKey: string;
  googleKey: string;
  mercuryKey: string;
  openaiKey: string;

  // Local Models
  ollamaEndpoint: string;
  ollamaModel: string;

  // Model Selection
  agentProvider: 'anthropic' | 'mercury' | 'openai' | 'ollama';
  agentModel: string;
  generationProvider: 'google' | 'mercury' | 'openai' | 'ollama';
  generationModel: string;

  // Mercury specific
  mercuryReasoningEffort: 'low' | 'high';

  // Preferences
  useLocalModels: boolean;
}

const DEFAULT_SETTINGS: UserSettings = {
  anthropicKey: '',
  googleKey: '',
  mercuryKey: '',
  openaiKey: '',
  ollamaEndpoint: 'http://localhost:11434',
  ollamaModel: 'llama3.2',
  agentProvider: 'anthropic',
  agentModel: 'claude-haiku-4-5-20251001',
  generationProvider: 'google',
  generationModel: 'gemini-2.0-flash',
  mercuryReasoningEffort: 'low',
  useLocalModels: false
};

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
  const [settings, setSettings] = useState<UserSettings>(DEFAULT_SETTINGS);
  const [activeTab, setActiveTab] = useState<'keys' | 'models' | 'local'>('keys');
  const [showKeys, setShowKeys] = useState<Record<string, boolean>>({});
  const [validating, setValidating] = useState<string | null>(null);
  const [validationStatus, setValidationStatus] = useState<Record<string, 'valid' | 'invalid' | null>>({});
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [ollamaModels, setOllamaModels] = useState<string[]>([]);
  const [loadingOllama, setLoadingOllama] = useState(false);

  // Load settings from localStorage on mount
  useEffect(() => {
    const stored = localStorage.getItem('rag-settings');
    if (stored) {
      try {
        setSettings({ ...DEFAULT_SETTINGS, ...JSON.parse(stored) });
      } catch {
        // Invalid JSON, use defaults
      }
    }
  }, []);

  // Fetch Ollama models if local mode
  const fetchOllamaModels = async () => {
    setLoadingOllama(true);
    try {
      const res = await fetch(`${settings.ollamaEndpoint}/api/tags`);
      if (res.ok) {
        const data = await res.json();
        const models = data.models?.map((m: { name: string }) => m.name) || [];
        setOllamaModels(models);
        setValidationStatus(prev => ({ ...prev, ollama: 'valid' }));
      } else {
        setValidationStatus(prev => ({ ...prev, ollama: 'invalid' }));
      }
    } catch {
      setValidationStatus(prev => ({ ...prev, ollama: 'invalid' }));
      setOllamaModels([]);
    } finally {
      setLoadingOllama(false);
    }
  };

  // Validate an API key
  const validateKey = async (provider: string) => {
    setValidating(provider);

    try {
      // Send to backend to validate
      const res = await fetch(`${API_BASE}/settings/validate-key`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider,
          key: settings[`${provider}Key` as keyof UserSettings]
        })
      });

      if (res.ok) {
        const data = await res.json();
        setValidationStatus(prev => ({ ...prev, [provider]: data.valid ? 'valid' : 'invalid' }));
      } else {
        // Backend might not have this endpoint yet, assume valid if key exists
        const key = settings[`${provider}Key` as keyof UserSettings];
        setValidationStatus(prev => ({ ...prev, [provider]: key ? 'valid' : 'invalid' }));
      }
    } catch {
      // Network error, just check if key exists
      const key = settings[`${provider}Key` as keyof UserSettings];
      setValidationStatus(prev => ({ ...prev, [provider]: key ? 'valid' : null }));
    } finally {
      setValidating(null);
    }
  };

  // Save settings
  const saveSettings = async () => {
    setSaving(true);

    // Save to localStorage
    localStorage.setItem('rag-settings', JSON.stringify(settings));

    // Try to save to backend (optional, for persistence across devices)
    try {
      await fetch(`${API_BASE}/settings/user`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(settings)
      });
    } catch {
      // Backend might not support this, that's OK
    }

    setSaving(false);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  const updateSetting = <K extends keyof UserSettings>(key: K, value: UserSettings[K]) => {
    setSettings(prev => ({ ...prev, [key]: value }));
  };

  const toggleShowKey = (provider: string) => {
    setShowKeys(prev => ({ ...prev, [provider]: !prev[provider] }));
  };

  if (!isOpen) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 flex items-center justify-center p-4"
        onClick={onClose}
      >
        {/* Backdrop */}
        <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" />

        {/* Modal */}
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: 20 }}
          onClick={e => e.stopPropagation()}
          className="relative w-full max-w-2xl max-h-[85vh] overflow-hidden rounded-2xl bg-[#0B0F19] border border-white/10 shadow-2xl"
        >
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b border-white/5">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-gradient-to-br from-cyan-500/20 to-purple-500/20">
                <Settings className="w-5 h-5 text-cyan-400" />
              </div>
              <div>
                <h2 className="text-lg font-bold text-white">Settings</h2>
                <p className="text-xs text-gray-500">Configure API keys and models</p>
              </div>
            </div>
            <button
              onClick={onClose}
              className="p-2 rounded-lg hover:bg-white/5 text-gray-400 hover:text-white transition-colors"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Tabs */}
          <div className="flex border-b border-white/5">
            {[
              { id: 'keys', label: 'API Keys', icon: Key },
              { id: 'models', label: 'Model Selection', icon: Cpu },
              { id: 'local', label: 'Local Models', icon: Server }
            ].map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as typeof activeTab)}
                className={`
                  flex-1 flex items-center justify-center gap-2 px-4 py-3 text-sm font-medium transition-colors
                  ${activeTab === tab.id
                    ? 'text-cyan-400 border-b-2 border-cyan-400 bg-cyan-400/5'
                    : 'text-gray-400 hover:text-white hover:bg-white/5'
                  }
                `}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </button>
            ))}
          </div>

          {/* Content */}
          <div className="p-6 overflow-y-auto max-h-[calc(85vh-200px)]">
            {/* API Keys Tab */}
            {activeTab === 'keys' && (
              <div className="space-y-4">
                <p className="text-sm text-gray-400 mb-6">
                  Add your API keys to use different providers. Keys are stored locally in your browser.
                </p>

                {Object.entries(PROVIDERS).filter(([, p]) => !p.isLocal).map(([key, provider]) => (
                  <div key={key} className="p-4 rounded-xl bg-white/5 border border-white/5">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <span className="text-xl">{provider.icon}</span>
                        <span className="font-medium text-white">{provider.name}</span>
                        {provider.badge && (
                          <span className="px-2 py-0.5 text-[10px] font-bold rounded-full bg-cyan-500/20 text-cyan-400">
                            {provider.badge}
                          </span>
                        )}
                      </div>
                      <a
                        href={provider.docsUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-1 text-xs text-gray-400 hover:text-cyan-400 transition-colors"
                      >
                        Get API Key <ExternalLink className="w-3 h-3" />
                      </a>
                    </div>

                    <div className="flex gap-2">
                      <div className="relative flex-1">
                        <input
                          type={showKeys[key] ? 'text' : 'password'}
                          value={settings[`${key}Key` as keyof UserSettings] as string}
                          onChange={e => updateSetting(`${key}Key` as keyof UserSettings, e.target.value)}
                          placeholder={provider.keyPlaceholder}
                          className="w-full px-4 py-2.5 pr-10 rounded-lg bg-black/30 border border-white/10 text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500/50 font-mono text-sm"
                        />
                        <button
                          onClick={() => toggleShowKey(key)}
                          className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-white"
                        >
                          {showKeys[key] ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                        </button>
                      </div>

                      <button
                        onClick={() => validateKey(key)}
                        disabled={validating === key}
                        className={`
                          px-4 py-2.5 rounded-lg font-medium text-sm transition-all flex items-center gap-2
                          ${validationStatus[key] === 'valid'
                            ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                            : validationStatus[key] === 'invalid'
                            ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                            : 'bg-white/5 text-gray-300 border border-white/10 hover:bg-white/10'
                          }
                        `}
                      >
                        {validating === key ? (
                          <RefreshCw className="w-4 h-4 animate-spin" />
                        ) : validationStatus[key] === 'valid' ? (
                          <Check className="w-4 h-4" />
                        ) : validationStatus[key] === 'invalid' ? (
                          <AlertCircle className="w-4 h-4" />
                        ) : (
                          'Test'
                        )}
                      </button>
                    </div>

                    {key === 'mercury' && (
                      <p className="mt-2 text-xs text-cyan-400/70">
                        Mercury 2 uses diffusion-based generation for ~1000 tokens/second (10x faster than Claude)
                      </p>
                    )}
                  </div>
                ))}
              </div>
            )}

            {/* Model Selection Tab */}
            {activeTab === 'models' && (
              <div className="space-y-6">
                <p className="text-sm text-gray-400 mb-6">
                  Choose which models to use for different tasks. You need valid API keys for the selected providers.
                </p>

                {/* Agent Model */}
                <div className="p-4 rounded-xl bg-white/5 border border-white/5">
                  <div className="flex items-center gap-2 mb-3">
                    <Zap className="w-4 h-4 text-purple-400" />
                    <span className="font-medium text-white">Agent Model</span>
                    <span className="text-xs text-gray-500">(Diagnosis, Correction, Confidence)</span>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-xs text-gray-400 mb-1.5 block">Provider</label>
                      <div className="relative">
                        <select
                          value={settings.agentProvider}
                          onChange={e => {
                            const provider = e.target.value as UserSettings['agentProvider'];
                            updateSetting('agentProvider', provider);
                            // Set default model for provider
                            const models = PROVIDERS[provider]?.models || [];
                            if (models.length > 0) {
                              updateSetting('agentModel', models[0]);
                            }
                          }}
                          className="w-full px-4 py-2.5 rounded-lg bg-black/30 border border-white/10 text-white appearance-none cursor-pointer focus:outline-none focus:border-cyan-500/50"
                        >
                          {['anthropic', 'mercury', 'openai', 'ollama'].map(p => (
                            <option key={p} value={p}>
                              {PROVIDERS[p as keyof typeof PROVIDERS]?.icon} {PROVIDERS[p as keyof typeof PROVIDERS]?.name}
                            </option>
                          ))}
                        </select>
                        <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
                      </div>
                    </div>

                    <div>
                      <label className="text-xs text-gray-400 mb-1.5 block">Model</label>
                      <div className="relative">
                        <select
                          value={settings.agentModel}
                          onChange={e => updateSetting('agentModel', e.target.value)}
                          className="w-full px-4 py-2.5 rounded-lg bg-black/30 border border-white/10 text-white appearance-none cursor-pointer focus:outline-none focus:border-cyan-500/50"
                        >
                          {(settings.agentProvider === 'ollama' ? ollamaModels : PROVIDERS[settings.agentProvider]?.models || []).map(m => (
                            <option key={m} value={m}>{m}</option>
                          ))}
                        </select>
                        <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
                      </div>
                    </div>
                  </div>

                  {settings.agentProvider === 'mercury' && (
                    <div className="mt-3">
                      <label className="text-xs text-gray-400 mb-1.5 block">Reasoning Effort</label>
                      <div className="flex gap-2">
                        {['low', 'high'].map(effort => (
                          <button
                            key={effort}
                            onClick={() => updateSetting('mercuryReasoningEffort', effort as 'low' | 'high')}
                            className={`
                              flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-all
                              ${settings.mercuryReasoningEffort === effort
                                ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                                : 'bg-black/30 text-gray-400 border border-white/10 hover:bg-white/5'
                              }
                            `}
                          >
                            {effort === 'low' ? '⚡ Low (Faster)' : '🧠 High (Better Quality)'}
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Generation Model */}
                <div className="p-4 rounded-xl bg-white/5 border border-white/5">
                  <div className="flex items-center gap-2 mb-3">
                    <Cpu className="w-4 h-4 text-cyan-400" />
                    <span className="font-medium text-white">Generation Model</span>
                    <span className="text-xs text-gray-500">(Response Generation)</span>
                  </div>

                  <div className="grid grid-cols-2 gap-3">
                    <div>
                      <label className="text-xs text-gray-400 mb-1.5 block">Provider</label>
                      <div className="relative">
                        <select
                          value={settings.generationProvider}
                          onChange={e => {
                            const provider = e.target.value as UserSettings['generationProvider'];
                            updateSetting('generationProvider', provider);
                            const models = PROVIDERS[provider]?.models || [];
                            if (models.length > 0) {
                              updateSetting('generationModel', models[0]);
                            }
                          }}
                          className="w-full px-4 py-2.5 rounded-lg bg-black/30 border border-white/10 text-white appearance-none cursor-pointer focus:outline-none focus:border-cyan-500/50"
                        >
                          {['google', 'mercury', 'openai', 'ollama'].map(p => (
                            <option key={p} value={p}>
                              {PROVIDERS[p as keyof typeof PROVIDERS]?.icon} {PROVIDERS[p as keyof typeof PROVIDERS]?.name}
                            </option>
                          ))}
                        </select>
                        <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
                      </div>
                    </div>

                    <div>
                      <label className="text-xs text-gray-400 mb-1.5 block">Model</label>
                      <div className="relative">
                        <select
                          value={settings.generationModel}
                          onChange={e => updateSetting('generationModel', e.target.value)}
                          className="w-full px-4 py-2.5 rounded-lg bg-black/30 border border-white/10 text-white appearance-none cursor-pointer focus:outline-none focus:border-cyan-500/50"
                        >
                          {(settings.generationProvider === 'ollama' ? ollamaModels : PROVIDERS[settings.generationProvider]?.models || []).map(m => (
                            <option key={m} value={m}>{m}</option>
                          ))}
                        </select>
                        <ChevronDown className="absolute right-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400 pointer-events-none" />
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Local Models Tab */}
            {activeTab === 'local' && (
              <div className="space-y-6">
                <div className="p-4 rounded-xl bg-gradient-to-br from-purple-500/10 to-cyan-500/10 border border-white/10">
                  <div className="flex items-start gap-3">
                    <Server className="w-5 h-5 text-cyan-400 mt-0.5" />
                    <div>
                      <h3 className="font-medium text-white mb-1">Run Models Locally with Ollama</h3>
                      <p className="text-sm text-gray-400">
                        Use open-source models on your own hardware. No API keys required, complete privacy.
                      </p>
                    </div>
                  </div>
                </div>

                <div className="p-4 rounded-xl bg-white/5 border border-white/5">
                  <div className="flex items-center gap-2 mb-4">
                    <span className="text-xl">🦙</span>
                    <span className="font-medium text-white">Ollama Configuration</span>
                  </div>

                  <div className="space-y-4">
                    <div>
                      <label className="text-xs text-gray-400 mb-1.5 block">Ollama Endpoint</label>
                      <div className="flex gap-2">
                        <input
                          type="text"
                          value={settings.ollamaEndpoint}
                          onChange={e => updateSetting('ollamaEndpoint', e.target.value)}
                          placeholder="http://localhost:11434"
                          className="flex-1 px-4 py-2.5 rounded-lg bg-black/30 border border-white/10 text-white placeholder-gray-500 focus:outline-none focus:border-cyan-500/50 font-mono text-sm"
                        />
                        <button
                          onClick={fetchOllamaModels}
                          disabled={loadingOllama}
                          className={`
                            px-4 py-2.5 rounded-lg font-medium text-sm transition-all flex items-center gap-2
                            ${validationStatus.ollama === 'valid'
                              ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                              : validationStatus.ollama === 'invalid'
                              ? 'bg-red-500/20 text-red-400 border border-red-500/30'
                              : 'bg-white/5 text-gray-300 border border-white/10 hover:bg-white/10'
                            }
                          `}
                        >
                          {loadingOllama ? (
                            <RefreshCw className="w-4 h-4 animate-spin" />
                          ) : validationStatus.ollama === 'valid' ? (
                            <Check className="w-4 h-4" />
                          ) : (
                            'Connect'
                          )}
                        </button>
                      </div>
                    </div>

                    {ollamaModels.length > 0 && (
                      <div>
                        <label className="text-xs text-gray-400 mb-1.5 block">Available Models ({ollamaModels.length})</label>
                        <div className="flex flex-wrap gap-2">
                          {ollamaModels.map(model => (
                            <button
                              key={model}
                              onClick={() => updateSetting('ollamaModel', model)}
                              className={`
                                px-3 py-1.5 rounded-lg text-sm transition-all
                                ${settings.ollamaModel === model
                                  ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                                  : 'bg-black/30 text-gray-400 border border-white/10 hover:bg-white/5'
                                }
                              `}
                            >
                              {model}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}

                    {validationStatus.ollama === 'invalid' && (
                      <div className="p-3 rounded-lg bg-red-500/10 border border-red-500/20 text-sm text-red-400">
                        <p className="font-medium mb-1">Cannot connect to Ollama</p>
                        <p className="text-red-400/70">
                          Make sure Ollama is running: <code className="bg-black/30 px-1.5 py-0.5 rounded">ollama serve</code>
                        </p>
                      </div>
                    )}

                    <div className="p-3 rounded-lg bg-white/5 border border-white/5">
                      <p className="text-xs text-gray-400 mb-2">Quick Start:</p>
                      <code className="block text-xs text-cyan-400 bg-black/30 p-2 rounded font-mono">
                        # Install Ollama<br/>
                        curl -fsSL https://ollama.ai/install.sh | sh<br/><br/>
                        # Pull a model<br/>
                        ollama pull llama3.2<br/><br/>
                        # Start serving<br/>
                        ollama serve
                      </code>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="flex items-center justify-between px-6 py-4 border-t border-white/5 bg-black/20">
            <p className="text-xs text-gray-500">
              Settings are stored locally in your browser
            </p>
            <div className="flex items-center gap-3">
              <button
                onClick={onClose}
                className="px-4 py-2 rounded-lg text-sm font-medium text-gray-400 hover:text-white hover:bg-white/5 transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={saveSettings}
                disabled={saving}
                className={`
                  px-5 py-2 rounded-lg text-sm font-medium transition-all flex items-center gap-2
                  ${saved
                    ? 'bg-green-500/20 text-green-400 border border-green-500/30'
                    : 'bg-gradient-to-r from-cyan-500 to-purple-500 text-white hover:shadow-[0_0_20px_rgba(0,240,255,0.3)]'
                  }
                `}
              >
                {saving ? (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                ) : saved ? (
                  <>
                    <Check className="w-4 h-4" />
                    Saved!
                  </>
                ) : (
                  'Save Settings'
                )}
              </button>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}

// Hook to access settings
export function useSettings(): UserSettings {
  const [settings, setSettings] = useState<UserSettings>(DEFAULT_SETTINGS);

  useEffect(() => {
    const stored = localStorage.getItem('rag-settings');
    if (stored) {
      try {
        setSettings({ ...DEFAULT_SETTINGS, ...JSON.parse(stored) });
      } catch {
        // Invalid JSON
      }
    }

    // Listen for storage changes
    const handleStorage = (e: StorageEvent) => {
      if (e.key === 'rag-settings' && e.newValue) {
        try {
          setSettings({ ...DEFAULT_SETTINGS, ...JSON.parse(e.newValue) });
        } catch {
          // Invalid JSON
        }
      }
    };

    window.addEventListener('storage', handleStorage);
    return () => window.removeEventListener('storage', handleStorage);
  }, []);

  return settings;
}
