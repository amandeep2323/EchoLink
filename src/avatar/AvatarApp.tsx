/**
 * AvatarApp — AvatarLink shell. Three layers:
 *   top: Thought Cloud overlay
 *   middle: transparent Three.js canvas (avatar)
 *   bottom: translucent input bar + controls
 * Frameless-window drag handled via -webkit-app-region on the grip strip.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { AnimationMixer } from 'three';
import { useAvatarScene } from './useAvatarScene';
import { loadAvatar, type LoadedAvatar } from './AvatarModel';
import { KinematicsEngine, type QueueItem } from './KinematicsEngine';
import {
  loadFingerspellingSet,
  loadClipManifest,
  loadPoseFile,
  type Pose,
  type ClipManifest,
} from './poseStore';
import { tokenize, tokenizerHealthy, TokenizerError } from './tokenizerClient';
import type { SignToken } from './types';
import { ThoughtCloud, type ThoughtBubble } from './ThoughtCloud';
import { InputBar } from './InputBar';
import { getAvatarAPI } from './modeRouter';

const drag = { WebkitAppRegion: 'drag' } as React.CSSProperties;
const noDrag = { WebkitAppRegion: 'no-drag' } as React.CSSProperties;

export function AvatarApp() {
  const { mountRef, handlesRef, ready, onTick } = useAvatarScene();
  const engineRef = useRef<KinematicsEngine | null>(null);
  const fingerspellRef = useRef<Map<string, Pose>>(new Map());
  const manifestRef = useRef<ClipManifest>({ words: {} });

  const [loadError, setLoadError] = useState<string | null>(null);
  const [status, setStatus] = useState<string>('Loading avatar…');
  const [bubbles, setBubbles] = useState<ThoughtBubble[]>([]);
  const [activeIndex, setActiveIndex] = useState(-1);
  const [cloudVisible, setCloudVisible] = useState(false);
  const [busy, setBusy] = useState(true);
  const lastTextRef = useRef<string>('');

  // ── Load pose data (parallel with scene) ──
  useEffect(() => {
    (async () => {
      fingerspellRef.current = await loadFingerspellingSet();
      manifestRef.current = await loadClipManifest();
    })();
  }, []);

  // ── Once the scene is ready, load the avatar model ──
  useEffect(() => {
    if (!ready || !handlesRef.current) return;
    let disposed = false;
    (async () => {
      try {
        const avatar: LoadedAvatar = await loadAvatar();
        if (disposed) return;
        const { scene } = handlesRef.current!;
        scene.add(avatar.root);

        const mixer = new AnimationMixer(avatar.root);
        const engine = new KinematicsEngine(
          avatar.bones,
          avatar.restPose,
          {
            onTokenStart: (i) => setActiveIndex(i),
            onComplete: () => {
              setActiveIndex(-1);
              setTimeout(() => setCloudVisible(false), 400);
            },
          }
        );
        engine.attachMixer(mixer);
        engineRef.current = engine;
        onTick((dt) => engine.update(dt));

        const healthy = await tokenizerHealthy();
        setStatus(healthy ? 'Ready' : 'Tokenizer offline — type to retry');
        setBusy(false);
      } catch (e) {
        if (!disposed) {
          setLoadError('Failed to load avatar model. Place avatar.glb in public/avatar/.');
          setStatus('Avatar load error');
          setBusy(false);
        }
      }
    })();
    return () => { disposed = true; };
  }, [ready, handlesRef, onTick]);

  // ── Resolve a token sequence into engine queue items ──
  const buildQueue = useCallback(async (tokens: SignToken[]): Promise<QueueItem[]> => {
    const items: QueueItem[] = [];
    for (const tok of tokens) {
      if (tok.type === 'word') {
        const entry = manifestRef.current.words[tok.gloss];
        if (entry && entry.kind === 'pose') {
          const pose = await loadPoseFile(entry.file);
          if (pose) items.push({ label: tok.gloss, frames: pose.frames });
        }
        // gltf-kind clips: skipped gracefully in MVP (no clip → skip)
      } else {
        for (const letter of tok.letters) {
          const pose = fingerspellRef.current.get(letter.char);
          if (pose) items.push({ label: letter.char, frames: pose.frames });
        }
      }
    }
    return items;
  }, []);

  const handleSubmit = useCallback(async (text: string) => {
    lastTextRef.current = text;
    setStatus('Translating…');
    try {
      const resp = await tokenize(text);
      const labels: ThoughtBubble[] = resp.tokens.map((t) =>
        t.type === 'word' ? { label: t.gloss } : { label: t.word }
      );
      setBubbles(labels);
      setActiveIndex(-1);
      setCloudVisible(true);
      const items = await buildQueue(resp.tokens);
      if (items.length === 0) {
        setStatus('No signable content');
        setCloudVisible(false);
        return;
      }
      engineRef.current?.play(items);
      setStatus('Signing…');
    } catch (e) {
      const msg = e instanceof TokenizerError ? e.message : 'translation failed';
      setStatus(msg);
      setCloudVisible(false);
    }
  }, [buildQueue]);

  const handleReplay = useCallback(() => {
    if (lastTextRef.current) handleSubmit(lastTextRef.current);
  }, [handleSubmit]);

  const handleStop = useCallback(() => {
    engineRef.current?.stop();
    setCloudVisible(false);
    setActiveIndex(-1);
    setStatus('Stopped');
  }, []);

  const exitToEcholink = () => getAvatarAPI()?.closeAvatar();

  return (
    <div className="relative w-screen h-screen overflow-hidden" style={{ background: 'transparent' }}>
      {/* Top grip strip (drag) + exit */}
      <div className="absolute top-0 inset-x-0 h-8 flex items-center justify-between px-2 z-20" style={drag}>
        <span className="text-[10px] text-slate-400/80 select-none">AvatarLink</span>
        <button
          onClick={exitToEcholink}
          title="Back to EchoLink"
          className="rounded-md bg-slate-800/60 hover:bg-slate-700 text-slate-200 text-xs px-2 py-0.5"
          style={noDrag}
        >
          ✕
        </button>
      </div>

      {/* Thought cloud (top layer) */}
      <div className="absolute top-8 inset-x-0 z-10 flex justify-center">
        <ThoughtCloud bubbles={bubbles} activeIndex={activeIndex} visible={cloudVisible} />
      </div>

      {/* 3D canvas (middle layer) */}
      <div ref={mountRef} className="absolute inset-0 z-0" />

      {/* Error overlay */}
      {loadError && (
        <div className="absolute inset-0 z-10 flex items-center justify-center p-4 text-center">
          <p className="text-sm text-rose-300 bg-slate-900/70 rounded-lg px-3 py-2">{loadError}</p>
        </div>
      )}

      {/* Input bar (bottom layer) */}
      <div className="absolute bottom-0 inset-x-0 z-20">
        <InputBar
          onSubmit={handleSubmit}
          onReplay={handleReplay}
          onStop={handleStop}
          disabled={busy}
          status={status}
        />
      </div>
    </div>
  );
}
