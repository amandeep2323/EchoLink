/**
 * useAvatarScene — sets up a transparent Three.js scene, camera, lights, and a
 * render loop bound to a canvas mount. Returns handles the rest of the avatar
 * pipeline (model loader, kinematics engine) plug into.
 */

import { useEffect, useRef, useState } from 'react';
import {
  Scene,
  PerspectiveCamera,
  WebGLRenderer,
  AmbientLight,
  DirectionalLight,
  Clock,
} from 'three';

export interface AvatarSceneHandles {
  scene: Scene;
  camera: PerspectiveCamera;
  renderer: WebGLRenderer;
}

export type RenderTickCallback = (deltaSeconds: number) => void;

export function useAvatarScene() {
  const mountRef = useRef<HTMLDivElement | null>(null);
  const handlesRef = useRef<AvatarSceneHandles | null>(null);
  const tickCallbacksRef = useRef<Set<RenderTickCallback>>(new Set());
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const scene = new Scene();
    const camera = new PerspectiveCamera(35, 1, 0.1, 100);
    camera.position.set(0, 1.5, 2.2);
    camera.lookAt(0, 1.4, 0);

    const renderer = new WebGLRenderer({
      alpha: true,
      premultipliedAlpha: false,
      antialias: true,
    });
    renderer.setClearColor(0x000000, 0); // transparent background
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    mount.appendChild(renderer.domElement);

    scene.add(new AmbientLight(0xffffff, 0.9));
    const key = new DirectionalLight(0xffffff, 1.1);
    key.position.set(1, 2, 2);
    scene.add(key);

    const resize = () => {
      const w = mount.clientWidth || 1;
      const h = mount.clientHeight || 1;
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(mount);

    handlesRef.current = { scene, camera, renderer };
    setReady(true);

    const clock = new Clock();
    let raf = 0;
    const loop = () => {
      const dt = clock.getDelta();
      tickCallbacksRef.current.forEach((cb) => cb(dt));
      renderer.render(scene, camera);
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      renderer.dispose();
      if (renderer.domElement.parentNode === mount) {
        mount.removeChild(renderer.domElement);
      }
      handlesRef.current = null;
      setReady(false);
    };
  }, []);

  const onTick = (cb: RenderTickCallback) => {
    tickCallbacksRef.current.add(cb);
    return () => tickCallbacksRef.current.delete(cb);
  };

  return { mountRef, handlesRef, ready, onTick };
}
