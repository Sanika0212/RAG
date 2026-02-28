'use client';

import { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';

export default function GlassCanvas({ children }: { children: React.ReactNode }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Animated mesh gradient effect
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationId: number;
    let time = 0;

    const resize = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };

    const draw = () => {
      time += 0.002;

      // Create gradient
      const gradient = ctx.createRadialGradient(
        canvas.width * (0.3 + Math.sin(time) * 0.1),
        canvas.height * (0.3 + Math.cos(time * 0.7) * 0.1),
        0,
        canvas.width * 0.5,
        canvas.height * 0.5,
        canvas.width * 0.8
      );

      // Shifting colors - deep purples and midnight blues
      gradient.addColorStop(0, `hsla(${260 + Math.sin(time) * 10}, 80%, 8%, 1)`);
      gradient.addColorStop(0.5, `hsla(${240 + Math.cos(time * 0.5) * 15}, 70%, 5%, 1)`);
      gradient.addColorStop(1, '#050510');

      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Add subtle glow orbs
      const drawOrb = (x: number, y: number, radius: number, hue: number, alpha: number) => {
        const orbGradient = ctx.createRadialGradient(x, y, 0, x, y, radius);
        orbGradient.addColorStop(0, `hsla(${hue}, 100%, 50%, ${alpha})`);
        orbGradient.addColorStop(0.5, `hsla(${hue}, 100%, 30%, ${alpha * 0.3})`);
        orbGradient.addColorStop(1, 'transparent');
        ctx.fillStyle = orbGradient;
        ctx.fillRect(x - radius, y - radius, radius * 2, radius * 2);
      };

      // Cyan orb (top right area)
      drawOrb(
        canvas.width * (0.7 + Math.sin(time * 0.3) * 0.05),
        canvas.height * (0.2 + Math.cos(time * 0.4) * 0.05),
        300,
        185,
        0.08
      );

      // Purple orb (bottom left area)
      drawOrb(
        canvas.width * (0.2 + Math.cos(time * 0.2) * 0.05),
        canvas.height * (0.7 + Math.sin(time * 0.5) * 0.05),
        350,
        280,
        0.06
      );

      animationId = requestAnimationFrame(draw);
    };

    resize();
    draw();
    window.addEventListener('resize', resize);

    return () => {
      cancelAnimationFrame(animationId);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return (
    <div className="relative min-h-screen overflow-hidden">
      {/* Animated gradient canvas */}
      <canvas
        ref={canvasRef}
        className="fixed inset-0 -z-20"
      />

      {/* Noise overlay */}
      <div
        className="fixed inset-0 -z-10 pointer-events-none opacity-[0.03]"
        style={{
          backgroundImage: `url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='noiseFilter'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23noiseFilter)'/%3E%3C/svg%3E")`,
          backgroundRepeat: 'repeat',
        }}
      />

      {/* Grid overlay for tech feel */}
      <div
        className="fixed inset-0 -z-10 pointer-events-none opacity-[0.02]"
        style={{
          backgroundImage: `
            linear-gradient(rgba(255,255,255,0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px)
          `,
          backgroundSize: '50px 50px',
        }}
      />

      {/* Content */}
      <div className="relative z-0">
        {children}
      </div>
    </div>
  );
}
