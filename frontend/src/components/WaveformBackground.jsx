import React, { useEffect, useRef } from 'react';

const WaveformBackground = () => {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let animationFrameId;

    let width = window.innerWidth;
    let height = window.innerHeight;
    canvas.width = width;
    canvas.height = height;

    let mouseX = width / 2;
    let mouseY = height / 2;

    const handleResize = () => {
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = width;
      canvas.height = height;
    };

    const handleMouseMove = (e) => {
      mouseX = e.clientX;
      mouseY = e.clientY;
    };

    window.addEventListener('resize', handleResize);
    window.addEventListener('mousemove', handleMouseMove);

    let time = 0;
    const lines = 6;

    const draw = () => {
      ctx.clearRect(0, 0, width, height);
      time += 0.015;

      const centerY = height * 0.4;

      for (let i = 0; i < lines; i++) {
        ctx.beginPath();

        // New Palette: Maroon, Gold, Beige
        const gradient = ctx.createLinearGradient(0, 0, width, 0);
        if (i % 3 === 0) {
          // Maroon
          gradient.addColorStop(0,   'rgba(166, 58, 63, 0)');
          gradient.addColorStop(0.5, 'rgba(166, 58, 63, 0.50)');
          gradient.addColorStop(1,   'rgba(166, 58, 63, 0)');
        } else if (i % 3 === 1) {
          // Soft Gold
          gradient.addColorStop(0,   'rgba(198, 167, 94, 0)');
          gradient.addColorStop(0.5, 'rgba(198, 167, 94, 0.45)');
          gradient.addColorStop(1,   'rgba(198, 167, 94, 0)');
        } else {
          // Warm Beige
          gradient.addColorStop(0,   'rgba(232, 220, 200, 0)');
          gradient.addColorStop(0.5, 'rgba(232, 220, 200, 0.25)');
          gradient.addColorStop(1,   'rgba(232, 220, 200, 0)');
        }

        ctx.strokeStyle = gradient;
        ctx.lineWidth = 1.5;

        for (let x = 0; x < width; x += 5) {
          const dx = x - mouseX;
          const dist = Math.abs(dx);

          let mouseInfluence = Math.max(0, 1 - dist / 500);
          mouseInfluence = Math.pow(mouseInfluence, 2) * 120;

          const baseAmp = 40 + (i * 15);
          const freq = 0.002 + (i * 0.0005);

          const yOffset = Math.sin(x * freq + time + i) * Math.cos(x * 0.001 - time) * (baseAmp + mouseInfluence);
          const y = centerY + yOffset;

          if (x === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      animationFrameId = requestAnimationFrame(draw);
    };

    draw();

    return () => {
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('mousemove', handleMouseMove);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return <canvas ref={canvasRef} id="bg-canvas" />;
};

export default WaveformBackground;
