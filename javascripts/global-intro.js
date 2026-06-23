/* =======================================================================
   MACHINE GNOSTICS — GLOBAL INTRO OVERLAY
   Runs on full page open/refresh across all pages.
   Skips on internal link navigation to avoid replay while browsing docs.
   ======================================================================= */
(function () {
  'use strict';

  const SKIP_KEY = 'gn-skip-intro-next-load';
  const PLAYED_KEY = 'gn-global-intro-played';
  const DURATION_SEC = 4.0;

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const rand = (lo, hi) => lo + Math.random() * (hi - lo);
  const roundedRectPath = (ctx, x, y, w, h, r) => {
    const rr = Math.max(0, Math.min(r, Math.min(w, h) * 0.5));
    ctx.beginPath();
    ctx.moveTo(x + rr, y);
    ctx.lineTo(x + w - rr, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + rr);
    ctx.lineTo(x + w, y + h - rr);
    ctx.quadraticCurveTo(x + w, y + h, x + w - rr, y + h);
    ctx.lineTo(x + rr, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - rr);
    ctx.lineTo(x, y + rr);
    ctx.quadraticCurveTo(x, y, x + rr, y);
    ctx.closePath();
  };

  const isLight = () => {
    const s = (document.documentElement.getAttribute('data-md-color-scheme') ||
      document.body.getAttribute('data-md-color-scheme') || '').toLowerCase();
    return s === 'default';
  };

  const shouldSkipIntro = () => {
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      return true;
    }

    // Play the intro once per browser session (new tab / direct URL visit).
    // Using sessionStorage means it replays in every fresh session but not
    // on in-page navigation or refresh within the same tab.
    try {
      if (sessionStorage.getItem(PLAYED_KEY) === '1') {
        return true;
      }
    } catch {
      /* no-op */
    }

    try {
      if (sessionStorage.getItem(SKIP_KEY) === '1') {
        sessionStorage.removeItem(SKIP_KEY);
        return true;
      }
    } catch {
      /* no-op */
    }
    return false;
  };

  const markInternalNavLinks = () => {
    document.addEventListener('click', (ev) => {
      const a = ev.target && ev.target.closest ? ev.target.closest('a[href]') : null;
      if (!a) return;
      if (a.target && a.target !== '_self') return;
      if (a.hasAttribute('download')) return;

      const href = a.getAttribute('href') || '';
      if (!href || href.startsWith('#') || href.startsWith('mailto:') || href.startsWith('tel:')) return;

      let url;
      try {
        url = new URL(href, location.href);
      } catch {
        return;
      }

      const sameOrigin = url.origin === location.origin;
      if (!sameOrigin) return;

      try {
        sessionStorage.setItem(SKIP_KEY, '1');
      } catch {
        /* no-op */
      }
    }, { capture: true, passive: true });
  };

  const runIntro = () => {
    // Resolve the theme bg colour immediately so the overlay covers the page
    // before the canvas paints its first frame — eliminates the 1-frame flash.
    const initialBg = isLight() ? '#f5fffb' : '#00010a';

    const overlay = document.createElement('div');
    overlay.setAttribute('aria-hidden', 'true');
    overlay.style.cssText = [
      'position:fixed',
      'inset:0',
      'z-index:99999',
      'pointer-events:none',
      'opacity:1',
      'transition:opacity 520ms ease-out',
      `background:${initialBg}`,
    ].join(';');

    const canvas = document.createElement('canvas');
    canvas.style.cssText = [
      'position:absolute',
      'inset:0',
      'width:100%',
      'height:100%',
      'display:block',
    ].join(';');
    overlay.appendChild(canvas);
    document.body.appendChild(overlay);

    document.documentElement.classList.add('gn-global-intro-active');
    document.body.classList.add('gn-global-intro-active');

    const ctx = canvas.getContext('2d');
    let W = 0;
    let H = 0;
    let t = 0;
    let lastTs = performance.now();
    let raf = null;

    const photons = [];
    const shockwaves = [];
    const coreParticles = [];
    const ignitionStars = [];
    let blastShell = null;
    let secondaryWaveTriggered = false;
    let rayPhase = rand(0, Math.PI * 2);

    const COMET_DARK = [
      [0, 212, 170], [0, 229, 255], [0, 230, 118], [100, 255, 218],
      [255, 220, 80], [255, 165, 60], [160, 200, 255], [255, 200, 140],
    ];
    const COMET_LIGHT = [
      [0, 130, 105], [0, 130, 168], [0, 148, 65], [0, 155, 145],
      [180, 120, 0], [200, 140, 20], [30, 140, 190], [120, 170, 40],
    ];

    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = window.innerWidth;
      H = window.innerHeight;
      canvas.width = Math.floor(W * dpr);
      canvas.height = Math.floor(H * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      initIgnitionStars();
    };

    const initCoreParticles = () => {
      coreParticles.length = 0;
      const palette = isLight() ? COMET_LIGHT : COMET_DARK;
      const n = 28;
      for (let i = 0; i < n; i++) {
        coreParticles.push({
          angle: rand(0, Math.PI * 2),
          speed: rand(0.018, 0.050),
          ring: rand(0.40, 0.98),
          size: rand(0.7, 1.8),
          wobble: rand(0, Math.PI * 2),
          color: palette[Math.floor(Math.random() * palette.length)],
        });
      }
    };

    const initIgnitionStars = () => {
      ignitionStars.length = 0;
      const palette = isLight()
        ? [
            [0, 118, 132],
            [0, 138, 178],
            [0, 145, 120],
            [168, 108, 16],
            [184, 128, 22],
            [120, 92, 30],
          ]
        : [
            [210, 245, 255],
            [0, 229, 255],
            [0, 212, 170],
            [255, 220, 96],
            [255, 176, 94],
            [255, 208, 168],
          ];
      const n = Math.max(140, Math.floor((W * H) / 9000));
      for (let i = 0; i < n; i++) {
        const col = palette[Math.floor(Math.random() * palette.length)];
        ignitionStars.push({
          x: rand(0, W),
          y: rand(0, H),
          r: rand(0.35, 1.4),
          delay: rand(1.18, 2.55),
          twinkle: rand(0.8, 2.6),
          phase: rand(0, Math.PI * 2),
          color: col,
        });
      }
    };

    const spawnBurst = () => {
      const cx = W * 0.5;
      const cy = H * 0.5;
      const palette = isLight() ? COMET_LIGHT : COMET_DARK;
      const n = Math.max(110, Math.floor((W * H) / 12000));
      for (let i = 0; i < n; i++) {
        const azimuth = rand(0, Math.PI * 2);
        const u = rand(-1, 1);
        const radial = Math.sqrt(Math.max(0, 1 - (u * u)));
        const speed = rand(2.0, 5.2);
        const dx = radial * Math.cos(azimuth);
        const dy = radial * Math.sin(azimuth);
        const dz = u;
        const col = palette[Math.floor(Math.random() * palette.length)];
        const lifeSpan = rand(2.8, 4.1);
        photons.push({
          x: cx,
          y: cy,
          vx: dx * speed,
          vy: dy * speed,
          z: rand(-18, 18),
          vz: dz * speed * 1.7,
          life: lifeSpan,
          maxLife: lifeSpan,
          r: rand(0.8, 2.0),
          color: col,
          trail: [],
          trailLen: Math.floor(rand(10, 20)),
        });
      }

      const base = Math.min(W, H);
      shockwaves.push(
        { r: 0, speed: base * 1.15, alpha: 0.72, z: -120, vz: 0.46, tiltX: 0.82, tiltY: 0.62, rot: rand(0, Math.PI * 2), rotSpeed: 0.010 },
        { r: 0, speed: base * 0.98, alpha: 0.58, z: -60, vz: 0.34, tiltX: 0.88, tiltY: 0.68, rot: rand(0, Math.PI * 2), rotSpeed: -0.008 },
        { r: 0, speed: base * 0.84, alpha: 0.46, z: 0, vz: 0.28, tiltX: 0.92, tiltY: 0.72, rot: rand(0, Math.PI * 2), rotSpeed: 0.007 }
      );

      blastShell = {
        r: 0,
        speed: Math.hypot(W, H) * 1.05,
        alpha: isLight() ? 0.30 : 0.72,
      };
    };

    let exploded = false;

    const drawFrame = () => {
      const light = isLight();
      const bg = light ? '#f5fffb' : '#00010a';
      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, W, H);

      const cx = W * 0.5;
      const cy = H * 0.5;
      const seedT = clamp((t - 0.20) / 0.90, 0, 1);
      const bangT = clamp((t - 1.10) / 0.70, 0, 1);

      const shakeRise = clamp((t - 1.10) / 0.10, 0, 1);
      const shakeFall = clamp((2.00 - t) / 0.55, 0, 1);
      const shake = shakeRise * shakeFall;

      const diag = Math.hypot(W, H);
      const project = (wx, wy, wz) => {
        const fov = Math.min(W, H) * 1.05;
        const cameraZ = 520;
        const denom = Math.max(130, cameraZ + wz);
        const scale = fov / denom;
        return {
          x: cx + (wx - cx) * scale,
          y: cy + (wy - cy) * scale,
          scale,
        };
      };
      const camPush = clamp((t - 1.08) / 0.65, 0, 1) * clamp((2.30 - t) / 0.90, 0, 1);
      const camZoom = 1 + camPush * 0.055;
      const camRoll = Math.sin(t * 24 + rayPhase) * camPush * 0.016;

      /* cinematic atmospheric vignette */
      const vignette = ctx.createRadialGradient(cx, cy, Math.min(W, H) * 0.20, cx, cy, diag * 0.70);
      if (light) {
        vignette.addColorStop(0, 'rgba(255,255,255,0)');
        vignette.addColorStop(1, 'rgba(0,35,30,0.15)');
      } else {
        vignette.addColorStop(0, 'rgba(0,0,0,0)');
        vignette.addColorStop(1, 'rgba(0,0,0,0.50)');
      }

      /* starfield ignition: stars appear after detonation with random delays */
      for (const s of ignitionStars) {
        const appear = clamp((t - s.delay) / 0.30, 0, 1);
        if (appear <= 0) continue;
        const tw = 0.55 + 0.45 * Math.sin(t * s.twinkle * 11 + s.phase);
        const aBase = light ? 0.22 : 0.72;
        const a = appear * aBase * tw;
        const col = s.color || (light ? [18, 118, 132] : [210, 245, 255]);

        const glowA = light ? a * 0.34 : a * 0.56;
        const glow = ctx.createRadialGradient(s.x, s.y, 0, s.x, s.y, s.r * 6.2);
        glow.addColorStop(0, `rgba(${col[0]}, ${col[1]}, ${col[2]}, ${glowA.toFixed(3)})`);
        glow.addColorStop(1, 'transparent');
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r * 6.2, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = `rgba(${col[0]}, ${col[1]}, ${col[2]}, ${a.toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fill();
      }

      ctx.save();
      if (shake > 0) {
        const amp = 2.6 * shake * shake;
        ctx.translate((Math.random() * 2 - 1) * amp, (Math.random() * 2 - 1) * amp);
      }

      /* camera push + slight roll for 3D cinematic impact */
      ctx.translate(cx, cy);
      ctx.rotate(camRoll);
      ctx.scale(camZoom, camZoom);
      ctx.translate(-cx, -cy);

      /* space distortion lens pulse near explosion */
      const lensRise = clamp((t - 1.06) / 0.14, 0, 1);
      const lensFall = clamp((1.92 - t) / 0.72, 0, 1);
      const lensT = lensRise * lensFall;
      if (lensT > 0) {
        const s = 1 + lensT * 0.018 * Math.sin(t * 44);
        ctx.translate(cx, cy);
        ctx.scale(s, s);
        ctx.translate(-cx, -cy);
      }

      if (seedT > 0) {
        const postT = clamp((t - 1.10) / 1.35, 0, 1);
        const prePulse = t < 1.10 ? (0.93 + 0.07 * Math.sin(t * 12)) : 1;
        const r = Math.min(W, H) * (0.018 + seedT * 0.07 + postT * 0.17) * prePulse;
        const fade = 1 - postT;

        const corona = ctx.createRadialGradient(cx, cy, 0, cx, cy, r * 7);
        if (light) {
          corona.addColorStop(0, `rgba(0,170,150, ${(0.24 * seedT * fade).toFixed(3)})`);
          corona.addColorStop(0.35, `rgba(0,145,190, ${(0.20 * seedT * fade).toFixed(3)})`);
        } else {
          corona.addColorStop(0, `rgba(220,255,245, ${(0.78 * seedT * fade).toFixed(3)})`);
          corona.addColorStop(0.35, `rgba(0,229,255, ${(0.58 * seedT * fade).toFixed(3)})`);
        }
        corona.addColorStop(1, 'transparent');
        ctx.fillStyle = corona;
        ctx.beginPath();
        ctx.arc(cx, cy, r * 7, 0, Math.PI * 2);
        ctx.fill();

        const core = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
        if (light) {
          core.addColorStop(0, `rgba(255,255,255, ${(0.95 * fade).toFixed(3)})`);
          core.addColorStop(0.5, `rgba(0,175,160, ${(0.78 * fade).toFixed(3)})`);
          core.addColorStop(1, `rgba(0,130,120, ${(0.62 * fade).toFixed(3)})`);
        } else {
          core.addColorStop(0, `rgba(245,255,255, ${(0.98 * fade).toFixed(3)})`);
          core.addColorStop(0.5, `rgba(0,245,210, ${(0.88 * fade).toFixed(3)})`);
          core.addColorStop(1, `rgba(0,195,165, ${(0.74 * fade).toFixed(3)})`);
        }
        ctx.fillStyle = core;
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, Math.PI * 2);
        ctx.fill();

        /* volumetric rays from core to mimic energetic plasma jets */
        const rayA = clamp((seedT * 0.26) + (bangT * 0.20), 0, light ? 0.16 : 0.26);
        if (rayA > 0.01) {
          ctx.save();
          ctx.translate(cx, cy);
          ctx.rotate(rayPhase + t * 0.35);
          for (let k = 0; k < 8; k++) {
            const a = (k / 8) * Math.PI * 2;
            ctx.save();
            ctx.rotate(a);
            const rg = ctx.createLinearGradient(0, 0, diag * 0.34, 0);
            if (light) {
              rg.addColorStop(0, `rgba(0,150,135, ${(rayA * 0.55).toFixed(3)})`);
              rg.addColorStop(0.45, `rgba(0,130,170, ${(rayA * 0.30).toFixed(3)})`);
            } else {
              rg.addColorStop(0, `rgba(0,229,255, ${(rayA * 0.68).toFixed(3)})`);
              rg.addColorStop(0.45, `rgba(0,212,170, ${(rayA * 0.36).toFixed(3)})`);
            }
            rg.addColorStop(1, 'transparent');
            ctx.fillStyle = rg;
            ctx.beginPath();
            ctx.moveTo(0, -2.8);
            ctx.lineTo(diag * 0.34, -0.9);
            ctx.lineTo(diag * 0.34, 0.9);
            ctx.lineTo(0, 2.8);
            ctx.closePath();
            ctx.fill();
            ctx.restore();
          }
          ctx.restore();
        }

        /* inner spinning particles around the main circle */
        const orbitFade = fade * clamp((1.35 - t) / 0.45, 0, 1);
        if (orbitFade > 0.02) {
          for (const cp of coreParticles) {
            const wob = 1 + 0.12 * Math.sin(t * 9 + cp.wobble);
            const or = r * cp.ring * wob;
            const a = cp.angle + t * (28 * cp.speed);
            const pz = 0.78 + 0.22 * Math.sin(t * 5 + cp.wobble);
            const px = cx + Math.cos(a) * or * pz;
            const py = cy + Math.sin(a) * or;
            const pa = orbitFade * (isLight() ? 0.55 : 0.82);

            const pg = ctx.createRadialGradient(px, py, 0, px, py, cp.size * 3.8);
            pg.addColorStop(0, `rgba(255,255,255, ${(pa * 0.95).toFixed(3)})`);
            pg.addColorStop(0.36, `rgba(${cp.color[0]},${cp.color[1]},${cp.color[2]}, ${(pa * 0.72).toFixed(3)})`);
            pg.addColorStop(1, 'transparent');
            ctx.fillStyle = pg;
            const pr = cp.size * (2.8 + pz * 1.4);
            ctx.beginPath();
            ctx.arc(px, py, pr, 0, Math.PI * 2);
            ctx.fill();

            ctx.fillStyle = `rgba(${cp.color[0]},${cp.color[1]},${cp.color[2]}, ${(pa * 0.95).toFixed(3)})`;
            ctx.beginPath();
            ctx.arc(px, py, cp.size, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }

      if (bangT > 0) {
        const diag = Math.hypot(W, H);
        for (let i = 0; i < 6; i++) {
          const ringT = clamp((bangT - i * 0.12) / (1 - i * 0.12), 0, 1);
          if (ringT <= 0) continue;
          const rr = 6 + ringT * (diag * (0.22 + i * 0.05));
          const ra = (1 - ringT) * (light ? 0.28 : 0.50 - i * 0.06);
          const split = 3.8 + i * 0.45 + bangT * 2.2;

          ctx.lineWidth = Math.max(0.4, 1.1 - i * 0.12);

          ctx.strokeStyle = light
            ? `rgba(255,255,255, ${(ra * 0.75).toFixed(3)})`
            : `rgba(255,255,255, ${(ra * 0.92).toFixed(3)})`;
          ctx.beginPath(); ctx.arc(cx, cy, rr, 0, Math.PI * 2); ctx.stroke();

          ctx.strokeStyle = light
            ? `rgba(0,145,190, ${(ra * 0.72).toFixed(3)})`
            : `rgba(0,229,255, ${(ra * 0.78).toFixed(3)})`;
          ctx.beginPath(); ctx.arc(cx, cy, rr + split, 0, Math.PI * 2); ctx.stroke();

          ctx.strokeStyle = light
            ? `rgba(0,155,130, ${(ra * 0.70).toFixed(3)})`
            : `rgba(0,212,170, ${(ra * 0.76).toFixed(3)})`;
          ctx.beginPath(); ctx.arc(cx, cy, Math.max(1, rr - split), 0, Math.PI * 2); ctx.stroke();
        }
      }

      for (const w of shockwaves) {
        const swHead = project(cx, cy, w.z || 0);
        const rr = Math.max(1, w.r * swHead.scale);
        const rx = rr * (w.tiltX || 0.9);
        const ry = rr * (w.tiltY || 0.7);
        const blurStrength = clamp(
          ((w.speed || 0) / Math.max(1, Math.min(W, H))) * 1.55,
          0,
          0.55
        ) * clamp(w.alpha * 1.2, 0, 1);

        const planes = [
          { rot: 0, sx: 1.00, sy: 1.00, alpha: 1.00 },
          { rot: 1.03, sx: 0.90, sy: 1.10, alpha: 0.56 },
          { rot: -0.88, sx: 1.10, sy: 0.92, alpha: 0.48 },
        ];

        const drawShell = (plane, offX, offY, col, lw, a0, a1) => {
          const prx = rx * plane.sx;
          const pry = ry * plane.sy;
          const rot = (w.rot || 0) + plane.rot;
          const mx = Math.cos(rot);
          const my = Math.sin(rot);
          ctx.save();
          ctx.translate(swHead.x + offX, swHead.y + offY);
          ctx.rotate(rot);
          ctx.scale(prx, pry);
          ctx.lineWidth = lw / Math.max(0.001, Math.min(prx, pry));
          ctx.strokeStyle = col;
          ctx.beginPath();
          ctx.arc(0, 0, 1, a0, a1);
          ctx.stroke();
          ctx.restore();

          /* subtle motion blur for fast shell edges */
          if (blurStrength > 0.045) {
            const blurPx = (0.8 + rr / 220) * (0.35 + blurStrength * 1.85) * plane.alpha;
            for (let b = 1; b <= 2; b++) {
              const f = b / 2;
              const shift = blurPx * f;
              const alphaMul = (0.26 / b) * blurStrength;

              ctx.save();
              ctx.globalAlpha *= alphaMul;
              ctx.translate(swHead.x + offX + (mx * shift), swHead.y + offY + (my * shift));
              ctx.rotate(rot);
              ctx.scale(prx, pry);
              ctx.lineWidth = (lw * 0.92) / Math.max(0.001, Math.min(prx, pry));
              ctx.strokeStyle = col;
              ctx.beginPath();
              ctx.arc(0, 0, 1, a0, a1);
              ctx.stroke();
              ctx.restore();

              ctx.save();
              ctx.globalAlpha *= alphaMul * 0.82;
              ctx.translate(swHead.x + offX - (mx * shift), swHead.y + offY - (my * shift));
              ctx.rotate(rot);
              ctx.scale(prx, pry);
              ctx.lineWidth = (lw * 0.88) / Math.max(0.001, Math.min(prx, pry));
              ctx.strokeStyle = col;
              ctx.beginPath();
              ctx.arc(0, 0, 1, a0, a1);
              ctx.stroke();
              ctx.restore();
            }
          }
        };

        for (const plane of planes) {
          const nearA = light
            ? w.alpha * 0.44 * plane.alpha
            : w.alpha * 0.96 * plane.alpha;
          const farA = light
            ? w.alpha * 0.18 * plane.alpha
            : w.alpha * 0.42 * plane.alpha;

          /* back side dimmer, front side brighter for depth cue */
          drawShell(
            plane,
            0,
            0,
            light
              ? `rgba(20,90,90, ${farA.toFixed(3)})`
              : `rgba(255,255,255, ${farA.toFixed(3)})`,
            1.7,
            Math.PI,
            Math.PI * 2
          );
          drawShell(
            plane,
            0,
            0,
            light
              ? `rgba(20,90,90, ${nearA.toFixed(3)})`
              : `rgba(255,255,255, ${nearA.toFixed(3)})`,
            2.1,
            0,
            Math.PI
          );

          const chromaShift = (4.0 + swHead.scale * 2.4) * plane.alpha;
          drawShell(
            plane,
            chromaShift,
            -chromaShift * 0.35,
            light
              ? `rgba(0,130,170, ${(w.alpha * 0.46 * plane.alpha).toFixed(3)})`
              : `rgba(0,229,255, ${(w.alpha * 0.74 * plane.alpha).toFixed(3)})`,
            1.2,
            0,
            Math.PI * 2
          );
          drawShell(
            plane,
            -chromaShift,
            chromaShift * 0.35,
            light
              ? `rgba(0,140,120, ${(w.alpha * 0.44 * plane.alpha).toFixed(3)})`
              : `rgba(0,212,170, ${(w.alpha * 0.70 * plane.alpha).toFixed(3)})`,
            1.15,
            0,
            Math.PI * 2
          );
        }
      }

      if (blastShell && blastShell.alpha > 0.01) {
        const rr = blastShell.r;
        const a = blastShell.alpha;

        ctx.lineWidth = 2.6;
        ctx.strokeStyle = light
          ? `rgba(255,255,255, ${(a * 0.44).toFixed(3)})`
          : `rgba(255,255,255, ${(a * 0.92).toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(cx, cy, rr, 0, Math.PI * 2);
        ctx.stroke();

        ctx.lineWidth = 1.4;
        ctx.strokeStyle = light
          ? `rgba(0,140,190, ${(a * 0.42).toFixed(3)})`
          : `rgba(0,229,255, ${(a * 0.74).toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(cx, cy, rr + 6, 0, Math.PI * 2);
        ctx.stroke();

        ctx.strokeStyle = light
          ? `rgba(0,155,135, ${(a * 0.40).toFixed(3)})`
          : `rgba(0,212,170, ${(a * 0.70).toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(cx, cy, Math.max(1, rr - 6), 0, Math.PI * 2);
        ctx.stroke();
      }

      if (lensT > 0) {
        const lensR = 20 + lensT * Math.min(W, H) * 0.42;
        const lg = ctx.createRadialGradient(cx, cy, lensR * 0.6, cx, cy, lensR * 1.18);
        if (light) {
          lg.addColorStop(0, `rgba(255,255,255, ${(0.08 * lensT).toFixed(3)})`);
          lg.addColorStop(0.35, `rgba(0,138,178, ${(0.16 * lensT).toFixed(3)})`);
          lg.addColorStop(0.65, `rgba(0,155,135, ${(0.12 * lensT).toFixed(3)})`);
        } else {
          lg.addColorStop(0, `rgba(255,255,255, ${(0.14 * lensT).toFixed(3)})`);
          lg.addColorStop(0.35, `rgba(0,229,255, ${(0.22 * lensT).toFixed(3)})`);
          lg.addColorStop(0.65, `rgba(0,212,170, ${(0.16 * lensT).toFixed(3)})`);
        }
        lg.addColorStop(1, 'transparent');
        ctx.strokeStyle = lg;
        ctx.lineWidth = 2.4;
        ctx.beginPath();
        ctx.arc(cx, cy, lensR, 0, Math.PI * 2);
        ctx.stroke();
      }

      for (const p of photons) {
        const life = clamp(p.life / Math.max(0.001, p.maxLife || 3.2), 0, 1);
        const head = project(p.x, p.y, p.z);
        const depthFade = clamp(1.10 - (p.z / 520), 0.28, 1.06);
        const sizeDecay = 0.50 + (0.50 * life);
        const pr = Math.max(0.10, p.r * head.scale * 3.0 * sizeDecay);
        if (p.trail.length > 1) {
          for (let i = 0; i < p.trail.length - 1; i++) {
            const progress = (i + 1) / p.trail.length;
            const p0 = project(p.trail[i].x, p.trail[i].y, p.trail[i].z);
            const p1 = project(p.trail[i + 1].x, p.trail[i + 1].y, p.trail[i + 1].z);
            const a = progress * life * depthFade * (light ? 0.40 : 0.70);
            ctx.strokeStyle = `rgba(${p.color[0]},${p.color[1]},${p.color[2]},${a.toFixed(3)})`;
            ctx.lineWidth = Math.max(0.2, pr * 1.4 * progress);
            ctx.beginPath();
            ctx.moveTo(p0.x, p0.y);
            ctx.lineTo(p1.x, p1.y);
            ctx.stroke();
          }
        }

        const glow = ctx.createRadialGradient(head.x, head.y, 0, head.x, head.y, pr * 5.4);
        glow.addColorStop(0, `rgba(255,255,255, ${(life * depthFade * (light ? 0.62 : 0.90)).toFixed(3)})`);
        glow.addColorStop(0.35, `rgba(${p.color[0]},${p.color[1]},${p.color[2]}, ${(life * depthFade * (light ? 0.35 : 0.62)).toFixed(3)})`);
        glow.addColorStop(1, 'transparent');
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(head.x, head.y, pr * 5.4, 0, Math.PI * 2);
        ctx.fill();
      }

      /* draw vignette last for cinematic edge falloff */
      ctx.fillStyle = vignette;
      ctx.fillRect(0, 0, W, H);

      ctx.restore();

      /* brand + status appears after blast shell fades */
      const shellGone = !blastShell || blastShell.alpha < 0.10;
      const brandT = shellGone ? clamp((t - 2.35) / 0.70, 0, 1) : 0;
      if (brandT > 0) {
        const ease = 1 - Math.pow(1 - brandT, 3);
        const panelW = Math.min(500, W * 0.72);
        const panelH = 108;
        const px = cx - panelW * 0.5;
        const py = cy - panelH * 0.5;

        ctx.save();
        ctx.globalAlpha = ease;

        const panelGrad = ctx.createLinearGradient(px, py, px, py + panelH);
        if (light) {
          panelGrad.addColorStop(0, 'rgba(255,255,255,0.76)');
          panelGrad.addColorStop(1, 'rgba(238,253,250,0.70)');
          ctx.strokeStyle = 'rgba(0,130,130,0.40)';
        } else {
          panelGrad.addColorStop(0, 'rgba(2,20,28,0.74)');
          panelGrad.addColorStop(1, 'rgba(2,14,22,0.68)');
          ctx.strokeStyle = 'rgba(0,212,170,0.40)';
        }
        ctx.fillStyle = panelGrad;
        ctx.lineWidth = 1;
        roundedRectPath(ctx, px, py, panelW, panelH, 10);
        ctx.fill();
        ctx.stroke();

        const textY = py + 36;
        ctx.fillStyle = light ? 'rgba(0,96,104,0.95)' : 'rgba(188,247,238,0.95)';
        ctx.font = '700 22px Space Grotesk, Space Mono, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('MACHINE GNOSTICS', cx, textY);

        const progress = clamp((t - 2.75) / 0.72, 0, 1);
        const barX = px + 24;
        const barY = py + 56;
        const barW = panelW - 48;
        const barH = 12;

        ctx.fillStyle = light ? 'rgba(0,120,130,0.20)' : 'rgba(0,212,170,0.22)';
        ctx.fillRect(barX, barY, barW, barH);

        const fillGrad = ctx.createLinearGradient(barX, barY, barX + barW, barY);
        if (light) {
          fillGrad.addColorStop(0, 'rgba(0,138,178,0.92)');
          fillGrad.addColorStop(1, 'rgba(0,155,130,0.92)');
        } else {
          fillGrad.addColorStop(0, 'rgba(0,229,255,0.94)');
          fillGrad.addColorStop(1, 'rgba(0,212,170,0.94)');
        }
        ctx.fillStyle = fillGrad;
        ctx.fillRect(barX, barY, barW * progress, barH);

        ctx.font = '700 11px Space Mono, monospace';
        ctx.fillStyle = light ? 'rgba(0,96,104,0.90)' : 'rgba(188,247,238,0.92)';
        const typingStart = 2.55;
        const typingRate = 42;
        const typing = 'ENCODING: LAWS OF NATURE';
        let typedLen = clamp(Math.floor((t - typingStart) * typingRate), 0, typing.length);
        if (t >= 3.35) typedLen = typing.length;
        const typedText = t < typingStart ? 'ENCODING:' : typing.slice(0, typedLen);
        const cursorOn = typedLen < typing.length && Math.floor(t * 6) % 2 === 0;
        ctx.fillText(`${typedText}${cursorOn ? '_' : ''}`, cx, barY + 28);

        ctx.restore();
      }
    };

    const step = () => {
      const now = performance.now();
      const dt = clamp((now - lastTs) / 1000, 1 / 120, 0.05);
      lastTs = now;
      t += dt;

      if (!exploded && t >= 1.10) {
        exploded = true;
        spawnBurst();
      }

      if (exploded && !secondaryWaveTriggered && t >= 1.36) {
        secondaryWaveTriggered = true;
        const base = Math.min(W, H);
        shockwaves.push(
          { r: 0, speed: base * 0.70, alpha: 0.34, z: 80, vz: 0.44, tiltX: 0.86, tiltY: 0.64, rot: rand(0, Math.PI * 2), rotSpeed: -0.006 },
          { r: 0, speed: base * 0.58, alpha: 0.27, z: 120, vz: 0.36, tiltX: 0.90, tiltY: 0.70, rot: rand(0, Math.PI * 2), rotSpeed: 0.005 }
        );
      }

      const frameScale = clamp(dt * 60, 0.45, 2.2);
      const diag = Math.hypot(W, H) * 1.2;

      for (let i = photons.length - 1; i >= 0; i--) {
        const p = photons[i];
        const decay = t < 3.35 ? 0.52 : 1.35;
        p.life -= dt * decay;
        if (p.life <= 0) {
          photons.splice(i, 1);
          continue;
        }
        p.trail.push({ x: p.x, y: p.y, z: p.z });
        if (p.trail.length > p.trailLen) p.trail.shift();
        p.x += p.vx * frameScale;
        p.y += p.vy * frameScale;
        p.z += p.vz * frameScale * 0.14;
        p.vx *= 0.996;
        p.vy *= 0.996;
        p.vz *= 0.997;

        const out = 180;
        if (p.x < -out || p.x > W + out || p.y < -out || p.y > H + out || p.z < -360 || p.z > 540) {
          p.life -= dt * 0.8;
        }
      }

      for (let i = shockwaves.length - 1; i >= 0; i--) {
        const w = shockwaves[i];
        w.r += w.speed * dt;
        w.z = (w.z || 0) + (w.vz || 0) * frameScale * 0.85;
        w.rot = (w.rot || 0) + (w.rotSpeed || 0) * frameScale;
        w.alpha = Math.max(0, w.alpha - dt * 0.42);
        if (w.r > diag || w.alpha <= 0.01 || (w.z || 0) > 620) shockwaves.splice(i, 1);
      }

      rayPhase += dt * 0.8;

      if (blastShell) {
        blastShell.r += blastShell.speed * dt;
        blastShell.alpha = Math.max(0, blastShell.alpha - dt * 0.36);
        if (blastShell.r > diag || blastShell.alpha <= 0.01) {
          blastShell = null;
        }
      }

      drawFrame();

      if (t >= DURATION_SEC) {
        overlay.style.opacity = '0';
        setTimeout(() => {
          cancelAnimationFrame(raf);
          overlay.remove();
          document.documentElement.classList.remove('gn-global-intro-active');
          document.body.classList.remove('gn-global-intro-active');
        }, 540);
        return;
      }

      raf = requestAnimationFrame(step);
    };

    resize();
    initCoreParticles();
    // Paint the first frame synchronously so there is zero gap between the
    // overlay being inserted into the DOM and the canvas covering the page.
    drawFrame();
    window.addEventListener('resize', resize, { passive: true });
    raf = requestAnimationFrame(step);
  };

  const init = () => {
    markInternalNavLinks();
    if (shouldSkipIntro()) return;

    // Mark as played for this session so consent-triggered reloads don't replay.
    try {
      sessionStorage.setItem(PLAYED_KEY, '1');
    } catch {
      /* no-op */
    }

    runIntro();
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init, { once: true });
  } else {
    init();
  }
})();
