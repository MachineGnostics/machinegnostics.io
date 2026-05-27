/* =======================================================================
   MACHINE GNOSTICS — HOLOGRAPHIC INTERACTIVE EFFECTS
   Photon trail SVG paths, card hover light tracer,
   entropy particle overlays, gravitational ripple clicks.
   ======================================================================= */
(function () {
  'use strict';

  const TEAL  = [0, 212, 170];
  const CYAN  = [0, 229, 255];
  const GREEN = [0, 230, 118];
  const rgba  = ([r,g,b], a) => `rgba(${r},${g},${b},${a.toFixed(3)})`;
  const rand  = (lo, hi) => lo + Math.random() * (hi - lo);
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  const isLight = () => {
    const s = (document.documentElement.getAttribute('data-md-color-scheme') ||
               document.body.getAttribute('data-md-color-scheme') || '');
    return s.toLowerCase() === 'default';
  };

  /* ================================================================
     1. PHOTON TRAIL SVG — curved light paths on hero
     ================================================================ */
  function initPhotonTrails(heroEl) {
    if (!heroEl || window.innerWidth < 600) return;
    if (heroEl.querySelector('.gn-photon-svg')) return;

    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('aria-hidden', 'true');
    svg.classList.add('gn-photon-svg');
    svg.style.cssText = [
      'position:absolute', 'inset:0', 'width:100%', 'height:100%',
      'pointer-events:none', 'overflow:visible', 'z-index:0',
      'border-radius:inherit',
    ].join(';');
    heroEl.appendChild(svg);

    const W = heroEl.offsetWidth  || 900;
    const H = heroEl.offsetHeight || 320;

    /* gravity well on right side */
    const wellX = W * 0.72, wellY = H * 0.30;
    const nRays  = 6;

    for (let i = 0; i < nRays; i++) {
      const startY = H * 0.12 + (i / (nRays - 1)) * H * 0.76;
      /* deflect toward the well */
      const deflect = (1 - Math.abs(startY / H - wellY / H)) * 0.65;
      const cpX = wellX * (0.5 + rand(0, 0.3));
      const cpY = wellY + (startY - wellY) * (1 - deflect) + rand(-14, 14);
      const endX = W + 25;
      const endY = startY + (startY - wellY) * deflect * 0.25 + rand(-8, 8);

      const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      path.setAttribute('d', `M -20 ${startY.toFixed(1)} Q ${cpX.toFixed(1)} ${cpY.toFixed(1)} ${endX.toFixed(1)} ${endY.toFixed(1)}`);
      path.setAttribute('fill', 'none');

      const col  = i % 3 === 0 ? CYAN : (i % 3 === 1 ? TEAL : GREEN);
      const baseA = isLight() ? rand(0.12, 0.24) : rand(0.10, 0.22);
      path.setAttribute('stroke', rgba(col, baseA));
      path.setAttribute('stroke-width', rand(0.5, 1.3).toFixed(2));
      path.setAttribute('stroke-linecap', 'round');

      /* animate with dasharray */
      const dur = rand(6, 14).toFixed(1);
      const del = rand(0, 10).toFixed(1);
      const len = 2200; /* generous estimate */
      path.style.strokeDasharray  = `${len}`;
      path.style.strokeDashoffset = `${len}`;
      path.style.setProperty('--path-len', len);
      path.style.animation = `photon-travel ${dur}s ${del}s linear infinite`;

      svg.appendChild(path);
    }
  }

  /* ================================================================
     2. CARD HOVER LIGHT TRACER — mouse-follow radial glow on cards
     ================================================================ */
  function initCardHoverTracer() {
    document.querySelectorAll('.gn-feature, .gn-card').forEach(el => {
      el.addEventListener('mousemove', e => {
        const r  = el.getBoundingClientRect();
        const x  = ((e.clientX - r.left) / r.width  * 100).toFixed(1);
        const y  = ((e.clientY - r.top)  / r.height * 100).toFixed(1);
        el.style.setProperty('--mx', `${x}%`);
        el.style.setProperty('--my', `${y}%`);
        el.classList.add('gn-hover-light');
      });
      el.addEventListener('mouseleave', () => {
        el.classList.remove('gn-hover-light');
      });
    });
  }

  /* ================================================================
     3. GRAVITATIONAL RIPPLE ON CLICK — expanding ring from cursor
     ================================================================ */
  function initGravitationalRipple() {
    document.addEventListener('click', e => {
      const existing = document.querySelectorAll('.gn-ripple');
      if (existing.length > 6) return; /* cap ripples */

      const ripple = document.createElement('div');
      ripple.className = 'gn-ripple';
      ripple.style.cssText = `
        position: fixed;
        left: ${e.clientX}px;
        top:  ${e.clientY}px;
        width: 0; height: 0;
        border: 1.5px solid ${rgba(TEAL, 0.55)};
        border-radius: 50%;
        transform: translate(-50%, -50%);
        pointer-events: none;
        z-index: 9999;
        animation: gn-ripple-out 1.1s cubic-bezier(0.22,1,0.36,1) forwards;
      `;
      document.body.appendChild(ripple);

      /* inject keyframe once */
      if (!document.getElementById('gn-ripple-kf')) {
        const s = document.createElement('style');
        s.id = 'gn-ripple-kf';
        s.textContent = `
          @keyframes gn-ripple-out {
            0%   { width: 0;    height: 0;    opacity: 0.7; }
            70%  { width: 120px; height: 120px; opacity: 0.3; }
            100% { width: 180px; height: 180px; opacity: 0; }
          }
        `;
        document.head.appendChild(s);
      }
      ripple.addEventListener('animationend', () => ripple.remove(), { once: true });
    });
  }

  /* ================================================================
     4. ENTROPY BADGE COUNTER — telemetry readout in overline
     ================================================================ */
  function initTelemetryOverline() {
    const overline = document.querySelector('.gn-overline');
    if (!overline) return;
    const base = overline.textContent.trim();
    /* append a tiny live "entropy" readout */
    const span = document.createElement('span');
    span.style.cssText = 'margin-left:1.2em; opacity:0.55; font-size:0.85em; font-variant-numeric:tabular-nums;';
    overline.appendChild(span);
    let S = 0;
    const tick = () => {
      S = (S + rand(0.001, 0.003)) % 1;
      span.textContent = `ΔS=${S.toFixed(4)}`;
      setTimeout(tick, 220 + rand(0, 120));
    };
    tick();
  }

  /* ================================================================
     5. SECTION CORNER GRAVITY RING — dynamic pulse on sections
     ================================================================ */
  function initSectionRings() {
    document.querySelectorAll('.gn-home .gn-section').forEach(el => {
      const ring = document.createElement('div');
      ring.className = 'gn-gravity-ring';
      ring.setAttribute('aria-hidden', 'true');
      el.style.position = 'relative';
      el.style.overflow  = 'hidden';
      el.appendChild(ring);
    });
  }

  /* ================================================================
     6. PILL GRID — stagger animation on scroll reveal
     ================================================================ */
  function initPillStagger() {
    const pills = document.querySelectorAll('.gn-pill-grid span, .gn-tags span');
    pills.forEach((p, i) => {
      p.style.transitionDelay = `${i * 45}ms`;
    });
  }

  /* ================================================================
     7. STATS COUNTER ANIMATION — counts up when visible
     ================================================================ */
  function initCounters() {
    /* nothing to count yet; hook for future use */
  }

  /* ================================================================
     7b. CURSOR GLOW FOLLOWER — soft radial glow tracks the mouse
     ================================================================ */
  function initCursorGlow() {
    /* don't show on touch-only devices */
    if (window.matchMedia('(hover: none)').matches) return;

    const dot = document.createElement('div');
    dot.className = 'gn-cursor-glow';
    dot.setAttribute('aria-hidden', 'true');
    document.body.appendChild(dot);

    let tx = -200, ty = -200;
    let cx = -200, cy = -200;
    let rafId = null;

    const move = e => { tx = e.clientX; ty = e.clientY; };
    const hide = () => { dot.style.opacity = '0'; };
    const show = () => { dot.style.opacity = '1'; };

    document.addEventListener('mousemove', move, { passive: true });
    document.addEventListener('mouseleave', hide);
    document.addEventListener('mouseenter', show);

    /* smooth follow */
    const step = () => {
      cx += (tx - cx) * 0.14;
      cy += (ty - cy) * 0.14;
      dot.style.left = `${cx}px`;
      dot.style.top  = `${cy}px`;
      rafId = requestAnimationFrame(step);
    };
    step();

    /* expand glow when hovering interactive elements */
    document.querySelectorAll('.gn-feature, .gn-card, .md-button, .global-floating-btn').forEach(el => {
      el.addEventListener('mouseenter', () => {
        dot.style.width  = '56px';
        dot.style.height = '56px';
      });
      el.addEventListener('mouseleave', () => {
        dot.style.width  = '28px';
        dot.style.height = '28px';
      });
    });
  }

  /* ================================================================
     8. MAIN INIT
     ================================================================ */
  function init() {
    const hero = document.querySelector('.gn-home .gn-hero');
    if (hero) initPhotonTrails(hero);

    initCardHoverTracer();
    initGravitationalRipple();
    initSectionRings();
    initPillStagger();
    initCounters();
    initCursorGlow();

    /* only show entropy counter on dark mode */
    if (!isLight()) initTelemetryOverline();
  }

  /* run after DOM ready */
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  /* MkDocs instant navigation re-init */
  document.addEventListener('DOMContentLoaded', init);

})();
