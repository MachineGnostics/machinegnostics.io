/**
 * MACHINE GNOSTICS — UNIVERSE PHYSICS CANVAS
 * Simulates: spacetime curvature, relativistic particles, thermodynamic
 * entropy fields, gravitational waves, photon trails, and curved manifolds.
 * Inspired by: Riemannian geometry, relativistic mechanics, thermodynamics.
 */

(function () {
  'use strict';

  // ─── CONFIG ──────────────────────────────────────────────────────────────────
  const CFG = {
    // Particle system
    PARTICLE_COUNT: 90,
    PARTICLE_MIN_RADIUS: 0.8,
    PARTICLE_MAX_RADIUS: 2.2,
    PARTICLE_MIN_SPEED: 0.08,
    PARTICLE_MAX_SPEED: 0.55,
    PARTICLE_TRAIL_LENGTH: 14,
    RELATIVISTIC_THRESHOLD: 0.38,

    // Connection field (entropy coupling)
    CONNECTION_DISTANCE: 130,
    CONNECTION_MAX_OPACITY: 0.22,

    // Gravity wells
    GRAVITY_WELL_COUNT: 3,
    GRAVITY_WELL_STRENGTH: 60,
    GRAVITY_WELL_RADIUS: 180,

    // Spacetime grid
    GRID_COLS: 22,
    GRID_ROWS: 14,
    GRID_WARP_STRENGTH: 28,

    // Gravitational waves (circular ripples)
    WAVE_INTERVAL_MIN: 3500,
    WAVE_INTERVAL_MAX: 6500,
    WAVE_SPEED: 1.8,
    WAVE_MAX_RADIUS: 700,

    // Curved manifolds (Riemannian background)
    MANIFOLD_COUNT: 3,
    MANIFOLD_OPACITY: 0.035,
    MANIFOLD_ROTATE_SPEED: 0.00012,

    // Colors — exact Machine Gnostics brand palette
    COLOR_PARTICLE_PRIMARY: '45, 212, 191',
    COLOR_PARTICLE_SECONDARY: '56, 189, 248',
    COLOR_PARTICLE_ENERGY: '245, 158, 11',
    COLOR_CONNECTION_NEAR: '124, 58, 237',
    COLOR_CONNECTION_FAR: '245, 158, 11',
    COLOR_WAVE: '45, 212, 191',
    COLOR_GRID: '48, 54, 61',
    COLOR_MANIFOLD: '124, 58, 237',
  };

  // ─── CANVAS SETUP ────────────────────────────────────────────────────────────
  const canvas = document.getElementById('mg-universe');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  let W, H;

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }
  resize();
  window.addEventListener('resize', () => { resize(); initGravityWells(); });

  // ─── GRAVITY WELLS ───────────────────────────────────────────────────────────
  let gravityWells = [];
  function initGravityWells() {
    gravityWells = [];
    for (let i = 0; i < CFG.GRAVITY_WELL_COUNT; i++) {
      gravityWells.push({
        x: Math.random() * W,
        y: Math.random() * H,
        strength: (0.5 + Math.random() * 0.5) * CFG.GRAVITY_WELL_STRENGTH,
        radius: (0.6 + Math.random() * 0.8) * CFG.GRAVITY_WELL_RADIUS,
        vx: (Math.random() - 0.5) * 0.04,
        vy: (Math.random() - 0.5) * 0.04,
      });
    }
  }
  initGravityWells();

  function updateGravityWells() {
    for (const well of gravityWells) {
      well.x += well.vx;
      well.y += well.vy;
      if (well.x < 0 || well.x > W) well.vx *= -1;
      if (well.y < 0 || well.y > H) well.vy *= -1;
    }
  }

  // ─── PARTICLES ───────────────────────────────────────────────────────────────
  let particles = [];
  function initParticles() {
    particles = [];
    for (let i = 0; i < CFG.PARTICLE_COUNT; i++) {
      const speed = CFG.PARTICLE_MIN_SPEED + Math.random() * (CFG.PARTICLE_MAX_SPEED - CFG.PARTICLE_MIN_SPEED);
      const angle = Math.random() * Math.PI * 2;
      const roll = Math.random();
      let colorRGB;
      if (roll < 0.70) colorRGB = CFG.COLOR_PARTICLE_PRIMARY;
      else if (roll < 0.90) colorRGB = CFG.COLOR_PARTICLE_SECONDARY;
      else colorRGB = CFG.COLOR_PARTICLE_ENERGY;

      particles.push({
        x: Math.random() * W,
        y: Math.random() * H,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        r: CFG.PARTICLE_MIN_RADIUS + Math.random() * (CFG.PARTICLE_MAX_RADIUS - CFG.PARTICLE_MIN_RADIUS),
        colorRGB,
        trail: [],
        opacity: 0.4 + Math.random() * 0.5,
        pulse: Math.random() * Math.PI * 2,
        pulseSpeed: 0.015 + Math.random() * 0.025,
      });
    }
  }
  initParticles();

  function updateParticles() {
    for (const p of particles) {
      p.trail.push({ x: p.x, y: p.y });
      if (p.trail.length > CFG.PARTICLE_TRAIL_LENGTH) p.trail.shift();

      for (const well of gravityWells) {
        const dx = well.x - p.x;
        const dy = well.y - p.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < well.radius && dist > 1) {
          const force = (well.strength / (dist * dist)) * 0.01;
          p.vx += dx * force;
          p.vy += dy * force;
        }
      }

      const speed = Math.sqrt(p.vx * p.vx + p.vy * p.vy);
      if (speed > CFG.PARTICLE_MAX_SPEED) {
        p.vx = (p.vx / speed) * CFG.PARTICLE_MAX_SPEED;
        p.vy = (p.vy / speed) * CFG.PARTICLE_MAX_SPEED;
      }
      p.vx += (Math.random() - 0.5) * 0.008;
      p.vy += (Math.random() - 0.5) * 0.008;

      p.x += p.vx;
      p.y += p.vy;

      if (p.x < 0)  p.x = W;
      if (p.x > W)  p.x = 0;
      if (p.y < 0)  p.y = H;
      if (p.y > H)  p.y = 0;

      p.pulse += p.pulseSpeed;
    }
  }

  function drawParticles() {
    for (const p of particles) {
      const speed = Math.sqrt(p.vx * p.vx + p.vy * p.vy);
      const isRelativistic = speed > CFG.RELATIVISTIC_THRESHOLD;

      if (isRelativistic && p.trail.length > 2) {
        for (let i = 1; i < p.trail.length; i++) {
          const trailOpacity = (i / p.trail.length) * 0.35 * (speed / CFG.PARTICLE_MAX_SPEED);
          ctx.beginPath();
          ctx.moveTo(p.trail[i - 1].x, p.trail[i - 1].y);
          ctx.lineTo(p.trail[i].x, p.trail[i].y);
          ctx.strokeStyle = `rgba(${p.colorRGB}, ${trailOpacity})`;
          ctx.lineWidth = p.r * 0.6 * (i / p.trail.length);
          ctx.stroke();
        }
      }

      const pulseOpacity = p.opacity * (0.75 + 0.25 * Math.sin(p.pulse));
      const glowRadius = p.r * (isRelativistic ? 4 : 2.5);

      const grad = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, glowRadius);
      grad.addColorStop(0,   `rgba(${p.colorRGB}, ${pulseOpacity})`);
      grad.addColorStop(0.4, `rgba(${p.colorRGB}, ${pulseOpacity * 0.4})`);
      grad.addColorStop(1,   `rgba(${p.colorRGB}, 0)`);

      ctx.beginPath();
      ctx.arc(p.x, p.y, glowRadius, 0, Math.PI * 2);
      ctx.fillStyle = grad;
      ctx.fill();

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${p.colorRGB}, ${pulseOpacity})`;
      ctx.fill();
    }
  }

  // ─── ENTROPY CONNECTION FIELD ─────────────────────────────────────────────────
  function drawConnections() {
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const a = particles[i];
        const b = particles[j];
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist > CFG.CONNECTION_DISTANCE) continue;

        const t = dist / CFG.CONNECTION_DISTANCE;
        const opacity = CFG.CONNECTION_MAX_OPACITY * (1 - t);

        const r = Math.round(124 + t * (245 - 124));
        const g = Math.round(58  + t * (158 - 58));
        const b_ = Math.round(237 + t * (11  - 237));

        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.strokeStyle = `rgba(${r}, ${g}, ${b_}, ${opacity})`;
        ctx.lineWidth = 0.6;
        ctx.stroke();
      }
    }
  }

  // ─── SPACETIME GRID (curved by gravity wells) ─────────────────────────────────
  function drawSpacetimeGrid() {
    const cellW = W / CFG.GRID_COLS;
    const cellH = H / CFG.GRID_ROWS;

    ctx.strokeStyle = `rgba(${CFG.COLOR_GRID}, 0.25)`;
    ctx.lineWidth = 0.4;

    for (let row = 0; row <= CFG.GRID_ROWS; row++) {
      ctx.beginPath();
      for (let col = 0; col <= CFG.GRID_COLS; col++) {
        const baseX = col * cellW;
        const baseY = row * cellH;
        const warped = warpPoint(baseX, baseY);
        if (col === 0) ctx.moveTo(warped.x, warped.y);
        else ctx.lineTo(warped.x, warped.y);
      }
      ctx.stroke();
    }

    for (let col = 0; col <= CFG.GRID_COLS; col++) {
      ctx.beginPath();
      for (let row = 0; row <= CFG.GRID_ROWS; row++) {
        const baseX = col * cellW;
        const baseY = row * cellH;
        const warped = warpPoint(baseX, baseY);
        if (row === 0) ctx.moveTo(warped.x, warped.y);
        else ctx.lineTo(warped.x, warped.y);
      }
      ctx.stroke();
    }
  }

  function warpPoint(x, y) {
    let wx = x, wy = y;
    for (const well of gravityWells) {
      const dx = x - well.x;
      const dy = y - well.y;
      const dist = Math.sqrt(dx * dx + dy * dy);
      if (dist < 1) continue;
      const warp = (CFG.GRID_WARP_STRENGTH * well.strength * 0.012) / (1 + dist * 0.008);
      wx -= (dx / dist) * warp;
      wy -= (dy / dist) * warp;
    }
    return { x: wx, y: wy };
  }

  // ─── GRAVITATIONAL WAVE RIPPLES ───────────────────────────────────────────────
  let waves = [];
  function spawnWave() {
    waves.push({
      x: Math.random() * W,
      y: Math.random() * H,
      radius: 0,
      maxRadius: 600,
      opacity: 0.55,
    });
    const delay = CFG.WAVE_INTERVAL_MIN + Math.random() * (CFG.WAVE_INTERVAL_MAX - CFG.WAVE_INTERVAL_MIN);
    setTimeout(spawnWave, delay);
  }
  setTimeout(spawnWave, 1200);

  function updateAndDrawWaves() {
    waves = waves.filter(w => w.opacity > 0.005);
    for (const w of waves) {
      w.radius += CFG.WAVE_SPEED;
      w.opacity *= 0.985;

      ctx.beginPath();
      ctx.arc(w.x, w.y, w.radius, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(${CFG.COLOR_WAVE}, ${w.opacity})`;
      ctx.lineWidth = 0.8;
      ctx.stroke();

      if (w.radius > 40) {
        ctx.beginPath();
        ctx.arc(w.x, w.y, w.radius * 0.65, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(${CFG.COLOR_WAVE}, ${w.opacity * 0.35})`;
        ctx.lineWidth = 0.4;
        ctx.stroke();
      }
    }
  }

  // ─── RIEMANNIAN MANIFOLDS (curved ellipse backgrounds) ────────────────────────
  let manifolds = [];
  function initManifolds() {
    for (let i = 0; i < CFG.MANIFOLD_COUNT; i++) {
      manifolds.push({
        cx: W * (0.2 + i * 0.3),
        cy: H * (0.25 + Math.random() * 0.5),
        rx: 180 + Math.random() * 220,
        ry: 80  + Math.random() * 120,
        angle: Math.random() * Math.PI,
        rotateSpeed: CFG.MANIFOLD_ROTATE_SPEED * (i % 2 === 0 ? 1 : -1),
      });
    }
  }
  initManifolds();

  function drawManifolds() {
    for (const m of manifolds) {
      m.angle += m.rotateSpeed;
      ctx.save();
      ctx.translate(m.cx, m.cy);
      ctx.rotate(m.angle);
      ctx.beginPath();
      ctx.ellipse(0, 0, m.rx, m.ry, 0, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(${CFG.COLOR_MANIFOLD}, ${CFG.MANIFOLD_OPACITY})`;
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.beginPath();
      ctx.ellipse(0, 0, m.rx * 0.6, m.ry * 0.6, 0, 0, Math.PI * 2);
      ctx.strokeStyle = `rgba(${CFG.COLOR_MANIFOLD}, ${CFG.MANIFOLD_OPACITY * 0.5})`;
      ctx.lineWidth = 0.5;
      ctx.stroke();
      ctx.restore();
    }
  }

  // ─── MAIN ANIMATION LOOP ─────────────────────────────────────────────────────
  function render() {
    ctx.fillStyle = 'rgba(13, 17, 23, 0.18)';
    ctx.fillRect(0, 0, W, H);

    updateGravityWells();
    updateParticles();

    drawManifolds();
    drawSpacetimeGrid();
    drawConnections();
    updateAndDrawWaves();
    drawParticles();

    requestAnimationFrame(render);
  }

  // Respect prefers-reduced-motion
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
  if (!reducedMotion.matches) {
    render();
  } else {
    ctx.fillStyle = 'rgba(13, 17, 23, 1)';
    ctx.fillRect(0, 0, W, H);
    drawManifolds();
    drawSpacetimeGrid();
  }

})();
