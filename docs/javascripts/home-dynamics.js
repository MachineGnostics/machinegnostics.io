/* =======================================================================
   MACHINE GNOSTICS — ASTROPHYSICS CANVAS ENGINE
   Renders: space-time curvature grid, star field, nebulae,
            gravitational wells with wave rings, photon drift nodes.
   Color palette: Teal #00d4aa | Cyan #00e5ff | Green #00e676
   ======================================================================= */
document.addEventListener('DOMContentLoaded', () => {
  const homeRoot = document.querySelector('.gn-home');
  if (!homeRoot) return;

  const canvas = homeRoot.querySelector('.gn-web-canvas');
  if (!canvas) return;

  /* ── helpers ──────────────────────────────────────────────────────── */
  const ctx  = canvas.getContext('2d');
  const rand = (lo, hi) => lo + Math.random() * (hi - lo);
  const lerp = (a, b, t) => a + (b - a) * t;
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  /* ── colour helpers ───────────────────────────────────────────────── */
  const TEAL  = [0, 212, 170];
  const CYAN  = [0, 229, 255];
  const GREEN = [0, 230, 118];
  const WHITE = [220, 245, 245];

  const rgba = ([r, g, b], a) => `rgba(${r},${g},${b},${a.toFixed(3)})`;

  /* ── comet particle colour palettes ──────────────────────────────── */
  // Dark mode: broad neon spectrum — teal/cyan/green anchored, plus warm accents
  const COMET_DARK = [
    [0,   212, 170],  // teal (brand)
    [0,   229, 255],  // cyan (brand)
    [0,   230, 118],  // green (brand)
    [100, 255, 218],  // bright teal
    [0,   200, 240],  // deep cyan
    [60,  250, 180],  // mint
    [0,   255, 200],  // aqua
    [140, 255, 230],  // ice teal
    [180, 255, 140],  // lime green
    [255, 220,  80],  // golden yellow
    [255, 165,  60],  // amber orange
    [200, 255, 100],  // yellow-green
    [80,  230, 255],  // sky blue
    [160, 200, 255],  // periwinkle
    [255, 200, 140],  // warm peach
    [200, 255, 200],  // pale green
  ];
  // Light mode: deep saturated — readable on white/light grey
  const COMET_LIGHT = [
    [0,   130, 105],  // deep teal
    [0,   130, 168],  // ocean blue
    [0,   148,  65],  // forest green
    [0,   155, 145],  // slate teal
    [10,  115, 155],  // steel cyan
    [0,   168, 115],  // jade
    [0,   180, 155],  // aqua teal
    [30,  145, 190],  // sky teal
    [90,  155,  30],  // olive green
    [180, 120,   0],  // amber
    [200, 140,  20],  // golden
    [30,  140, 190],  // cobalt
    [0,   120, 180],  // deep blue
    [120, 170,  40],  // lime
    [180,  90,  20],  // burnt orange
    [0,   160, 130],  // persian teal
  ];

  /* ── mouse cursor state ───────────────────────────────────────────── */
  const mouse = { x: -9999, y: -9999, active: false };

  /* ── light/dark mode ──────────────────────────────────────────────── */
  const isLight = () => {
    const s = (document.documentElement.getAttribute('data-md-color-scheme') ||
               document.body.getAttribute('data-md-color-scheme') || '');
    return s.toLowerCase() === 'default';
  };

  /* ── state ────────────────────────────────────────────────────────── */
  let W = 0, H = 0;
  let time = 0;
  let raf  = null;

  /* gravity wells  ── positions are set proportionally on resize */
  const wells = [
    { fx: 0.72, fy: 0.26, mass: 1.0 },
    { fx: 0.20, fy: 0.68, mass: 0.60 },
  ];
  let wArr = [];   /* computed pixel coords */

  /* stars */
  let stars = [];

  /* comet particles with colour trails */
  let comets = [];

  /* freely-drifting black holes */
  let mobHoles = [];

  /* periodic system events */
  let burstTimerSec = 60;
  let burstClickCount = 0;
  let burstActiveSec = 0;
  let superPhotonTimerSec = 10;
  let superPhotons = [];

  /* galaxy background — stardust band + core + distant galaxies */
  let galaxyDust    = [];
  let galaxySmudges = [];
  let galaxyCore    = { x: 0, y: 0 };

  /* first-visit intro: void -> big bang -> photon inflow */
  let intro = {
    enabled: false,
    phase: 'done',
    timerSec: 0,
    center: { x: 0, y: 0 },
    photons: [],
    blastPhotons: [],
    bangTriggered: false,
    shockwaves: [],
  };

  const shouldRunIntro = () => {
    return false;
  };

  const setIntroPageHidden = (hidden) => {
    homeRoot.classList.toggle('gn-intro-active', hidden);
  };

  const triggerStardustReveal = () => {
    homeRoot.classList.add('gn-intro-reveal');
    setTimeout(() => {
      homeRoot.classList.remove('gn-intro-reveal');
    }, 1700);
  };

  const newIntroPhoton = (center) => {
    const a = rand(0, Math.PI * 2);
    const speed = rand(2.4, 6.2);
    const palette = isLight() ? COMET_LIGHT : COMET_DARK;
    const col = palette[Math.floor(Math.random() * palette.length)];
    const jitter = Math.min(W, H) * 0.010;

    return {
      x: center.x + rand(-jitter, jitter),
      y: center.y + rand(-jitter, jitter),
      vx: Math.cos(a) * speed,
      vy: Math.sin(a) * speed,
      delaySec: rand(0, 1.20),
      life: 1,
      r: rand(0.9, 2.0),
      pulse: rand(0, Math.PI * 2),
      trail: [],
      trailLen: Math.floor(rand(8, 18)),
      color: col,
    };
  };

  const newIntroBlastPhoton = (center) => {
    const a = rand(0, Math.PI * 2);
    const speed = rand(3.6, 8.8);
    const palette = isLight() ? COMET_LIGHT : COMET_DARK;
    const col = palette[Math.floor(Math.random() * palette.length)];
    return {
      x: center.x,
      y: center.y,
      vx: Math.cos(a) * speed,
      vy: Math.sin(a) * speed,
      life: rand(1.3, 2.2),
      r: rand(0.9, 2.3),
      pulse: rand(0, Math.PI * 2),
      trail: [],
      trailLen: Math.floor(rand(8, 16)),
      color: col,
    };
  };

  const initIntro = () => {
    intro.enabled = true;
    intro.phase = 'void';
    intro.timerSec = 0;
    intro.center = { x: W * 0.5, y: H * 0.5 };
    /* performance-tuned outward photons with wide spread */
    const n = Math.min(260, Math.max(90, Math.floor((W * H) / 14000)));
    intro.photons = Array.from({ length: n }, () => newIntroPhoton(intro.center));
    intro.blastPhotons = [];
    intro.bangTriggered = false;
    intro.shockwaves = [];
    setIntroPageHidden(true);
  };

  const updateIntro = (dtSec) => {
    intro.timerSec += dtSec;

    if (intro.timerSec < 0.22) {
      intro.phase = 'void';
      return;
    }
    if (intro.timerSec < 1.08) {
      intro.phase = 'sun';
      return;
    }
    if (intro.timerSec < 1.88) {
      intro.phase = 'bang';
    } else if (intro.timerSec < 4.0) {
      intro.phase = 'inflow';
    } else {
      intro.enabled = false;
      intro.phase = 'done';
      setIntroPageHidden(false);
      triggerStardustReveal();
      buildScene();
      emitPhotonBurst(intro.center);
      return;
    }

    if (!intro.bangTriggered && intro.timerSec >= 1.08) {
      intro.bangTriggered = true;
      /* reduced density for smooth playback while keeping dramatic spread */
      const nBlast = Math.min(300, Math.max(120, Math.floor((W * H) / 10666)));
      intro.blastPhotons = Array.from({ length: nBlast }, () => newIntroBlastPhoton(intro.center));
      /* multiple very fast shock fronts so rings sweep beyond screen */
      intro.shockwaves = [
        { r: 0, speed: Math.min(W, H) * 1.15, alpha: 0.70 },
        { r: 0, speed: Math.min(W, H) * 1.00, alpha: 0.58 },
        { r: 0, speed: Math.min(W, H) * 0.88, alpha: 0.46 },
        { r: 0, speed: Math.min(W, H) * 0.76, alpha: 0.38 },
      ];
    }

    const flowTime = intro.timerSec - 1.88;
    const frameScale = clamp(dtSec * 60, 0.45, 2.2);
    for (const p of intro.photons) {
      if (flowTime < p.delaySec) continue;

      p.trail.push({ x: p.x, y: p.y });
      if (p.trail.length > p.trailLen) p.trail.shift();

      p.x += p.vx * frameScale;
      p.y += p.vy * frameScale;
      p.vx *= 0.998;
      p.vy *= 0.998;
      p.pulse += 0.06;

      const outMargin = 150;
      if (p.x < -outMargin || p.x > W + outMargin || p.y < -outMargin || p.y > H + outMargin) {
        p.life = Math.max(0, p.life - dtSec * 4.0);
      }
      p.life = Math.max(0, p.life - dtSec * 0.22);
    }

    for (let i = intro.blastPhotons.length - 1; i >= 0; i--) {
      const p = intro.blastPhotons[i];
      p.life -= dtSec;
      if (p.life <= 0) {
        intro.blastPhotons.splice(i, 1);
        continue;
      }

      p.trail.push({ x: p.x, y: p.y });
      if (p.trail.length > p.trailLen) p.trail.shift();

      p.x += p.vx * frameScale;
      p.y += p.vy * frameScale;
      p.vx *= 0.996;
      p.vy *= 0.996;
      p.pulse += 0.05;
    }

    const diag = Math.hypot(W, H) * 1.20;
    for (let i = intro.shockwaves.length - 1; i >= 0; i--) {
      const w = intro.shockwaves[i];
      w.r += w.speed * dtSec;
      w.alpha = Math.max(0, w.alpha - dtSec * 0.42);
      if (w.r > diag || w.alpha <= 0.01) intro.shockwaves.splice(i, 1);
    }
  };

  /* ── canvas resize ────────────────────────────────────────────────── */
  const resize = () => {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    W = window.innerWidth;
    H = window.innerHeight;
    canvas.width  = Math.floor(W * dpr);
    canvas.height = Math.floor(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    buildScene();
  };

  const buildScene = () => {
    /* gravity wells in pixels */
    wArr = wells.map(w => ({
      x: w.fx * W, y: w.fy * H, mass: w.mass,
      phase: rand(0, Math.PI * 2),
    }));

    /* star field */
    const n = Math.max(120, Math.floor((W * H) / 6000));
    const starColors = isLight()
      ? [[0,140,110], [0,130,168], [0,148,65], [60,80,100], [80,100,120]]  /* subtle dots on light */
      : [WHITE, CYAN, TEAL, [180,245,240], [160,235,230]];                  /* bright on dark */
    stars = Array.from({ length: n }, () => ({
      x:     rand(0, W),
      y:     rand(0, H),
      r:     rand(0.25, 1.6),
      alpha: isLight() ? rand(0.08, 0.30) : rand(0.15, 0.90),
      speed: rand(0.0008, 0.004),
      phase: rand(0, Math.PI * 2),
      color: starColors[Math.floor(Math.random() * starColors.length)],
    }));

    /* comet particles with colour trails */
    const pn = Math.max(45, Math.floor((W * H) / 16000));
    comets = Array.from({ length: pn }, () => newComet());

    /* mobile black holes — 4 small freely-drifting holes */
    /* stagger initial age so the 4 holes don't explode at the same time */
    mobHoles = Array.from({ length: 4 }, (_, i) => {
      const bh = newMobHole();
      bh.ageSec += i * 8;
      return bh;
    });

    superPhotons = [];
    burstTimerSec = 60;
    burstClickCount = 0;
    burstActiveSec = 0;
    superPhotonTimerSec = 10;

    /* ── galaxy background data ─────────────────────────────────────── */
    const gCX = W * 0.52, gCY = H * 0.46;   /* galactic centre on screen */
    galaxyCore = { x: gCX, y: gCY };

    /* galactic band tilted ~25° — sample points biased toward it */
    const bAng  = 0.42;          /* band tilt in radians */
    const bCos  = Math.cos(bAng), bSin = Math.sin(bAng);
    const bHalf = H * 0.26;      /* half-width of bright band */
    const coreR = Math.min(W, H) * 0.24;  /* core influence radius */
    const nDust = Math.max(800, Math.floor((W * H) / 1800));
    galaxyDust = Array.from({ length: nDust }, () => {
      const px = rand(0, W), py = rand(0, H);
      const rx =  (px - gCX) * bCos + (py - gCY) * bSin;  /* along band */
      const ry = -(px - gCX) * bSin + (py - gCY) * bCos;  /* across band */
      const bandDens = Math.exp(-(ry * ry) / (bHalf * bHalf * 0.55));
      const coreDens = Math.exp(-Math.sqrt(rx*rx + ry*ry) / coreR);
      const density  = bandDens * 0.72 + coreDens * 0.28;
      const h = Math.random();
      return {
        x:     px,
        y:     py,
        r:     rand(0.10, 0.52),
        base:  rand(0.04, 0.50) * (0.30 + density * 0.70),
        speed: rand(0.0003, 0.0014),
        phase: rand(0, Math.PI * 2),
        hue:   h,   /* 0=blue-white, .33=white, .66=warm, 1=ice */
      };
    });

    /* distant edge-on + face-on galaxy smudges */
    galaxySmudges = [
      { x: W*0.10, y: H*0.16, rx: W*0.060, ry: H*0.006, ang:  0.30, col: [200,210,255], a: 0.13 },
      { x: W*0.90, y: H*0.78, rx: W*0.045, ry: H*0.005, ang: -0.50, col: [255,215,200], a: 0.11 },
      { x: W*0.78, y: H*0.10, rx: W*0.034, ry: H*0.034, ang:  0.00, col: [215,205,255], a: 0.09 },
      { x: W*0.04, y: H*0.82, rx: W*0.028, ry: H*0.004, ang:  1.10, col: [195,240,255], a: 0.10 },
      { x: W*0.55, y: H*0.92, rx: W*0.022, ry: H*0.003, ang: -0.20, col: [220,200,255], a: 0.08 },
    ];

    if (intro.enabled) {
      initIntro();
    }
  };

  const newMobHole = () => {
    const angle = rand(0, Math.PI * 2);
    const spd   = rand(0.12, 0.38);
    const baseR = rand(7, 17);
    return {
      x:             rand(W * 0.05, W * 0.95),
      y:             rand(H * 0.05, H * 0.95),
      vx:            Math.cos(angle) * spd,
      vy:            Math.sin(angle) * spd,
      mass:          rand(0.30, 0.65),
      baseR,
      r:             baseR,
      phase:         rand(0, Math.PI * 2),
      turn:          rand(-0.005, 0.005),
      consumed:      0,
      capturedCount: 0,
      lifeSpanSec:   60,
      ageSec:        rand(0, 14),
      state:         'active',        /* 'active' | 'exploding' */
      explodeTimerSec: 0,
      explodeRing:   0,
      flashAlpha:    0,
    };
  };

  const newComet = () => {
    const palette = isLight() ? COMET_LIGHT : COMET_DARK;
    const col     = palette[Math.floor(Math.random() * palette.length)];
    const spd     = rand(0.18, 0.55);
    const angle   = rand(0, Math.PI * 2);
    return {
      x:        rand(0, W),
      y:        rand(0, H),
      vx:       Math.cos(angle) * spd,
      vy:       Math.sin(angle) * spd,
      r:        rand(1.0, 2.4),
      alpha:    rand(0.50, 0.90),
      pulse:    rand(0, Math.PI * 2),
      color:    col,
      trail:    [],
      trailLen: Math.floor(rand(14, 32)),
    };
  };

  const spawnSuperPhoton = () => {
    const side = Math.floor(rand(0, 4));
    const margin = 28;
    let x = 0, y = 0, tx = 0, ty = 0;
    if (side === 0) {
      x = rand(0, W); y = -margin;
      tx = rand(0, W); ty = H + margin;
    } else if (side === 1) {
      x = W + margin; y = rand(0, H);
      tx = -margin; ty = rand(0, H);
    } else if (side === 2) {
      x = rand(0, W); y = H + margin;
      tx = rand(0, W); ty = -margin;
    } else {
      x = -margin; y = rand(0, H);
      tx = W + margin; ty = rand(0, H);
    }

    /* slight cross-axis jitter so paths feel less uniform */
    tx = clamp(tx + rand(-W * 0.18, W * 0.18), -margin, W + margin);
    ty = clamp(ty + rand(-H * 0.18, H * 0.18), -margin, H + margin);

    const dx = tx - x;
    const dy = ty - y;
    const d = Math.sqrt(dx * dx + dy * dy) + 0.001;
    const lifeSec = 3;
    /* cross entire path within lifespan (units: px/frame at 60fps) */
    const speed = (d / (lifeSec * 60)) * rand(1.03, 1.12);

    superPhotons.push({
      x,
      y,
      vx: (dx / d) * speed,
      vy: (dy / d) * speed,
      r: rand(1.0, 2.4), /* same size range as regular photons */
      lifeSec,
      z: rand(-0.9, 0.9),
      vz: rand(-1.2, 1.2),
      trail: [],
      trailLen: Math.floor(rand(18, 30)),
      phase: rand(0, Math.PI * 2),
    });
  };

  const emitPhotonBurst = (origin = null) => {
    const cx = origin ? origin.x : rand(W * 0.20, W * 0.80);
    const cy = origin ? origin.y : rand(H * 0.20, H * 0.80);
    const burst = 28;
    for (let i = 0; i < burst; i++) {
      const ba = (i / burst) * Math.PI * 2 + rand(-0.20, 0.20);
      const bs = rand(0.9, 2.2);
      const c  = newComet();
      c.x = cx;
      c.y = cy;
      c.vx = Math.cos(ba) * bs;
      c.vy = Math.sin(ba) * bs;
      c.trail = [];
      c.r = rand(0.9, 2.0);
      comets.push(c);
    }
    burstActiveSec = 2.0;

    const maxC = Math.max(90, Math.floor((W * H) / 11000));
    if (comets.length > maxC) comets.splice(0, comets.length - maxC);
  };

  const triggerHoleExplosion = (bh) => {
    if (!bh || bh.state !== 'active') return;

    bh.state = 'exploding';
    bh.explodeTimerSec = 1.8;
    bh.flashAlpha = 1.0;

    /* release all orbiting comets outward (slower repulsion) */
    for (const p of comets) {
      if (p.capturedBy === bh) {
        const oa = (p.orbitAngle || 0) + rand(-0.9, 0.9);
        p.vx = Math.cos(oa) * rand(0.8, 1.9);
        p.vy = Math.sin(oa) * rand(0.8, 1.9);
        p.capturedBy = null;
        p.trail = [];
      }
    }

    /* emit burst photons in all directions */
    const burst = clamp(12 + Math.floor(bh.consumed * 1.3), 12, 24);
    for (let i = 0; i < burst; i++) {
      const ba = (i / burst) * Math.PI * 2 + rand(-0.35, 0.35);
      const bs = rand(1.0, 2.4);
      const c  = newComet();
      c.x = bh.x;
      c.y = bh.y;
      c.vx = Math.cos(ba) * bs;
      c.vy = Math.sin(ba) * bs;
      c.trail = [];
      comets.push(c);
    }

    const maxC = Math.max(90, Math.floor((W * H) / 11000));
    if (comets.length > maxC) comets.splice(0, comets.length - maxC);
  };

  /* ── gravity warp offset ─────────────────────────────────────────── */
  const warpAt = (gx, gy) => {
    let dx = 0, dy = 0;
    const scale = Math.min(W, H);
    for (const w of [...wArr, ...mobHoles]) {
      const ex = gx - w.x;
      const ey = gy - w.y;
      const d2 = ex * ex + ey * ey;
      const maxR = scale * 0.45;
      const falloff = 1 / (1 + d2 / (maxR * maxR * 0.07));
      const strength = w.mass * 32 * falloff;
      const d = Math.sqrt(d2) + 0.001;
      dx -= (ex / d) * strength;
      dy -= (ey / d) * strength;
    }

    /* super photons distort local space-time like tiny fast black holes */
    for (const sp of superPhotons) {
      const ex = gx - sp.x;
      const ey = gy - sp.y;
      const d2 = ex * ex + ey * ey;
      const maxR = scale * 0.22;
      const falloff = 1 / (1 + d2 / (maxR * maxR * 0.06));
      const life = clamp(sp.lifeSec / 3, 0, 1);
      const strength = (5.0 + life * 2.0) * falloff;
      const d = Math.sqrt(d2) + 0.001;
      dx -= (ex / d) * strength;
      dy -= (ey / d) * strength;
    }

    return { dx, dy };
  };

  /* ── update: mobile black holes (positions + lifecycle state machine) ── */
  const updateMobHoles = (dtSec) => {
    const frameScale = clamp(dtSec * 60, 0.45, 2.2);
    for (const bh of mobHoles) {

      /* ── EXPLODING: animate shockwave, then respawn ── */
      if (bh.state === 'exploding') {
        bh.explodeTimerSec -= dtSec;
        bh.explodeRing += Math.min(W, H) * 0.16 * dtSec; /* 2x slower repel wave */
        bh.flashAlpha = clamp(bh.explodeTimerSec / 0.95, 0, 1);
        if (bh.explodeTimerSec <= 0) {
          /* respawn at fresh random position */
          const a2 = rand(0, Math.PI * 2), s2 = rand(0.12, 0.38);
          bh.x = rand(W * 0.08, W * 0.92);  bh.y = rand(H * 0.08, H * 0.92);
          bh.vx = Math.cos(a2) * s2;        bh.vy = Math.sin(a2) * s2;
          bh.turn = rand(-0.005, 0.005);
          bh.consumed = 0;
          bh.capturedCount = 0;
          bh.ageSec = 0;
          bh.r = bh.baseR;
          bh.state = 'active';
          bh.explodeRing = 0;
          bh.flashAlpha = 0;
        }
        continue;
      }

      /* ── ACTIVE: drift + lifecycle countdown ── */
      const angle = Math.atan2(bh.vy, bh.vx) + bh.turn * frameScale;
      const spd   = Math.sqrt(bh.vx * bh.vx + bh.vy * bh.vy);
      bh.vx = Math.cos(angle) * spd;
      bh.vy = Math.sin(angle) * spd;
      bh.x += bh.vx * frameScale;
      bh.y += bh.vy * frameScale;
      bh.phase += 0.014 * frameScale;
      if (bh.x < -90)    bh.x = W + 90;
      if (bh.x > W + 90) bh.x = -90;
      if (bh.y < -90)    bh.y = H + 90;
      if (bh.y > H + 90) bh.y = -90;

      if (burstActiveSec <= 0) bh.ageSec += dtSec;

      const lifeRatio = clamp(bh.ageSec / bh.lifeSpanSec, 0, 1);
      const growthByLife = 1 + lifeRatio * 0.75;
      const growthByMass = 1 + Math.min(0.55, bh.consumed * 0.05);
      bh.r = bh.baseR * growthByLife * growthByMass;

      if (bh.ageSec >= bh.lifeSpanSec || bh.consumed >= 14) {
        triggerHoleExplosion(bh);
      }
    }
  };

  const updateSuperPhotons = (dtSec) => {
    const frameScale = clamp(dtSec * 60, 0.45, 2.2);

    superPhotonTimerSec -= dtSec;
    if (superPhotonTimerSec <= 0) {
      spawnSuperPhoton();
      superPhotonTimerSec = 10;
    }

    for (let i = superPhotons.length - 1; i >= 0; i--) {
      const sp = superPhotons[i];
      sp.lifeSec -= dtSec;
      if (sp.lifeSec <= 0) {
        superPhotons.splice(i, 1);
        continue;
      }

      sp.trail.push({ x: sp.x, y: sp.y, z: sp.z });
      if (sp.trail.length > sp.trailLen) sp.trail.shift();

      sp.x += sp.vx * frameScale;
      sp.y += sp.vy * frameScale;
      sp.z += sp.vz * dtSec;
      sp.phase += 0.08 * frameScale;

      if (sp.z < -1.25 || sp.z > 1.25) sp.vz *= -1;

      if (sp.x < -40 || sp.x > W + 40 || sp.y < -40 || sp.y > H + 40) {
        superPhotons.splice(i, 1);
      }
    }
  };

  /* ── draw: galaxy — deep space background, milky way band, stardust ── */
  const drawGalaxy = () => {
    const { x: cX, y: cY } = galaxyCore;

    /* ── deep black space fill (dark theme only) ── */
    if (!isLight()) {
      ctx.fillStyle = '#00020e';
      ctx.fillRect(0, 0, W, H);
    }

    /* ── milky way band ── */
    const bAng  = 0.42;
    const bLen  = Math.sqrt(W * W + H * H);
    const bHalf = H * 0.28;
    ctx.save();
    ctx.translate(cX, cY);
    ctx.rotate(bAng);
    /* wide diffuse outer glow */
    const bandOuter = ctx.createLinearGradient(0, -bHalf, 0, bHalf);
    bandOuter.addColorStop(0,    'transparent');
    bandOuter.addColorStop(0.28, isLight() ? 'rgba(0,140,110,0.012)' : 'rgba(155,175,255,0.016)');
    bandOuter.addColorStop(0.44, isLight() ? 'rgba(0,160,130,0.028)' : 'rgba(185,205,255,0.034)');
    bandOuter.addColorStop(0.50, isLight() ? 'rgba(0,175,140,0.038)' : 'rgba(210,225,255,0.052)');
    bandOuter.addColorStop(0.56, isLight() ? 'rgba(0,160,130,0.028)' : 'rgba(185,205,255,0.034)');
    bandOuter.addColorStop(0.72, isLight() ? 'rgba(0,140,110,0.012)' : 'rgba(155,175,255,0.016)');
    bandOuter.addColorStop(1,    'transparent');
    ctx.fillStyle = bandOuter;
    ctx.fillRect(-bLen * 0.52, -bHalf, bLen * 1.04, bHalf * 2);
    /* bright inner lane */
    const bInner = bHalf * 0.30;
    const bandInner = ctx.createLinearGradient(0, -bInner, 0, bInner);
    bandInner.addColorStop(0,    'transparent');
    bandInner.addColorStop(0.35, isLight() ? 'rgba(0,180,150,0.022)' : 'rgba(220,235,255,0.038)');
    bandInner.addColorStop(0.50, isLight() ? 'rgba(0,200,160,0.032)' : 'rgba(235,245,255,0.068)');
    bandInner.addColorStop(0.65, isLight() ? 'rgba(0,180,150,0.022)' : 'rgba(220,235,255,0.038)');
    bandInner.addColorStop(1,    'transparent');
    ctx.fillStyle = bandInner;
    ctx.fillRect(-bLen * 0.52, -bInner, bLen * 1.04, bInner * 2);
    ctx.restore();

    /* ── galactic core bloom ── */
    if (!isLight()) {
      /* outer warm haze */
      const cR1 = Math.min(W, H) * 0.32;
      const cG1 = ctx.createRadialGradient(cX, cY, 0, cX, cY, cR1);
      cG1.addColorStop(0,    'rgba(255,235,190,0.14)');
      cG1.addColorStop(0.15, 'rgba(255,220,160,0.09)');
      cG1.addColorStop(0.38, 'rgba(200,210,255,0.04)');
      cG1.addColorStop(0.70, 'rgba(140,160,255,0.015)');
      cG1.addColorStop(1,    'transparent');
      ctx.beginPath(); ctx.arc(cX, cY, cR1, 0, Math.PI * 2);
      ctx.fillStyle = cG1; ctx.fill();
      /* inner bright pinpoint */
      const cR2 = Math.min(W, H) * 0.055;
      const cG2 = ctx.createRadialGradient(cX, cY, 0, cX, cY, cR2);
      cG2.addColorStop(0,    'rgba(255,255,230,0.55)');
      cG2.addColorStop(0.30, 'rgba(255,245,200,0.22)');
      cG2.addColorStop(0.65, 'rgba(255,220,160,0.07)');
      cG2.addColorStop(1,    'transparent');
      ctx.beginPath(); ctx.arc(cX, cY, cR2, 0, Math.PI * 2);
      ctx.fillStyle = cG2; ctx.fill();
    }

    /* ── distant galaxy smudges ── */
    if (!isLight()) {
      for (const gs of galaxySmudges) {
        ctx.save();
        ctx.translate(gs.x, gs.y);
        ctx.rotate(gs.ang);
        ctx.scale(1, gs.ry / gs.rx);
        const g = ctx.createRadialGradient(0, 0, 0, 0, 0, gs.rx);
        g.addColorStop(0,   rgba(gs.col, gs.a));
        g.addColorStop(0.4, rgba(gs.col, gs.a * 0.42));
        g.addColorStop(1,   'transparent');
        ctx.beginPath(); ctx.arc(0, 0, gs.rx, 0, Math.PI * 2);
        ctx.fillStyle = g; ctx.fill();
        ctx.restore();
      }
    }

    /* ── galaxy dust (stardust layer) ── */
    for (const d of galaxyDust) {
      const tw = 0.40 + 0.60 * Math.sin(time * d.speed * 62 + d.phase);
      const a  = d.base * tw * (isLight() ? 0.55 : 1.0);
      if (a < 0.012) continue;
      /* star colour: blue-white / pure white / warm / ice based on hue */
      let col;
      if      (d.hue < 0.25) col = isLight() ? `rgba(0,130,110,${a.toFixed(3)})`   : `rgba(180,200,255,${a.toFixed(3)})`;
      else if (d.hue < 0.50) col = isLight() ? `rgba(0,140,140,${a.toFixed(3)})`   : `rgba(255,255,255,${a.toFixed(3)})`;
      else if (d.hue < 0.75) col = isLight() ? `rgba(100,80,30,${a.toFixed(3)})`   : `rgba(255,240,200,${a.toFixed(3)})`;
      else                   col = isLight() ? `rgba(20,100,160,${a.toFixed(3)})`  : `rgba(215,240,255,${a.toFixed(3)})`;
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.arc(d.x, d.y, d.r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  };

  /* ── draw: nebula background ─────────────────────────────────────── */
  const drawNebulae = () => {
    const t = time * 0.06;
    const cx1 = W * 0.16 + Math.sin(t) * 28;
    const cy1 = H * 0.22 + Math.cos(t * 0.7) * 20;
    const cx2 = W * 0.84 + Math.sin(t * 0.85 + 1) * 22;
    const cy2 = H * 0.76 + Math.cos(t * 1.05) * 16;
    const cx3 = W * 0.50 + Math.sin(t * 0.55 + 2) * 18;
    const cy3 = H * 0.48 + Math.cos(t * 0.45 + 1) * 14;

    const blob = (cx, cy, rx, ry, col, a) => {
      const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, Math.max(rx, ry));
      g.addColorStop(0,   rgba(col, a));
      g.addColorStop(0.5, rgba(col, a * 0.38));
      g.addColorStop(1,   'transparent');
      ctx.save();
      ctx.scale(1, ry / rx);
      ctx.beginPath();
      ctx.arc(cx, cy * rx / ry, rx, 0, Math.PI * 2);
      ctx.fillStyle = g;
      ctx.fill();
      ctx.restore();
    };

    if (isLight()) {
      blob(cx1, cy1, W * 0.28, H * 0.20, TEAL,  0.055);
      blob(cx2, cy2, W * 0.25, H * 0.18, CYAN,  0.045);
      blob(cx3, cy3, W * 0.18, H * 0.14, GREEN, 0.030);
    } else {
      blob(cx1, cy1, W * 0.30, H * 0.22, TEAL,  0.042);
      blob(cx2, cy2, W * 0.27, H * 0.20, CYAN,  0.032);
      blob(cx3, cy3, W * 0.20, H * 0.15, GREEN, 0.022);
    }
  };

  /* ── draw: stars ─────────────────────────────────────────────────── */
  const drawStars = () => {
    for (const s of stars) {
      const tw = 0.45 + 0.55 * Math.sin(time * s.speed * 90 + s.phase);
      ctx.globalAlpha = s.alpha * tw;
      ctx.fillStyle   = rgba(s.color, 1);
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  };

  /* ── draw: space-time curvature grid ─────────────────────────────── */
  const drawGrid = () => {
    const cols = 24, rows = 15;
    const cw = W / cols;
    const ch = H / rows;
    const light = isLight();

    ctx.save();
    ctx.lineWidth = light ? 0.55 : 0.50;

    const lineAlpha = (gx, gy) => {
      const minD2 = Math.min(...wArr.map(w =>
        (gx - w.x) ** 2 + (gy - w.y) ** 2
      ));
      const scale = Math.min(W, H) * 0.55;
      const t = clamp(Math.sqrt(minD2) / scale, 0, 1);
      return light
        ? lerp(0.22, 0.07, t)
        : lerp(0.18, 0.05, t);
    };

    /* horizontal lines */
    for (let j = 0; j <= rows; j++) {
      ctx.beginPath();
      let started = false;
      for (let i = 0; i <= cols; i++) {
        const gx = i * cw, gy = j * ch;
        const { dx, dy } = warpAt(gx, gy);
        const px = gx + dx, py = gy + dy;
        ctx.strokeStyle = rgba(TEAL, lineAlpha(gx, gy));
        if (!started) { ctx.moveTo(px, py); started = true; }
        else            ctx.lineTo(px, py);
      }
      ctx.stroke();
    }

    /* vertical lines */
    for (let i = 0; i <= cols; i++) {
      ctx.beginPath();
      let started = false;
      for (let j = 0; j <= rows; j++) {
        const gx = i * cw, gy = j * ch;
        const { dx, dy } = warpAt(gx, gy);
        const px = gx + dx, py = gy + dy;
        ctx.strokeStyle = rgba(CYAN, lineAlpha(gx, gy) * 0.6);
        if (!started) { ctx.moveTo(px, py); started = true; }
        else            ctx.lineTo(px, py);
      }
      ctx.stroke();
    }

    ctx.restore();
  };

  /* ── draw: gravity well event horizons + wave rings ─────────────── */
  const drawGravityWells = () => {
    const light = isLight();
    for (const w of wArr) {
      const pulse = 0.72 + 0.28 * Math.sin(time * 0.55 + w.phase);
      const r0 = w.mass * 30 * pulse;

      /* glow */
      const glow = ctx.createRadialGradient(w.x, w.y, 0, w.x, w.y, r0 * 5);
      glow.addColorStop(0,   rgba(TEAL, light ? 0.18 * w.mass : 0.22 * w.mass));
      glow.addColorStop(0.35,rgba(CYAN, light ? 0.06 * w.mass : 0.09 * w.mass));
      glow.addColorStop(1,  'transparent');
      ctx.beginPath();
      ctx.arc(w.x, w.y, r0 * 5, 0, Math.PI * 2);
      ctx.fillStyle = glow;
      ctx.fill();

      /* gravitational wave rings */
      for (let ring = 0; ring < 5; ring++) {
        const phase = ((time * 0.38 + ring * 0.55) % 1);
        const ringR = r0 * (1.2 + phase * 8);
        const ringA = (1 - phase) * (light ? 0.14 : 0.20) * w.mass;
        if (ringA < 0.005) continue;
        ctx.strokeStyle = rgba(TEAL, ringA);
        ctx.lineWidth   = 1.1 * (1 - phase * 0.7);
        ctx.beginPath();
        ctx.arc(w.x, w.y, ringR, 0, Math.PI * 2);
        ctx.stroke();
      }

      /* centre bright point */
      ctx.beginPath();
      ctx.arc(w.x, w.y, r0 * 0.35, 0, Math.PI * 2);
      const cp = ctx.createRadialGradient(w.x, w.y, 0, w.x, w.y, r0 * 0.35);
      cp.addColorStop(0, rgba(WHITE, light ? 0.55 : 0.70));
      cp.addColorStop(1, 'transparent');
      ctx.fillStyle = cp;
      ctx.fill();
    }
  };

  /* ── draw: mobile black holes ──────────────────────────────────── */
  const drawMobileBlackHoles = () => {
    const light = isLight();
    for (const bh of mobHoles) {
      const { x, y, r, mass, phase } = bh;

      /* ── EXPLODING: shockwave flash ── */
      if (bh.state === 'exploding') {
        const fa = bh.flashAlpha;
        /* central flash bloom */
        if (fa > 0.04) {
          const fr = r * 10 * fa;
          const fl = ctx.createRadialGradient(x, y, 0, x, y, fr);
          fl.addColorStop(0,    rgba(WHITE, fa * 0.92));
          fl.addColorStop(0.22, rgba(CYAN,  fa * 0.72));
          fl.addColorStop(0.55, rgba(TEAL,  fa * 0.38));
          fl.addColorStop(1,   'transparent');
          ctx.beginPath(); ctx.arc(x, y, fr, 0, Math.PI * 2);
          ctx.fillStyle = fl; ctx.fill();
        }
        /* expanding shockwave rings */
        if (bh.explodeRing > 0) {
          const t = clamp(1 - fa, 0, 1);
          for (let i = 0; i < 4; i++) {
            const wavePhase = t * 6.5 + i * 1.25;
            const ripple = Math.sin(wavePhase) * (2.2 - i * 0.35);
            const rr = bh.explodeRing * (1 - i * 0.16) + ripple;
            if (rr <= 1) continue;

            const wa = clamp(fa * (0.55 - i * 0.10), 0, 1);
            ctx.lineWidth = Math.max(0.22, 0.56 - i * 0.08); /* thin water-wave lines */
            ctx.strokeStyle = i % 2 === 0
              ? rgba(WHITE, wa)
              : rgba(TEAL, wa * 0.95);
            ctx.beginPath();
            ctx.arc(x, y, rr, 0, Math.PI * 2);
            ctx.stroke();
          }
        }
        continue; /* skip normal visuals */
      }

      /* ── ACTIVE ── */
      const pulse        = 0.82 + 0.18 * Math.sin(time * 0.9 + phase);
      const accreteBoost = 1 + (bh.capturedCount || 0) * 0.28; /* glow brightens as comets spiral in */
      const halfLife = bh.ageSec >= (bh.lifeSpanSec * 0.5);

      /* outer lensing glow */
      const glow = ctx.createRadialGradient(x, y, r * 0.5, x, y, r * 7);
      glow.addColorStop(0,   rgba(TEAL, (light ? 0.12 : 0.17) * mass * accreteBoost));
      glow.addColorStop(0.4, rgba(CYAN, (light ? 0.05 : 0.07) * mass * accreteBoost));
      glow.addColorStop(1,  'transparent');
      ctx.beginPath(); ctx.arc(x, y, r * 7, 0, Math.PI * 2);
      ctx.fillStyle = glow; ctx.fill();

      /* expanding gravitational wave ring */
      const wavePhase = (time * 0.38 + phase * 0.25) % 1;
      const waveR = r * (1.4 + wavePhase * 6);
      const waveA = (1 - wavePhase) * (light ? 0.14 : 0.22) * mass;
      if (waveA > 0.004) {
        ctx.strokeStyle = rgba(TEAL, waveA);
        ctx.lineWidth   = 0.9 * (1 - wavePhase * 0.65);
        ctx.beginPath(); ctx.arc(x, y, waveR, 0, Math.PI * 2); ctx.stroke();
      }

      /* photon sphere ring */
      ctx.save();
      ctx.strokeStyle = rgba(TEAL, (light ? 0.55 : 0.70) * mass * pulse * accreteBoost);
      ctx.lineWidth   = 1.4 * pulse;
      ctx.beginPath(); ctx.arc(x, y, r * 1.55 * pulse, 0, Math.PI * 2); ctx.stroke();

      /* half-life golden ring */
      if (halfLife) {
        const goldAlpha = (light ? 0.30 : 0.46) * (0.8 + 0.2 * Math.sin(time * 3.4 + phase));
        ctx.strokeStyle = `rgba(255, 212, 96, ${goldAlpha.toFixed(3)})`;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.arc(x, y, r * 1.95 + Math.sin(time * 1.8 + phase) * 0.9, 0, Math.PI * 2);
        ctx.stroke();
      }

      /* secondary cyan ring */
      ctx.strokeStyle = rgba(CYAN, (light ? 0.28 : 0.38) * mass);
      ctx.lineWidth   = 0.65;
      ctx.beginPath(); ctx.arc(x, y, r * 2.3, 0, Math.PI * 2); ctx.stroke();
      ctx.restore();

      /* black void core */
      const core = ctx.createRadialGradient(x, y, 0, x, y, r);
      core.addColorStop(0,    light ? 'rgba(0,0,0,0.96)' : 'rgba(0,4,10,0.92)');
      core.addColorStop(0.60, light ? 'rgba(0,0,0,0.82)' : 'rgba(1,6,12,0.78)');
      core.addColorStop(1,    'rgba(0,0,0,0)');
      ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.fillStyle = core; ctx.fill();
    }
  };

  /* ── draw: comet particles with colour trails ──────────────────── */
  const drawComets = () => {
    const light     = isLight();
    const REPULSE_R = 220;
    const REPULSE_F = 2.0;
    const MAX_SPD   = 2.5;
    const DAMP      = 0.991;
    const freezePulse = burstActiveSec > 0;

    for (const p of comets) {
      let orbiting = false;

      /* ── CAPTURED: spiral-inward orbital physics ── */
      if (p.capturedBy && p.capturedBy.state === 'active') {
        const bh = p.capturedBy;
        orbiting = true;

        /* shrink orbit radius + spin up (angular momentum pseudo-conservation) */
        p.orbitR   = Math.max(bh.r * 0.55, p.orbitR - 0.10);
        p.orbitSpd = clamp(p.orbitSpd * 1.004, -0.32, 0.32);
        p.orbitAngle += p.orbitSpd;

        p.trail.push({ x: p.x, y: p.y });
        if (p.trail.length > p.trailLen) p.trail.shift();

        p.x = bh.x + Math.cos(p.orbitAngle) * p.orbitR;
        p.y = bh.y + Math.sin(p.orbitAngle) * p.orbitR;
        p.pulse += 0.030;

        /* consumed — pass the event horizon */
        if (p.orbitR <= bh.r * 0.58) {
          bh.consumed++;
          bh.capturedCount = Math.max(0, bh.capturedCount - 1);
          p.capturedBy = null;
          /* respawn comet at a random edge */
          const edge = Math.floor(rand(0, 4));
          if      (edge === 0) { p.x = rand(0, W); p.y = -15; }
          else if (edge === 1) { p.x = rand(0, W); p.y = H + 15; }
          else if (edge === 2) { p.x = -15; p.y = rand(0, H); }
          else                 { p.x = W + 15; p.y = rand(0, H); }
          const ba = rand(0, Math.PI * 2);
          p.vx = Math.cos(ba) * rand(0.15, 0.45);
          p.vy = Math.sin(ba) * rand(0.15, 0.45);
          p.trail = [];
          continue; /* skip drawing this frame */
        }
      }

      /* ── NORMAL: standard physics (skip while orbiting) ── */
      if (!orbiting) {
        p.trail.push({ x: p.x, y: p.y });
        if (p.trail.length > p.trailLen) p.trail.shift();

        /* cursor repulsion */
        if (mouse.active) {
          const cx = p.x - mouse.x, cy = p.y - mouse.y;
          const cd = Math.sqrt(cx * cx + cy * cy) + 0.1;
          if (cd < REPULSE_R) {
            const f = REPULSE_F * Math.pow(1 - cd / REPULSE_R, 1.6);
            p.vx += (cx / cd) * f;  p.vy += (cy / cd) * f;
          }
        }

        /* static gravity wells */
        for (const w of wArr) {
          const ex = w.x - p.x, ey = w.y - p.y;
          const d  = Math.sqrt(ex * ex + ey * ey) + 0.1;
          const f  = (w.mass * 0.007) / (d * 0.014 + 1);
          p.vx += (ex / d) * f;  p.vy += (ey / d) * f;
        }

        /* mobile black hole attraction + capture */
        for (const bh of mobHoles) {
          if (bh.state !== 'active') continue;
          const ex = bh.x - p.x, ey = bh.y - p.y;
          const d  = Math.sqrt(ex * ex + ey * ey) + 0.1;

          /* strong gravity in influence zone — bends particle paths visibly */
          if (d < bh.r * 18) {
            const f = (bh.mass * 0.022) / (d * 0.010 + 1);
            p.vx += (ex / d) * f;  p.vy += (ey / d) * f;
          }

          /* capture into spiral when close enough */
          if (!p.capturedBy && d < bh.r * 2.8 && bh.capturedCount < 7) {
            p.capturedBy = bh;
            p.orbitR     = d;
            p.orbitAngle = Math.atan2(p.y - bh.y, p.x - bh.x);
            /* derive initial angular speed from tangential velocity */
            const nx = ex / d, ny = ey / d;
            const tang = -p.vy * nx + p.vx * ny;
            p.orbitSpd = clamp(tang / Math.max(d, 5), -0.18, 0.18);
            if (Math.abs(p.orbitSpd) < 0.04)
              p.orbitSpd = (Math.random() < 0.5 ? -1 : 1) * rand(0.05, 0.10);
            bh.capturedCount++;
          }
        }

        /* super-photon local distortion */
        for (const sp of superPhotons) {
          const ex = sp.x - p.x, ey = sp.y - p.y;
          const d = Math.sqrt(ex * ex + ey * ey) + 0.1;
          if (d < 110) {
            const f = (0.030 * clamp(sp.lifeSec / 3, 0, 1)) / (d * 0.012 + 1);
            p.vx += (ex / d) * f;
            p.vy += (ey / d) * f;
          }
        }

        /* damping + speed cap */
        p.vx *= DAMP;  p.vy *= DAMP;
        const spd = Math.sqrt(p.vx * p.vx + p.vy * p.vy);
        if (spd > MAX_SPD) { p.vx = (p.vx / spd) * MAX_SPD; p.vy = (p.vy / spd) * MAX_SPD; }

        p.x += p.vx;  p.y += p.vy;
        if (!freezePulse) p.pulse += 0.022;

        /* wrap */
        if (p.x < -25)    { p.x = W + 25; p.trail = []; }
        if (p.x > W + 25) { p.x = -25;    p.trail = []; }
        if (p.y < -25)    { p.y = H + 25; p.trail = []; }
        if (p.y > H + 25) { p.y = -25;    p.trail = []; }
      }

      /* ── draw trail ── */
      const tLen = p.trail.length;
      if (tLen > 1) {
        ctx.lineCap = 'round';
        for (let t = 0; t < tLen - 1; t++) {
          const progress = (t + 1) / tLen;
          const ta = progress * p.alpha * (light ? 0.65 : 0.55);
          const tw = p.r * 1.8 * progress;
          ctx.strokeStyle = rgba(p.color, ta);
          ctx.lineWidth   = Math.max(0.4, tw);
          ctx.beginPath();
          ctx.moveTo(p.trail[t].x,     p.trail[t].y);
          ctx.lineTo(p.trail[t + 1].x, p.trail[t + 1].y);
          ctx.stroke();
        }
      }

      /* ── draw head: glow halo + solid core ── */
      const pulse = 0.55 + 0.45 * Math.sin(p.pulse);
      const headA = p.alpha * pulse;

      const grd = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r * 3.2);
      grd.addColorStop(0,   rgba(p.color, headA * (light ? 0.85 : 1.0)));
      grd.addColorStop(0.4, rgba(p.color, headA * 0.35));
      grd.addColorStop(1,  'transparent');
      ctx.globalAlpha = 1;
      ctx.fillStyle   = grd;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r * 3.2, 0, Math.PI * 2);
      ctx.fill();

      ctx.globalAlpha = headA;
      ctx.fillStyle   = rgba(p.color, 1);
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  };

  const drawSuperPhotons = () => {
    for (const sp of superPhotons) {
      const perspective = 1 + sp.z * 0.24;
      const life = clamp(sp.lifeSec / 3, 0, 1);
      const headR = Math.max(0.28, sp.r * perspective);

      const tLen = sp.trail.length;
      if (tLen > 1) {
        ctx.lineCap = 'round';
        for (let t = 0; t < tLen - 1; t++) {
          const progress = (t + 1) / tLen;
          const tw = Math.max(0.25, headR * 1.7 * progress);
          const ta = progress * 0.55 * life;
          ctx.strokeStyle = `rgba(180, 242, 255, ${ta.toFixed(3)})`;
          ctx.lineWidth = tw;
          ctx.beginPath();
          ctx.moveTo(sp.trail[t].x, sp.trail[t].y);
          ctx.lineTo(sp.trail[t + 1].x, sp.trail[t + 1].y);
          ctx.stroke();
        }
      }

      const halo = ctx.createRadialGradient(sp.x, sp.y, 0, sp.x, sp.y, headR * 7);
      halo.addColorStop(0, `rgba(220, 250, 255, ${(0.95 * life).toFixed(3)})`);
      halo.addColorStop(0.25, `rgba(0, 229, 255, ${(0.60 * life).toFixed(3)})`);
      halo.addColorStop(0.58, `rgba(0, 212, 170, ${(0.24 * life).toFixed(3)})`);
      halo.addColorStop(1, 'transparent');
      ctx.fillStyle = halo;
      ctx.beginPath();
      ctx.arc(sp.x, sp.y, headR * 7, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = `rgba(255,255,255, ${(0.95 * life).toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(sp.x, sp.y, headR, 0, Math.PI * 2);
      ctx.fill();
    }
  };

  const drawIntro = () => {
    ctx.fillStyle = '#00010a';
    ctx.fillRect(0, 0, W, H);

    /* slight camera-shake around detonation peak */
    const shakeRise = clamp((intro.timerSec - 1.08) / 0.12, 0, 1);
    const shakeFall = clamp((2.18 - intro.timerSec) / 0.62, 0, 1);
    const shakeT = shakeRise * shakeFall;
    const shakeAmp = 4.2 * shakeT * shakeT;
    const shakeX = (Math.random() * 2 - 1) * shakeAmp + Math.sin(time * 62) * shakeAmp * 0.22;
    const shakeY = (Math.random() * 2 - 1) * shakeAmp + Math.cos(time * 57) * shakeAmp * 0.22;

    ctx.save();
    ctx.translate(shakeX, shakeY);

    const sunT = clamp((intro.timerSec - 0.22) / 0.86, 0, 1);
    if (sunT > 0) {
      const explodeStart = 1.08;
      const postExplodeT = clamp((intro.timerSec - explodeStart) / 1.40, 0, 1);
      const preExplodePulse = intro.timerSec < explodeStart
        ? (0.92 + 0.08 * Math.sin(time * 8.8))
        : 1;
      /* after detonation: core should expand continuously (no breathing) */
      const sunR = Math.min(W, H) * (0.020 + sunT * 0.075 + postExplodeT * 0.18) * preExplodePulse;
      const sunFade = 1 - postExplodeT;

      const corona = ctx.createRadialGradient(
        intro.center.x, intro.center.y, 0,
        intro.center.x, intro.center.y, sunR * 7
      );
      corona.addColorStop(0, `rgba(220,255,245, ${(0.82 * sunT * sunFade).toFixed(3)})`);
      corona.addColorStop(0.20, `rgba(0,229,255, ${(0.68 * sunT * sunFade).toFixed(3)})`);
      corona.addColorStop(0.45, `rgba(0,212,170, ${(0.42 * sunT * sunFade).toFixed(3)})`);
      corona.addColorStop(0.74, `rgba(0,150,130, ${(0.20 * sunT * sunFade).toFixed(3)})`);
      corona.addColorStop(1, 'transparent');
      ctx.fillStyle = corona;
      ctx.beginPath();
      ctx.arc(intro.center.x, intro.center.y, sunR * 7, 0, Math.PI * 2);
      ctx.fill();

      const core = ctx.createRadialGradient(
        intro.center.x, intro.center.y, 0,
        intro.center.x, intro.center.y, sunR
      );
      core.addColorStop(0, `rgba(245,255,255, ${(0.98 * sunFade).toFixed(3)})`);
      core.addColorStop(0.30, `rgba(188,255,246, ${(0.96 * sunFade).toFixed(3)})`);
      core.addColorStop(0.65, `rgba(0,245,210, ${(0.92 * sunFade).toFixed(3)})`);
      core.addColorStop(1, `rgba(0,195,165, ${(0.84 * sunFade).toFixed(3)})`);
      ctx.fillStyle = core;
      ctx.beginPath();
      ctx.arc(intro.center.x, intro.center.y, sunR, 0, Math.PI * 2);
      ctx.fill();
    }

    const bangT = clamp((intro.timerSec - 1.08) / 0.80, 0, 1);
    if (bangT > 0) {
      const ease = Math.pow(bangT, 0.58);
      const maxR = Math.hypot(W, H) * 0.72;
      const coreR = 2 + ease * maxR;

      const flash = ctx.createRadialGradient(
        intro.center.x, intro.center.y, 0,
        intro.center.x, intro.center.y, coreR
      );
      flash.addColorStop(0, `rgba(255,255,255, ${(0.95 - ease * 0.40).toFixed(3)})`);
      flash.addColorStop(0.18, `rgba(185,255,248, ${(0.78 - ease * 0.36).toFixed(3)})`);
      flash.addColorStop(0.50, `rgba(0,229,255, ${(0.48 - ease * 0.30).toFixed(3)})`);
      flash.addColorStop(1, 'transparent');

      ctx.fillStyle = flash;
      ctx.beginPath();
      ctx.arc(intro.center.x, intro.center.y, coreR, 0, Math.PI * 2);
      ctx.fill();

      for (let i = 0; i < 7; i++) {
        const ringT = clamp((bangT - i * 0.14) / (1 - i * 0.14), 0, 1);
        if (ringT <= 0) continue;
        const rr = 8 + ringT * (Math.hypot(W, H) * (0.22 + i * 0.06));
        const ra = (1 - ringT) * (0.52 - i * 0.06);

        const split = 4.4 + i * 0.55 + bangT * 2.8;
        ctx.lineWidth = Math.max(0.35, 1.2 - i * 0.16);

        /* chromatic ring split: white core + cyan outward + green inward */
        ctx.strokeStyle = `rgba(255,255,255, ${(ra * 0.92).toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(intro.center.x, intro.center.y, rr, 0, Math.PI * 2);
        ctx.stroke();

        ctx.strokeStyle = `rgba(0,229,255, ${(ra * 0.78).toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(intro.center.x, intro.center.y, rr + split, 0, Math.PI * 2);
        ctx.stroke();

        ctx.strokeStyle = `rgba(0,212,170, ${(ra * 0.76).toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(intro.center.x, intro.center.y, Math.max(1, rr - split), 0, Math.PI * 2);
        ctx.stroke();
      }
    }

    for (const w of intro.shockwaves) {
      const oscillation = 1 + 0.015 * Math.sin((w.r * 0.045) + time * 10);
      const rr = w.r * oscillation;
      ctx.strokeStyle = `rgba(255,255,255, ${(w.alpha * 0.85).toFixed(3)})`;
      ctx.lineWidth = 2.2;
      ctx.beginPath();
      ctx.arc(intro.center.x, intro.center.y, rr, 0, Math.PI * 2);
      ctx.stroke();

      ctx.strokeStyle = `rgba(0,229,255, ${(w.alpha * 0.72).toFixed(3)})`;
      ctx.lineWidth = 1.25;
      ctx.beginPath();
      ctx.arc(intro.center.x, intro.center.y, rr + 5, 0, Math.PI * 2);
      ctx.stroke();

      ctx.strokeStyle = `rgba(0,212,170, ${(w.alpha * 0.68).toFixed(3)})`;
      ctx.lineWidth = 1.15;
      ctx.beginPath();
      ctx.arc(intro.center.x, intro.center.y, Math.max(1, rr - 5), 0, Math.PI * 2);
      ctx.stroke();
    }

    for (const p of intro.blastPhotons) {
      const life = clamp(p.life / 1.3, 0, 1);
      const phase = 0.6 + 0.4 * Math.sin(p.pulse);
      const headA = life * phase;

      if (p.trail.length > 1) {
        for (let t = 0; t < p.trail.length - 1; t++) {
          const progress = (t + 1) / p.trail.length;
          const ta = progress * life * 0.70;
          ctx.strokeStyle = `rgba(${p.color[0]},${p.color[1]},${p.color[2]},${ta.toFixed(3)})`;
          ctx.lineWidth = Math.max(0.25, p.r * 1.5 * progress);
          ctx.beginPath();
          ctx.moveTo(p.trail[t].x, p.trail[t].y);
          ctx.lineTo(p.trail[t + 1].x, p.trail[t + 1].y);
          ctx.stroke();
        }
      }

      const halo = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r * 5.4);
      halo.addColorStop(0, `rgba(255,255,255, ${(headA * 0.9).toFixed(3)})`);
      halo.addColorStop(0.30, `rgba(${p.color[0]},${p.color[1]},${p.color[2]}, ${(headA * 0.66).toFixed(3)})`);
      halo.addColorStop(0.66, `rgba(${p.color[0]},${p.color[1]},${p.color[2]}, ${(headA * 0.32).toFixed(3)})`);
      halo.addColorStop(1, 'transparent');
      ctx.fillStyle = halo;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r * 5.4, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = `rgba(240,255,255, ${(headA * 0.95).toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
    }

    const flowTime = intro.timerSec - 1.88;
    if (flowTime <= 0) {
      ctx.restore();
      return;
    }

    for (const p of intro.photons) {
      if (flowTime < p.delaySec || p.life <= 0) continue;

      const phase = 0.6 + 0.4 * Math.sin(p.pulse);
      const headA = p.life * phase;

      if (p.trail.length > 1) {
        for (let t = 0; t < p.trail.length - 1; t++) {
          const progress = (t + 1) / p.trail.length;
          const ta = progress * p.life * 0.55;
          ctx.strokeStyle = `rgba(${p.color[0]},${p.color[1]},${p.color[2]}, ${ta.toFixed(3)})`;
          ctx.lineWidth = Math.max(0.2, p.r * 1.3 * progress);
          ctx.beginPath();
          ctx.moveTo(p.trail[t].x, p.trail[t].y);
          ctx.lineTo(p.trail[t + 1].x, p.trail[t + 1].y);
          ctx.stroke();
        }
      }

      const halo = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, p.r * 5);
      halo.addColorStop(0, `rgba(255,255,255, ${(headA * 0.95).toFixed(3)})`);
      halo.addColorStop(0.28, `rgba(${p.color[0]},${p.color[1]},${p.color[2]}, ${(headA * 0.60).toFixed(3)})`);
      halo.addColorStop(0.65, `rgba(${p.color[0]},${p.color[1]},${p.color[2]}, ${(headA * 0.25).toFixed(3)})`);
      halo.addColorStop(1, 'transparent');
      ctx.fillStyle = halo;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r * 5, 0, Math.PI * 2);
      ctx.fill();

      ctx.fillStyle = `rgba(255,255,255, ${(headA * 0.9).toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore();
  };

  /* ── main render loop ────────────────────────────────────────────── */
  let lastVisible = true;
  let lastTs = performance.now();

  const draw = () => {
    const now = performance.now();
    const dtSec = clamp((now - lastTs) / 1000, 1 / 120, 0.05);
    lastTs = now;

    time += dtSec;
    ctx.clearRect(0, 0, W, H);

    if (intro.enabled) {
      updateIntro(dtSec);
      if (intro.enabled) {
        drawIntro();
        raf = requestAnimationFrame(draw);
        return;
      }
    }

    burstTimerSec -= dtSec;
    if (burstTimerSec <= 0) {
      emitPhotonBurst();
      burstTimerSec = 60;
    }
    if (burstActiveSec > 0) burstActiveSec -= dtSec;

    updateSuperPhotons(dtSec);
    updateMobHoles(dtSec);     /* advance black-hole positions before grid uses them */

    drawGalaxy();
    drawNebulae();
    drawStars();
    drawGrid();
    drawGravityWells();
    drawMobileBlackHoles();
    drawComets();
    drawSuperPhotons();

    raf = requestAnimationFrame(draw);
  };

  /* ── init ─────────────────────────────────────────────────────────── */
  window.addEventListener('resize', resize, { passive: true });

  /* ── mouse tracking (full-page, since canvas is fixed) ──────────── */
  document.addEventListener('mousemove', e => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
    mouse.active = true;
  }, { passive: true });
  document.addEventListener('mouseleave', () => { mouse.active = false; });
  /* touch support */
  document.addEventListener('touchmove', e => {
    if (e.touches.length > 0) {
      mouse.x = e.touches[0].clientX;
      mouse.y = e.touches[0].clientY;
      mouse.active = true;
    }
  }, { passive: true });
  document.addEventListener('touchend', () => { mouse.active = false; }, { passive: true });

  document.addEventListener('click', e => {
    if (intro.enabled) return;

    const x = e.clientX;
    const y = e.clientY;

    /* click black hole -> immediate explosion */
    let target = null;
    let nearest = Infinity;
    for (const bh of mobHoles) {
      if (bh.state !== 'active') continue;
      const d = Math.hypot(x - bh.x, y - bh.y);
      const hitR = bh.r * 2.2;
      if (d <= hitR && d < nearest) {
        nearest = d;
        target = bh;
      }
    }
    if (target) triggerHoleExplosion(target);

    /* each 3 clicks -> photon burst + reset 60s timer */
    burstClickCount++;
    if (burstClickCount >= 3) {
      emitPhotonBurst({ x, y });
      burstClickCount = 0;
      burstTimerSec = 60;
    }
  }, { passive: true });

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      cancelAnimationFrame(raf);
      lastVisible = false;
    } else if (!lastVisible) {
      lastVisible = true;
      lastTs = performance.now();
      draw();
    }
  });

  /* Re-init on MkDocs theme toggle */
  new MutationObserver(() => buildScene())
    .observe(document.documentElement, { attributes: true, attributeFilter: ['data-md-color-scheme'] });
  new MutationObserver(() => buildScene())
    .observe(document.body, { attributes: true, attributeFilter: ['data-md-color-scheme'] });

  intro.enabled = shouldRunIntro();
  setIntroPageHidden(intro.enabled);

  resize();
  draw();

  /* ── scroll reveal ───────────────────────────────────────────────── */
  if (typeof IntersectionObserver !== 'undefined') {
    const io = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          e.target.classList.add('gn-revealed');
          io.unobserve(e.target);
        }
      });
    }, { threshold: 0.12 });
    homeRoot.querySelectorAll('.gn-reveal').forEach(el => io.observe(el));
  }

  /* ── hero parallax ───────────────────────────────────────────────── */
  const heroBg = homeRoot.querySelector('.gn-hero-bg');
  if (heroBg) {
    const onScroll = () => {
      const y = clamp(window.scrollY * 0.15, 0, 80);
      heroBg.style.transform = `translateY(${y}px)`;
    };
    onScroll();
    window.addEventListener('scroll', onScroll, { passive: true });
  }

  /* open docs links in new tab */
  document.querySelectorAll('a[href^="https://docs.machinegnostics.com"]').forEach(link => {
    link.target = '_blank';
    link.rel    = 'noopener noreferrer';
  });
});
