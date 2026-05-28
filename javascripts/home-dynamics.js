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
  const roundedRectPath = (c, x, y, w, h, r) => {
    const rr = Math.max(0, Math.min(r, Math.min(w, h) * 0.5));
    c.beginPath();
    c.moveTo(x + rr, y);
    c.lineTo(x + w - rr, y);
    c.quadraticCurveTo(x + w, y, x + w, y + rr);
    c.lineTo(x + w, y + h - rr);
    c.quadraticCurveTo(x + w, y + h, x + w - rr, y + h);
    c.lineTo(x + rr, y + h);
    c.quadraticCurveTo(x, y + h, x, y + h - rr);
    c.lineTo(x, y + rr);
    c.quadraticCurveTo(x, y, x + rr, y);
    c.closePath();
  };

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
  let astronautSuppressClick = false;
  let starStickers = [];
  let brandMarks = [];
  const scrollReact = {
    lastY: window.scrollY || 0,
    lastTs: performance.now(),
    cooldownSec: 0,
  };

  /* tiny fun astronaut */
  let astronaut = null;
  const ASTRONAUT_IDLE_PHRASES = [
    'You are awesome!',
    'Keep exploring!',
    'Great work, human!',
    'You got this!',
    'Brilliant mind detected!',
    'Space says hi!',
    'Curiosity level: legendary.',
    'Your ideas have escape velocity.',
    'Tiny astronaut approves this mission.',
    'Science mode looks good on you.',
    'You make this galaxy brighter.',
    'Stellar focus, captain!',
    'Breathe in, brilliance out.',
    'Your future self says thanks.',
    'I see genius in this orbit.',
    'Entropy says hi, order says wow.',
  ];
  const ASTRONAUT_CLICK_PHRASES = [
    'Can I help you?',
    'What up?',
    'Write me email, I am busy now!',
    'Mission support online. Need anything?',
    'Beep boop. Emotional support astronaut here.',
    'I wave, therefore I am.',
    'Status report: you are doing great.',
    'Coffee? I only have stardust.',
    'You clicked me. I feel important.',
    'Need ideas? I have a whole nebula.',
    'I can totally keep a secret. Probably.',
    'We got this, commander!',
    'Tell me your next bold move.',
    'Plot twist: you are awesome.',
    'My schedule is chaos, but for you I have 8 seconds.',
  ];
  const ASTRONAUT_EXPLOSION_PHRASES = [
    'Whoa! That black hole had drama.',
    'Kaboom confirmed. I meant to do that.',
    'That was scientifically spicy.',
    'Reminder: do not poke singularities.',
    'Explosion rating: 11 out of 10.',
    'Okay... that got personal.',
    'Black hole said goodbye very loudly.',
    'I call that a gravity mic drop.',
  ];
  const ASTRONAUT_STICKER_PHRASES = [
    'Stellar catch! +1 cosmic point.',
    'Nice grab. You are pure stardust energy.',
    'Keep collecting tiny wonders.',
    'You found a lucky star sticker.',
  ];
  const ASTRONAUT_BRAND_PHRASES = [
    'Machine Gnostics signal stamped in local space.',
    'Brand beacon online: Machine Gnostics.',
    'Orbit tag: MACHINE GNOSTICS locked and glowing.',
    'Friendly mark deployed. Machine Gnostics says hi.',
    'MG signature dropped into this star lane.',
  ];
  const ASTRONAUT_FRIDAY_SPECIAL = 'WEEKEND ORBIT: Friday detected. Keep the vibes in stable orbit.';
  const ASTRONAUT_TAGS = ['COMMS', 'EVA LOG', 'MISSION TIP', 'ORBIT NOTE'];
  const ASTRONAUT_MOODS = ['chill', 'excited', 'curious', 'sleepy'];
  const ASTRONAUT_GESTURES = ['wave', 'thumbs', 'salute', 'visor', 'spin'];

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

    initAstronaut();

    if (intro.enabled) {
      initIntro();
    }
  };

  const initAstronaut = () => {
    const size = clamp(Math.min(W, H) * 0.013, 8.4, 14.2);

    if (!astronaut) {
      const px = rand(0, 1) < 0.5 ? rand(W * 0.05, W * 0.22) : rand(W * 0.78, W * 0.95);
      astronaut = {
        x: px,
        y: rand(H * 0.16, H * 0.86),
        vx: rand(-0.24, 0.24),
        vy: rand(-0.18, 0.18),
        targetVx: rand(-0.24, 0.24),
        targetVy: rand(-0.18, 0.18),
        targetX: px,
        targetY: rand(H * 0.16, H * 0.86),
        driftTimerSec: rand(2.2, 5.5),
        bobPhase: rand(0, Math.PI * 2),
        wavePhase: rand(0, Math.PI * 2),
        spinPhase: 0,
        moodBoostSec: 0,
        mood: 'chill',
        moodTimerSec: rand(6.0, 12.0),
        gesture: 'wave',
        gestureTimerSec: rand(3.0, 6.5),
        brandTimerSec: rand(14.0, 30.0),
        hover: false,
        dragging: false,
        dragOffsetX: 0,
        dragOffsetY: 0,
        dragMoved: false,
        lastDragX: px,
        lastDragY: rand(H * 0.16, H * 0.86),
        lastDragTs: performance.now(),
        size,
        suitDrift: rand(0, Math.PI * 2),
        bubble: {
          text: '',
          timerSec: 0,
          cooldownSec: rand(3.5, 7.5),
          alpha: 0,
        },
      };
      return;
    }

    astronaut.size = size;
    astronaut.x = clamp(astronaut.x, 28, W - 28);
    astronaut.y = clamp(astronaut.y, 28, H - 28);
  };

  const pickAstronautTarget = () => {
    const margin = Math.max(26, astronaut ? astronaut.size * 2.2 : 28);
    const side = Math.floor(rand(0, 4));
    const innerL = W * 0.24;
    const innerR = W * 0.76;
    const innerT = H * 0.20;
    const innerB = H * 0.80;
    if (side === 0) return { x: rand(margin, innerL), y: rand(margin, H - margin) };      /* left band */
    if (side === 1) return { x: rand(innerR, W - margin), y: rand(margin, H - margin) };   /* right band */
    if (side === 2) return { x: rand(margin, W - margin), y: rand(margin, innerT) };        /* top band */
    return { x: rand(margin, W - margin), y: rand(innerB, H - margin) };                     /* bottom band */
  };

  const astronautInCenterZone = (ax, ay) => {
    return ax > W * 0.25 && ax < W * 0.75 && ay > H * 0.22 && ay < H * 0.80;
  };

  const setAstronautMood = (mood, durationSec = rand(5.0, 10.0)) => {
    if (!astronaut) return;
    astronaut.mood = mood;
    astronaut.moodTimerSec = durationSec;
  };

  const pickAstronautGesture = () => {
    if (!astronaut) return;
    const g = ASTRONAUT_GESTURES[Math.floor(Math.random() * ASTRONAUT_GESTURES.length)];
    astronaut.gesture = g;
    astronaut.gestureTimerSec = g === 'spin' ? rand(1.6, 2.2) : rand(2.6, 5.8);
  };

  const formatCommsLine = (message, category = 'auto') => {
    let tag;
    if (category === 'click') tag = 'COMMS';
    else if (category === 'event') tag = 'ORBIT NOTE';
    else if (category === 'explosion') tag = 'EVA LOG';
    else if (category === 'sticker') tag = 'MISSION TIP';
    else if (category === 'brand') tag = 'BRAND SIGNAL';
    else tag = ASTRONAUT_TAGS[Math.floor(Math.random() * ASTRONAUT_TAGS.length)];
    return `${tag}: ${message}`;
  };

  const beginAstronautDrag = (px, py) => {
    if (!astronaut || !astronautHit(px, py)) return false;
    astronaut.dragging = true;
    astronaut.dragOffsetX = astronaut.x - px;
    astronaut.dragOffsetY = astronaut.y - py;
    astronaut.dragMoved = false;
    astronaut.lastDragX = px;
    astronaut.lastDragY = py;
    astronaut.lastDragTs = performance.now();
    astronaut.hover = true;
    astronaut.moodBoostSec = 4;
    return true;
  };

  const moveAstronautDrag = (px, py) => {
    if (!astronaut || !astronaut.dragging) return;
    const margin = Math.max(20, astronaut.size * 1.6);
    const nx = clamp(px + astronaut.dragOffsetX, margin, W - margin);
    const ny = clamp(py + astronaut.dragOffsetY, margin, H - margin);

    if (Math.hypot(nx - astronaut.x, ny - astronaut.y) > 1.1) astronaut.dragMoved = true;

    const now = performance.now();
    const dt = Math.max(8, now - astronaut.lastDragTs);
    astronaut.vx = clamp((nx - astronaut.x) / dt * 7.8, -1.5, 1.5);
    astronaut.vy = clamp((ny - astronaut.y) / dt * 7.8, -1.5, 1.5);
    astronaut.lastDragX = px;
    astronaut.lastDragY = py;
    astronaut.lastDragTs = now;

    astronaut.x = nx;
    astronaut.y = ny;
  };

  const endAstronautDrag = () => {
    if (!astronaut || !astronaut.dragging) return;
    astronaut.dragging = false;
    astronaut.driftTimerSec = rand(1.2, 2.6);
    if (astronaut.dragMoved) astronautSuppressClick = true;
  };

  const triggerAstronautBubble = (forcedText = null, category = 'auto') => {
    if (!astronaut) return;
    let text = forcedText;
    if (!text) {
      if (new Date().getDay() === 5 && Math.random() < 0.28) {
        text = ASTRONAUT_FRIDAY_SPECIAL;
      } else {
        text = ASTRONAUT_IDLE_PHRASES[Math.floor(Math.random() * ASTRONAUT_IDLE_PHRASES.length)];
      }
    }
    text = formatCommsLine(text, category);
    astronaut.bubble.text = text;
    astronaut.bubble.timerSec = rand(2.2, 3.4);
    astronaut.bubble.cooldownSec = rand(5.0, 11.5);
  };

  const maybeSpawnStarSticker = (dtSec) => {
    if (!astronaut || astronaut.dragging) return;
    if (starStickers.length >= 2) return;
    const chance = dtSec * 0.018; /* very rare */
    if (Math.random() >= chance) return;

    starStickers.push({
      x: astronaut.x + rand(-astronaut.size * 1.4, astronaut.size * 1.4),
      y: astronaut.y + rand(-astronaut.size * 1.2, astronaut.size * 1.2),
      vx: rand(-0.12, 0.12),
      vy: rand(-0.08, 0.06),
      r: rand(3.6, 5.2),
      lifeSec: rand(8.0, 13.0),
      maxLife: 13.0,
      phase: rand(0, Math.PI * 2),
    });
  };

  const spawnBrandMark = () => {
    if (!astronaut) return;
    if (brandMarks.length >= 2) return;
    brandMarks.push({
      x: astronaut.x + rand(-astronaut.size * 2.1, astronaut.size * 2.1),
      y: astronaut.y + rand(-astronaut.size * 1.9, astronaut.size * 0.8),
      r: rand(16, 26),
      lifeSec: rand(3.2, 5.2),
      maxLife: 5.2,
      phase: rand(0, Math.PI * 2),
      spin: rand(-0.6, 0.6),
    });
  };

  const updateBrandMarks = (dtSec) => {
    for (let i = brandMarks.length - 1; i >= 0; i--) {
      const bm = brandMarks[i];
      bm.lifeSec -= dtSec;
      bm.phase += dtSec * (1.8 + bm.spin);
      bm.y -= dtSec * 3.8;
      if (bm.lifeSec <= 0) brandMarks.splice(i, 1);
    }
  };

  const drawBrandMarks = () => {
    const light = isLight();
    for (const bm of brandMarks) {
      const life = clamp(bm.lifeSec / bm.maxLife, 0, 1);
      const pulse = 0.55 + 0.45 * Math.sin(time * 5 + bm.phase);
      const a = life * pulse;

      ctx.save();
      ctx.translate(bm.x, bm.y);
      ctx.rotate(Math.sin(bm.phase) * 0.14);

      ctx.strokeStyle = light
        ? `rgba(0,126,152, ${(0.52 * a).toFixed(3)})`
        : `rgba(110,236,224, ${(0.62 * a).toFixed(3)})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(0, 0, bm.r, 0, Math.PI * 2);
      ctx.stroke();

      ctx.strokeStyle = light
        ? `rgba(255,154,56, ${(0.48 * a).toFixed(3)})`
        : `rgba(255,184,96, ${(0.54 * a).toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(0, 0, bm.r * 0.68, 0.12 * Math.PI, 1.52 * Math.PI);
      ctx.stroke();

      ctx.fillStyle = light
        ? `rgba(0,98,122, ${(0.92 * a).toFixed(3)})`
        : `rgba(205,250,245, ${(0.95 * a).toFixed(3)})`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.font = '700 9px Space Mono, monospace';
      ctx.fillText('MG', 0, -1);

      ctx.font = '700 6px Space Mono, monospace';
      ctx.fillText('MACHINE GNOSTICS', 0, bm.r + 8);
      ctx.restore();
    }
  };

  const updateStarStickers = (dtSec) => {
    for (let i = starStickers.length - 1; i >= 0; i--) {
      const st = starStickers[i];
      st.lifeSec -= dtSec;
      st.phase += dtSec * 2.2;
      st.x += st.vx;
      st.y += st.vy;
      st.vy -= 0.002;
      if (st.lifeSec <= 0 || st.x < -18 || st.x > W + 18 || st.y < -18 || st.y > H + 18) {
        starStickers.splice(i, 1);
      }
    }
  };

  const hitStarSticker = (x, y) => {
    for (let i = starStickers.length - 1; i >= 0; i--) {
      const st = starStickers[i];
      if (Math.hypot(x - st.x, y - st.y) <= st.r * 1.8) return i;
    }
    return -1;
  };

  const astronautHit = (x, y) => {
    if (!astronaut) return false;
    const hitR = astronaut.size * 2.7;
    return Math.hypot(x - astronaut.x, y - astronaut.y) <= hitR;
  };

  const updateAstronaut = (dtSec) => {
    if (!astronaut) return;

    const frameScale = clamp(dtSec * 60, 0.45, 2.2);
    astronaut.moodTimerSec -= dtSec;
    astronaut.gestureTimerSec -= dtSec;

    if (astronaut.moodTimerSec <= 0) {
      setAstronautMood(ASTRONAUT_MOODS[Math.floor(Math.random() * ASTRONAUT_MOODS.length)]);
    }
    if (astronaut.gestureTimerSec <= 0) {
      pickAstronautGesture();
    }

    if (astronaut.hover || astronaut.dragging) {
      setAstronautMood('excited', 2.2);
    }

    if (!astronaut.dragging) {
      astronaut.driftTimerSec -= dtSec;
      const reachTarget = Math.hypot(astronaut.x - astronaut.targetX, astronaut.y - astronaut.targetY) < Math.max(18, astronaut.size * 2.2);
      if (astronaut.driftTimerSec <= 0 || reachTarget || astronautInCenterZone(astronaut.x, astronaut.y)) {
        const target = pickAstronautTarget();
        astronaut.targetX = target.x;
        astronaut.targetY = target.y;
        astronaut.driftTimerSec = rand(1.4, 3.6);
      }

      const dx = astronaut.targetX - astronaut.x;
      const dy = astronaut.targetY - astronaut.y;
      const d = Math.hypot(dx, dy) + 0.001;
      let speedMood = 1.0;
      if (astronaut.mood === 'sleepy') speedMood = 0.72;
      else if (astronaut.mood === 'curious') speedMood = 1.10;
      else if (astronaut.mood === 'excited') speedMood = 1.22;
      const desiredVx = (dx / d) * clamp(d / 180, 0.09, 0.42) * speedMood;
      const desiredVy = (dy / d) * clamp(d / 180, 0.07, 0.34) * speedMood;

      astronaut.targetVx = desiredVx;
      astronaut.targetVy = desiredVy;
      astronaut.vx = lerp(astronaut.vx, astronaut.targetVx, 0.032 * frameScale);
      astronaut.vy = lerp(astronaut.vy, astronaut.targetVy, 0.032 * frameScale);

      /* center-avoidance push keeps idle motion near page periphery */
      if (astronautInCenterZone(astronaut.x, astronaut.y)) {
        const cx = W * 0.5;
        const cy = H * 0.5;
        const ex = astronaut.x - cx;
        const ey = astronaut.y - cy;
        const ed = Math.hypot(ex, ey) + 0.001;
        astronaut.vx += (ex / ed) * 0.06 * frameScale;
        astronaut.vy += (ey / ed) * 0.06 * frameScale;
      }

      astronaut.x += astronaut.vx * frameScale;
      astronaut.y += astronaut.vy * frameScale;
    }

    const margin = Math.max(20, astronaut.size * 1.6);
    astronaut.x = clamp(astronaut.x, margin, W - margin);
    astronaut.y = clamp(astronaut.y, margin, H - margin);
    astronaut.bobPhase += 0.014 * frameScale;
    let waveSpd = astronaut.hover || astronaut.dragging ? 0.20 : 0.11;
    if (astronaut.mood === 'excited') waveSpd *= 1.22;
    else if (astronaut.mood === 'sleepy') waveSpd *= 0.74;
    astronaut.wavePhase += waveSpd * frameScale;
    astronaut.spinPhase += (astronaut.gesture === 'spin' ? 0.18 : 0.03) * frameScale;
    astronaut.suitDrift += 0.012 * frameScale;

    astronaut.bubble.cooldownSec -= dtSec;
    if (astronaut.bubble.timerSec > 0) {
      astronaut.bubble.timerSec -= dtSec;
    } else if (astronaut.bubble.cooldownSec <= 0) {
      triggerAstronautBubble();
    }

    const bubbleT = astronaut.bubble.timerSec;
    if (bubbleT > 0) {
      const inA = clamp((3.4 - bubbleT) / 0.25, 0, 1);
      const outA = clamp(bubbleT / 0.45, 0, 1);
      astronaut.bubble.alpha = Math.min(inA, outA);
    } else {
      astronaut.bubble.alpha = 0;
    }

    if (astronaut.moodBoostSec > 0) astronaut.moodBoostSec -= dtSec;

    astronaut.brandTimerSec -= dtSec;
    if (astronaut.brandTimerSec <= 0 && !astronaut.dragging) {
      const msg = ASTRONAUT_BRAND_PHRASES[Math.floor(Math.random() * ASTRONAUT_BRAND_PHRASES.length)];
      triggerAstronautBubble(msg, 'brand');
      spawnBrandMark();
      astronaut.brandTimerSec = rand(18.0, 34.0);
    }

    maybeSpawnStarSticker(dtSec);
    updateStarStickers(dtSec);
    updateBrandMarks(dtSec);
  };

  const drawStarStickers = () => {
    const light = isLight();
    for (const st of starStickers) {
      const life = clamp(st.lifeSec / st.maxLife, 0, 1);
      const tw = 0.45 + 0.55 * Math.sin(time * 8 + st.phase);
      const a = life * tw;
      const col = light
        ? `rgba(0,128,150, ${(0.72 * a).toFixed(3)})`
        : `rgba(154,238,255, ${(0.88 * a).toFixed(3)})`;
      const core = light
        ? `rgba(255,194,92, ${(0.85 * a).toFixed(3)})`
        : `rgba(255,218,126, ${(0.92 * a).toFixed(3)})`;

      ctx.strokeStyle = col;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(st.x - st.r, st.y);
      ctx.lineTo(st.x + st.r, st.y);
      ctx.moveTo(st.x, st.y - st.r);
      ctx.lineTo(st.x, st.y + st.r);
      ctx.stroke();

      ctx.fillStyle = core;
      ctx.beginPath();
      ctx.arc(st.x, st.y, st.r * 0.42, 0, Math.PI * 2);
      ctx.fill();
    }
  };

  const drawAstronaut = () => {
    if (!astronaut) return;

    const x = astronaut.x;
    const y = astronaut.y + Math.sin(astronaut.bobPhase) * astronaut.size * 0.32;
    const s = astronaut.size;
    const light = isLight();
    const hour = new Date().getHours();
    const isNight = hour < 6 || hour >= 19;
    const mood = astronaut.moodBoostSec > 0 ? clamp(astronaut.moodBoostSec / 4, 0, 1) : 0;
    const gesture = astronaut.gesture || 'wave';

    const glowA = (light ? 0.06 : 0.13) + (astronaut.hover ? 0.06 : 0) + mood * 0.08;
    const glow = ctx.createRadialGradient(x, y, 0, x, y, s * 4.6);
    glow.addColorStop(0, light ? `rgba(0,145,165,${(glowA * 1.0).toFixed(3)})` : `rgba(0,229,255,${(glowA * 1.0).toFixed(3)})`);
    glow.addColorStop(0.42, light ? `rgba(0,132,152,${(glowA * 0.45).toFixed(3)})` : `rgba(0,212,170,${(glowA * 0.50).toFixed(3)})`);
    glow.addColorStop(1, 'transparent');
    ctx.fillStyle = glow;
    ctx.beginPath();
    ctx.arc(x, y, s * 4.6, 0, Math.PI * 2);
    ctx.fill();

    ctx.save();
    ctx.translate(x, y);
    const baseRot = Math.sin(astronaut.bobPhase * 0.8) * 0.06;
    const spinRot = gesture === 'spin' ? Math.sin(astronaut.spinPhase) * 0.22 : 0;
    ctx.rotate(baseRot + spinRot);

    /* detached suit piece drifting in space */
    const sx = s * (1.95 + Math.sin(astronaut.suitDrift) * 0.28);
    const sy = s * (-0.58 + Math.cos(astronaut.suitDrift * 0.9) * 0.20);
    ctx.save();
    ctx.translate(sx, sy);
    ctx.rotate(Math.sin(astronaut.suitDrift) * 0.38);
    const pieceGrad = ctx.createLinearGradient(-s * 0.22, -s * 0.18, s * 0.22, s * 0.18);
    pieceGrad.addColorStop(0, light ? 'rgba(170,190,202,0.88)' : 'rgba(176,210,224,0.88)');
    pieceGrad.addColorStop(1, light ? 'rgba(108,132,146,0.88)' : 'rgba(118,150,170,0.88)');
    ctx.fillStyle = pieceGrad;
    roundedRectPath(ctx, -s * 0.24, -s * 0.18, s * 0.50, s * 0.36, 2.2);
    ctx.fill();
    ctx.strokeStyle = light ? 'rgba(0,110,132,0.34)' : 'rgba(120,220,245,0.48)';
    ctx.lineWidth = 0.8;
    ctx.stroke();
    ctx.restore();

    /* tether strand */
    ctx.strokeStyle = light ? 'rgba(128,146,158,0.36)' : 'rgba(176,196,212,0.36)';
    ctx.setLineDash([2.0, 2.2]);
    ctx.lineWidth = 0.7;
    ctx.beginPath();
    ctx.moveTo(s * 0.70, -s * 0.08);
    ctx.quadraticCurveTo(s * 1.10, -s * 0.24, sx - s * 0.15, sy);
    ctx.stroke();
    ctx.setLineDash([]);

    /* backpack */
    const bagGrad = ctx.createLinearGradient(-s * 0.50, -s * 0.15, -s * 0.05, s * 0.75);
    bagGrad.addColorStop(0, light ? 'rgba(168,186,198,0.96)' : 'rgba(170,196,214,0.96)');
    bagGrad.addColorStop(1, light ? 'rgba(118,140,154,0.96)' : 'rgba(120,148,166,0.96)');
    ctx.fillStyle = bagGrad;
    roundedRectPath(ctx, -s * 0.54, -s * 0.12, s * 0.38, s * 0.86, 2.8);
    ctx.fill();

    /* torso */
    const torsoGrad = ctx.createLinearGradient(-s * 0.44, -s * 0.06, s * 0.36, s * 1.20);
    torsoGrad.addColorStop(0, light ? 'rgba(245,250,252,0.97)' : 'rgba(240,248,252,0.96)');
    torsoGrad.addColorStop(0.52, light ? 'rgba(218,230,236,0.97)' : 'rgba(210,224,232,0.96)');
    torsoGrad.addColorStop(1, light ? 'rgba(188,205,214,0.97)' : 'rgba(182,198,210,0.96)');
    ctx.fillStyle = torsoGrad;
    ctx.beginPath();
    ctx.ellipse(-s * 0.02, s * 0.36, s * 0.52, s * 0.88, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = light ? 'rgba(0,136,156,0.72)' : 'rgba(0,230,214,0.80)';
    ctx.lineWidth = 1.0;
    ctx.stroke();

    /* high-visibility suit bands */
    ctx.fillStyle = light ? 'rgba(255,136,38,0.88)' : 'rgba(255,156,64,0.90)';
    roundedRectPath(ctx, -s * 0.34, s * 0.08, s * 0.62, s * 0.08, 1.6);
    ctx.fill();
    roundedRectPath(ctx, -s * 0.20, s * 0.78, s * 0.36, s * 0.07, 1.4);
    ctx.fill();

    /* chest details */
    ctx.fillStyle = light ? 'rgba(136,156,170,0.72)' : 'rgba(144,166,182,0.78)';
    roundedRectPath(ctx, -s * 0.20, s * 0.26, s * 0.26, s * 0.18, 2.2);
    ctx.fill();
    ctx.fillStyle = light ? 'rgba(0,128,92,0.75)' : 'rgba(0,210,140,0.82)';
    ctx.beginPath();
    ctx.arc(-s * 0.10, s * 0.35, s * 0.03, 0, Math.PI * 2);
    ctx.fill();

    /* helmet shell */
    const helmetShell = ctx.createRadialGradient(-s * 0.06, -s * 0.50, s * 0.05, 0, -s * 0.36, s * 0.64);
    helmetShell.addColorStop(0, light ? 'rgba(245,252,255,0.98)' : 'rgba(236,248,255,0.96)');
    helmetShell.addColorStop(1, light ? 'rgba(188,206,220,0.96)' : 'rgba(176,198,214,0.96)');
    ctx.fillStyle = helmetShell;
    ctx.beginPath();
    ctx.arc(0, -s * 0.38, s * 0.60, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = light ? 'rgba(0,128,150,0.64)' : 'rgba(0,226,214,0.72)';
    ctx.lineWidth = 0.9;
    ctx.stroke();

    ctx.strokeStyle = light ? 'rgba(255,146,46,0.72)' : 'rgba(255,170,84,0.76)';
    ctx.lineWidth = 1.0;
    ctx.beginPath();
    ctx.arc(0, -s * 0.38, s * 0.60, 0.12 * Math.PI, 0.88 * Math.PI);
    ctx.stroke();

    /* visor */
    const visorGrad = ctx.createRadialGradient(-s * 0.12, -s * 0.52, s * 0.02, 0, -s * 0.40, s * 0.38);
    if (isNight) {
      visorGrad.addColorStop(0, light ? 'rgba(132,190,222,0.84)' : 'rgba(136,206,238,0.84)');
      visorGrad.addColorStop(0.58, light ? 'rgba(52,98,132,0.88)' : 'rgba(32,84,118,0.88)');
      visorGrad.addColorStop(1, light ? 'rgba(20,60,92,0.90)' : 'rgba(14,52,84,0.92)');
    } else {
      visorGrad.addColorStop(0, light ? 'rgba(198,156,104,0.84)' : 'rgba(220,170,118,0.82)');
      visorGrad.addColorStop(0.58, light ? 'rgba(120,88,54,0.88)' : 'rgba(112,78,50,0.88)');
      visorGrad.addColorStop(1, light ? 'rgba(72,48,32,0.90)' : 'rgba(62,42,28,0.92)');
    }
    ctx.fillStyle = visorGrad;
    ctx.beginPath();
    ctx.arc(0, -s * 0.40, s * 0.40, 0, Math.PI * 2);
    ctx.fill();

    /* visor reflections */
    ctx.fillStyle = 'rgba(220,248,255,0.42)';
    ctx.beginPath();
    ctx.ellipse(-s * 0.16, -s * 0.52, s * 0.11, s * 0.05, -0.48, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = 'rgba(190,236,248,0.22)';
    ctx.beginPath();
    ctx.ellipse(s * 0.02, -s * 0.46, s * 0.07, s * 0.03, -0.25, 0, Math.PI * 2);
    ctx.fill();

    /* subtle happy cue inside visor */
    ctx.strokeStyle = 'rgba(212,245,255,0.38)';
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    ctx.arc(0, -s * 0.30, s * 0.09, 0.25, Math.PI - 0.25);
    ctx.stroke();

    /* gesture-driven arms */
    ctx.strokeStyle = light ? 'rgba(220,234,242,0.92)' : 'rgba(216,232,242,0.94)';
    ctx.lineWidth = 1.7;
    ctx.lineCap = 'round';

    /* left arm baseline */
    ctx.beginPath();
    ctx.moveTo(-s * 0.22, s * 0.04);
    ctx.lineTo(-s * 0.52, gesture === 'salute' ? s * 0.08 : -s * 0.02);
    ctx.lineTo(-s * 0.72, gesture === 'salute' ? s * 0.20 : s * 0.06);
    ctx.stroke();

    if (gesture === 'thumbs') {
      ctx.save();
      ctx.translate(s * 0.24, s * 0.03);
      ctx.rotate(-1.42 + Math.sin(astronaut.wavePhase) * 0.14);
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(s * 0.34, -s * 0.01);
      ctx.lineTo(s * 0.54, s * 0.06);
      ctx.stroke();
      ctx.fillStyle = light ? 'rgba(228,240,246,0.96)' : 'rgba(222,236,246,0.96)';
      ctx.beginPath(); ctx.arc(s * 0.56, s * 0.05, s * 0.07, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(s * 0.60, -s * 0.04, s * 0.04, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
    } else if (gesture === 'salute') {
      ctx.beginPath();
      ctx.moveTo(s * 0.23, s * 0.03);
      ctx.lineTo(s * 0.44, -s * 0.16);
      ctx.lineTo(s * 0.26, -s * 0.34);
      ctx.stroke();
      ctx.fillStyle = light ? 'rgba(228,240,246,0.96)' : 'rgba(222,236,246,0.96)';
      ctx.beginPath(); ctx.arc(s * 0.24, -s * 0.35, s * 0.07, 0, Math.PI * 2); ctx.fill();
    } else if (gesture === 'visor') {
      ctx.beginPath();
      ctx.moveTo(s * 0.24, s * 0.04);
      ctx.lineTo(s * 0.34, -s * 0.24);
      ctx.lineTo(s * 0.02, -s * 0.42);
      ctx.stroke();
      ctx.fillStyle = light ? 'rgba(228,240,246,0.96)' : 'rgba(222,236,246,0.96)';
      ctx.beginPath(); ctx.arc(-s * 0.03, -s * 0.42, s * 0.07, 0, Math.PI * 2); ctx.fill();
    } else {
      const waveA = -0.95 + Math.sin(astronaut.wavePhase) * (gesture === 'spin' ? 0.30 : 0.50);
      ctx.save();
      ctx.translate(s * 0.25, s * 0.04);
      ctx.rotate(waveA);
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(s * 0.34, -s * 0.01);
      ctx.lineTo(s * 0.58, s * 0.08);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(s * 0.60, s * 0.09, s * 0.07, 0, Math.PI * 2);
      ctx.fillStyle = light ? 'rgba(228,240,246,0.95)' : 'rgba(222,236,246,0.95)';
      ctx.fill();
      ctx.restore();
    }

    /* legs and boots */
    ctx.beginPath();
    ctx.moveTo(-s * 0.16, s * 1.02);
    ctx.lineTo(-s * 0.22, s * 1.42);
    ctx.moveTo(s * 0.16, s * 1.02);
    ctx.lineTo(s * 0.22, s * 1.42);
    ctx.stroke();

    ctx.fillStyle = light ? 'rgba(170,188,200,0.96)' : 'rgba(164,186,202,0.96)';
    roundedRectPath(ctx, -s * 0.30, s * 1.38, s * 0.16, s * 0.09, 1.6);
    ctx.fill();
    roundedRectPath(ctx, s * 0.14, s * 1.38, s * 0.16, s * 0.09, 1.6);
    ctx.fill();

    /* tiny RCS thruster blink */
    const thrusterA = 0.10 + 0.12 * (0.5 + 0.5 * Math.sin(astronaut.suitDrift * 3.4));
    ctx.fillStyle = light
      ? `rgba(0,145,186,${thrusterA.toFixed(3)})`
      : `rgba(132,232,255,${thrusterA.toFixed(3)})`;
    ctx.beginPath();
    ctx.arc(-s * 0.56, s * 0.44, s * 0.05, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();

    /* speech cloud */
    const ba = astronaut.bubble.alpha;
    if (ba > 0.01 && astronaut.bubble.text) {
      const text = astronaut.bubble.text;
      ctx.save();
      ctx.font = '700 11px Space Mono, monospace';
      const padX = 10;
      const padY = 8;
      const lineH = 14;
      const maxBubbleW = clamp(W * 0.34, 180, 280);
      const maxTextW = Math.max(120, maxBubbleW - padX * 2);

      const words = text.trim().split(/\s+/).filter(Boolean);
      const lines = [];
      let line = '';

      for (const word of words) {
        const test = line ? `${line} ${word}` : word;
        if (ctx.measureText(test).width <= maxTextW) {
          line = test;
          continue;
        }

        if (line) lines.push(line);

        /* fallback for very long single token */
        if (ctx.measureText(word).width > maxTextW) {
          let chunk = '';
          for (const ch of word) {
            const chunkTest = chunk + ch;
            if (ctx.measureText(chunkTest).width <= maxTextW) {
              chunk = chunkTest;
            } else {
              lines.push(chunk);
              chunk = ch;
            }
          }
          line = chunk;
        } else {
          line = word;
        }
      }
      if (line) lines.push(line);

      const maxLines = 4;
      let bubbleLines = lines;
      if (bubbleLines.length > maxLines) {
        bubbleLines = lines.slice(0, maxLines);
        const last = bubbleLines[maxLines - 1];
        let trimmed = last;
        while (trimmed.length > 0 && ctx.measureText(`${trimmed}...`).width > maxTextW) {
          trimmed = trimmed.slice(0, -1);
        }
        bubbleLines[maxLines - 1] = `${trimmed}...`;
      }

      let textW = 0;
      for (const ln of bubbleLines) textW = Math.max(textW, ctx.measureText(ln).width);
      const tw = Math.min(maxBubbleW, textW + padX * 2);
      const th = Math.max(28, padY * 2 + lineH * bubbleLines.length);
      const bx = clamp(x + s * 1.2, 8, W - tw - 8);
      const by = clamp(y - s * 2.9, 8, H - th - 8);

      ctx.fillStyle = light
        ? `rgba(250,255,255, ${(0.86 * ba).toFixed(3)})`
        : `rgba(6,24,34, ${(0.82 * ba).toFixed(3)})`;
      ctx.strokeStyle = light
        ? `rgba(0,125,145, ${(0.40 * ba).toFixed(3)})`
        : `rgba(120,235,225, ${(0.46 * ba).toFixed(3)})`;
      ctx.lineWidth = 1;
      roundedRectPath(ctx, bx, by, tw, th, 9);
      ctx.fill();
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(bx + 14, by + th);
      ctx.lineTo(bx + 22, by + th);
      ctx.lineTo(x + s * 0.44, y - s * 1.1);
      ctx.closePath();
      ctx.fill();

      ctx.fillStyle = light
        ? `rgba(0,98,116, ${(0.92 * ba).toFixed(3)})`
        : `rgba(205,250,245, ${(0.95 * ba).toFixed(3)})`;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      for (let i = 0; i < bubbleLines.length; i++) {
        ctx.fillText(bubbleLines[i], bx + padX, by + padY + i * lineH);
      }
      ctx.restore();
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

    if (astronaut && !astronaut.dragging) {
      const funny = ASTRONAUT_EXPLOSION_PHRASES[Math.floor(Math.random() * ASTRONAUT_EXPLOSION_PHRASES.length)];
      triggerAstronautBubble(funny, 'explosion');
      astronaut.moodBoostSec = 3.2;
      setAstronautMood('excited', 3.4);
    }

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
    if (scrollReact.cooldownSec > 0) scrollReact.cooldownSec -= dtSec;
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
    updateAstronaut(dtSec);

    drawGalaxy();
    drawNebulae();
    drawStars();
    drawGrid();
    drawGravityWells();
    drawMobileBlackHoles();
    drawComets();
    drawSuperPhotons();
    drawStarStickers();
    drawBrandMarks();
    drawAstronaut();

    raf = requestAnimationFrame(draw);
  };

  /* ── init ─────────────────────────────────────────────────────────── */
  window.addEventListener('resize', resize, { passive: true });

  /* ── mouse tracking (full-page, since canvas is fixed) ──────────── */
  document.addEventListener('mousemove', e => {
    mouse.x = e.clientX;
    mouse.y = e.clientY;
    mouse.active = true;
    moveAstronautDrag(e.clientX, e.clientY);
    if (astronaut) astronaut.hover = astronautHit(e.clientX, e.clientY);
  }, { passive: true });
  document.addEventListener('mouseleave', () => {
    mouse.active = false;
    if (astronaut) astronaut.hover = false;
    endAstronautDrag();
  });

  document.addEventListener('mousedown', e => {
    if (intro.enabled) return;
    beginAstronautDrag(e.clientX, e.clientY);
  }, { passive: true });

  document.addEventListener('mouseup', () => {
    endAstronautDrag();
  }, { passive: true });

  /* touch support */
  document.addEventListener('touchstart', e => {
    if (intro.enabled) return;
    if (e.touches.length > 0) {
      beginAstronautDrag(e.touches[0].clientX, e.touches[0].clientY);
    }
  }, { passive: true });

  document.addEventListener('touchmove', e => {
    if (e.touches.length > 0) {
      mouse.x = e.touches[0].clientX;
      mouse.y = e.touches[0].clientY;
      mouse.active = true;
      moveAstronautDrag(e.touches[0].clientX, e.touches[0].clientY);
    }
  }, { passive: true });
  document.addEventListener('touchend', () => {
    mouse.active = false;
    if (astronaut) astronaut.hover = false;
    endAstronautDrag();
  }, { passive: true });

  document.addEventListener('click', e => {
    if (intro.enabled) return;

    if (astronautSuppressClick) {
      astronautSuppressClick = false;
      return;
    }

    const x = e.clientX;
    const y = e.clientY;

    const starIdx = hitStarSticker(x, y);
    if (starIdx >= 0) {
      starStickers.splice(starIdx, 1);
      if (astronaut) {
        setAstronautMood('excited', 3.2);
        astronaut.moodBoostSec = 3.2;
      }
      const msg = ASTRONAUT_STICKER_PHRASES[Math.floor(Math.random() * ASTRONAUT_STICKER_PHRASES.length)];
      triggerAstronautBubble(msg, 'sticker');
      return;
    }

    if (astronautHit(x, y)) {
      if (astronaut) {
        astronaut.moodBoostSec = 4;
        astronaut.hover = true;
        astronaut.dragMoved = false;
        setAstronautMood('excited', 3.6);
      }
      const text = ASTRONAUT_CLICK_PHRASES[Math.floor(Math.random() * ASTRONAUT_CLICK_PHRASES.length)];
      triggerAstronautBubble(text, 'click');
      return;
    }

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
      if (astronaut) {
        triggerAstronautBubble('Welcome back, commander.', 'event');
        setAstronautMood('curious', 3.0);
      }
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
      const now = performance.now();
      const yNow = window.scrollY || 0;
      const dt = Math.max(16, now - scrollReact.lastTs);
      const vy = Math.abs((yNow - scrollReact.lastY) / dt) * 1000;

      const y = clamp(window.scrollY * 0.15, 0, 80);
      heroBg.style.transform = `translateY(${y}px)`;

      if (vy > 900 && scrollReact.cooldownSec <= 0 && astronaut && !document.hidden) {
        triggerAstronautBubble('Hyperspace scroll detected. Wheee!', 'event');
        setAstronautMood('excited', 2.4);
        scrollReact.cooldownSec = 7.0;
      }

      scrollReact.lastY = yNow;
      scrollReact.lastTs = now;
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
