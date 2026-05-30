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
  const ASTRONAUT_UNLOCK_PROMPTS = [
    'Double click me to unlock my orbit.',
    'Double click to free Astro for a drift run.',
    'Astro is docked. Double click if you want free roam.',
    'Need a moving copilot? Double click and I will roam.',
    'Double click me and I will cruise around the stars.',
    'Double click on me and I will be free in space.',
  ];
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
    'Your thinking has excellent gravity.',
    'Nice work, star pilot.',
    'The universe likes your style.',
    'Keep steering by curiosity.',
    'Your focus is shining.',
    'That was a sharp move.',
    'I am cheering quietly in zero-g.',
    'You are building something cool.',
    'Mission control approves your pace.',
    'Small steps, huge orbit.',
    'Your curiosity is doing the heavy lifting.',
    'Nice pattern recognition.',
    'The stars are impressed.',
    'Your orbit is looking healthy.',
    'That idea just sparkled.',
    'Excellent trajectory.',
    'You are doing real science now.',
    'Tiny badge of cosmic honor.',
    'That felt like a win.',
    'I am proud of this data voyage.',
    'You bring the nice kind of chaos.',
    'Smooth operator in a noisy universe.',
    'You made that look easy.',
    'Keep the signal, lose the noise.',
    'Orbit stabilized nicely.',
    'Fresh ideas detected.',
    'Your brain is on a good wavelength.',
    'Very solid cosmic thinking.',
    'We are in a good timeline.',
    'You are charting new constellations.',
    'Nice one, commander.',
    'I trust your next move.',
    'The nebula noticed that.',
    'You just improved the universe.',
    'Click Connect to get in touch with us anytime.',
    'Try Machine Gnostics OSS by clicking the Install button.',
    'Go through the FAQ. It is full of mission-ready answers.',
    'We are happy to provide support whenever you need it.',
    'Double click on me and I will be free in space.',
  ];
  const ASTRONAUT_CLICK_PHRASES = [
    'Hey, Aestro is awake and available.',
    'Aestro says hello with extra visor energy.',
    'Need a guide? Aestro is already orbiting.',
    'Aestro is listening with the good ear of space.',
    'Tap again if you want another Aestro line.',
    'Aestro is in a chatty mood right now.',
    'That click reached Aestro HQ.',
    'Aestro can do a tiny wave and a big opinion.',
    'You found Aestro in interactive mode.',
    'Aestro is here for questions, vibes, and stardust.',
    'Aestro heard you. Nice timing.',
    'Aestro says: keep the questions coming.',
    'This is Aestro, professional helper of the orbit.',
    'Aestro is flattered by the click.',
    'Aestro can be helpful and dramatic at the same time.',
    'Aestro is practicing friendly zero-g replies.',
    'Aestro is ready for another move.',
    'Aestro approves this interaction.',
    'Aestro is doing the little astronaut thing again.',
    'Click success. Aestro noticed.',
    'Aestro has opinions, and they are mostly positive.',
    'Aestro is a good companion for curious clicks.',
    'Aestro can keep orbiting while you explore.',
    'Aestro says the current mission looks promising.',
    'Aestro is here with quiet confidence.',
    'Aestro can help you think out loud.',
    'Aestro likes this conversation already.',
    'Aestro is tiny, but the enthusiasm is large.',
    'Aestro is giving this click a thumbs-up.',
    'Aestro sees you. That counts as a meeting.',
    'Aestro is in interactive mode and feeling good.',
    'Aestro can do curiosity on demand.',
    'Aestro says your timing is excellent.',
    'Aestro is ready for another round.',
    'Aestro is busy making the orbit friendlier.',
    'Aestro welcomes your next bright idea.',
    'Aestro is happy to be the tiny mascot of this page.',
    'Aestro can answer with style.',
    'Aestro is keeping the mood light.',
    'Aestro likes a good click with purpose.',
    'Aestro is now officially engaged.',
    'Aestro can do this all day, sort of.',
    'Aestro thinks this is a decent use of time.',
    'Aestro is looking at the stars and at you.',
    'Aestro says good things happen after a click.',
    'Aestro is here, bright and ready.',
    'Aestro can be your tiny mission buddy.',
    'Aestro is grateful for the attention.',
    'Aestro is the sort of helper that waves back.',
    'Aestro says welcome to the friendly orbit.',
  ].map(message => message.replace(/Aestro/g, 'Astro'));
  const ASTRONAUT_SCROLL_PHRASES = [
    'Nice scroll. I am reading the orbit.',
    'That is a smooth page glide.',
    'Scroll speed: delightfully cosmic.',
    'You are moving the universe along.',
    'Astro likes this little gravity drift.',
    'Your scroll has excellent momentum.',
    'We are surfing the page together.',
    'That scroll felt scientifically tidy.',
    'I saw that page move. Very elegant.',
    'Orbit status: gently accelerating.',
    'You are making the page do a moonwalk.',
    'Nice, clean scroll energy.',
  ];
  const ASTRONAUT_RETURN_PHRASES = [
    'Welcome back, commander. I saved your orbit.',
    'Good to see you again. Astro remembers the vibe.',
    'You came back. That is my favorite signal.',
    'Re-entry complete. Glad you are here.',
    'Astro noticed the repeat visit and waved first.',
    'Same page, fresh orbit, new ideas.',
    'You are back. I am still on duty.',
    'Welcome home to the friendly side of space.',
  ];
  const ASTRONAUT_GREETING_PHRASES = [
    'Hello from low orbit.',
    'Hi there. Astro is online.',
    'Good to see you in this corner of space.',
    'Astro reporting in with bright visor energy.',
    'Welcome in. The orbit feels friendly today.',
    'Nice to have you here, explorer.',
    'Astro says hello and means it.',
    'Greetings from the page periphery.',
    'A fresh signal just arrived. Hi.',
    'Astro is here and the mission looks good.',
    'Hello, commander. Systems feel steady.',
    'Welcome aboard this quiet little orbit.',
    'A good visit has been detected.',
    'Astro waves from the scenic route.',
    'Good to have your attention for a moment.',
    'Astro welcomes you to the calm side of the galaxy.',
    'Hello again. I brought orbit-grade optimism.',
    'Warm greeting from a very small astronaut.',
    'Hello commander. Click Connect if you want to get in touch.',
    'Greetings. Try Machine Gnostics OSS by clicking Install.',
    'Hi there. We are happy to provide support.',
  ];
  const ASTRONAUT_TIME_PHRASES = {
    morning: [
      'Morning orbit looks clean and bright.',
      'Good morning. Astro is fully pressurized.',
      'Morning light makes this mission look sharp.',
      'Fresh day, fresh signal, steady orbit.',
      'The morning shift says hello.',
      'Astro likes the calm precision of early hours.',
      'Good morning. Time to do elegant work.',
      'The page is awake and so am I.',
    ],
    afternoon: [
      'Afternoon orbit check: still looking strong.',
      'The midday signal is stable.',
      'Afternoon energy detected. Nice pace.',
      'Astro approves this productive part of the day.',
      'Mid-orbit hours are good for sharp ideas.',
      'Afternoon glow suits this mission.',
      'A strong afternoon trajectory is forming.',
      'This part of the day has nice momentum.',
    ],
    evening: [
      'Evening orbit feels thoughtful and precise.',
      'Astro likes the color of this hour.',
      'Good evening. The page has a nice glow now.',
      'Evening mode looks excellent on this mission.',
      'The signal gets cinematic around this time.',
      'Evening orbit invites good ideas.',
      'Astro is cruising into a smart evening.',
      'This is a good hour for calm progress.',
    ],
    night: [
      'Night orbit is active and quietly beautiful.',
      'Astro is on the night shift.',
      'Late-hour signal still looks strong.',
      'Night mode gives this mission extra atmosphere.',
      'Space feels especially real at this hour.',
      'Astro is keeping a careful eye on the stars.',
      'A calm night orbit has entered the chat.',
      'Quiet hours, clear signal, steady thoughts.',
    ],
  };
  const ASTRONAUT_DAY_PHRASES = {
    sunday: [
      'Sunday orbit says take it steady.',
      'Sunday has a soft-launch kind of mood.',
      'Astro likes this quiet Sunday drift.',
      'Sunday is good for big-picture thinking.',
    ],
    monday: [
      'Monday mission briefing: strong start.',
      'Astro is treating Monday like a launch window.',
      'Monday orbit is locked in.',
      'A clean Monday trajectory is forming.',
    ],
    tuesday: [
      'Tuesday is excellent for careful progress.',
      'Astro sees a sturdy Tuesday signal.',
      'Tuesday orbit looks balanced and sharp.',
      'This Tuesday has good engineering energy.',
    ],
    wednesday: [
      'Wednesday is holding formation nicely.',
      'Midweek orbit is stable and readable.',
      'Astro calls this a solid Wednesday burn.',
      'Wednesday feels precise today.',
    ],
    thursday: [
      'Thursday orbit is moving with intent.',
      'Astro likes the confident pace of Thursday.',
      'Thursday is giving strong mission energy.',
      'A very respectable Thursday signal is present.',
    ],
    friday: [
      'Friday orbit has extra sparkle.',
      'Astro can feel the Friday lift.',
      'Friday signal is playful but controlled.',
      'A good Friday deserves a tiny celebration spin.',
    ],
    saturday: [
      'Saturday orbit is relaxed and bright.',
      'Astro enjoys the easy rhythm of Saturday.',
      'Saturday feels like open space with good coffee.',
      'This Saturday trajectory is delightfully smooth.',
    ],
  };
  const ASTRONAUT_MOOD_PHRASES = {
    chill: [
      'Astro is feeling steady and unbothered.',
      'Calm orbit. Clear signal.',
      'Everything feels nicely aligned right now.',
      'Astro is in a smooth and thoughtful lane.',
      'This is good quiet progress energy.',
      'A calm mission is still a strong mission.',
      'Astro is keeping the orbit balanced.',
      'Steady mood. Good trajectory.',
    ],
    excited: [
      'Astro is buzzing like a happy satellite.',
      'This mood has excellent launch energy.',
      'Excitement level: visibly orbital.',
      'Astro is ready to wave at the universe.',
      'The signal just got extra bright.',
      'This mission has real momentum now.',
      'Astro is doing tiny victory laps internally.',
      'Very lively orbit right now.',
    ],
    curious: [
      'Astro is in question-mark mode.',
      'Curiosity is steering the capsule today.',
      'There may be a clever pattern nearby.',
      'Astro is leaning toward discovery.',
      'This mood likes to inspect the details.',
      'Curious orbit means something interesting is close.',
      'Astro is peeking at possibilities.',
      'Question-rich air detected inside the helmet.',
    ],
    sleepy: [
      'Astro is moving in soft-focus mode.',
      'Low-fi orbit activated.',
      'This mood prefers gentle momentum.',
      'Astro is calm, slow, and still watching carefully.',
      'Sleepy orbit can still produce clever ideas.',
      'A quieter mood has entered the cabin.',
      'Astro is running on moonlight and manners.',
      'Soft orbit, stable mission.',
    ],
  };
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
    'Machine Gnostics OSS signal stamped in local space.',
    'Machine Gnostics OSS beacon online.',
    'Orbit tag: Machine Gnostics OSS locked and glowing.',
    'Friendly mark deployed. Machine Gnostics OSS says hi.',
    'Machine Gnostics OSS signature dropped into this star lane.',
  ];
  const ASTRONAUT_MARKETING_PHRASES = [
    'Machine Gnostics OSS turns data into delightful signals.',
    'Machine Gnostics OSS makes insight feel like a launch sequence.',
    'Smart models, fun orbit, less noise.',
    'Machine Gnostics OSS: where structure meets starlight.',
    'Better questions, brighter answers.',
    'Machine Gnostics OSS keeps the signal and loses the drama.',
    'Fun science for serious thinkers.',
    'Machine Gnostics OSS makes pattern finding feel playful.',
    'Cosmic clarity for your next decision.',
    'Machine Gnostics OSS: elegant learning for curious teams.',
    'Less guesswork, more gravity.',
    'The Machine Gnostics OSS way: calm data, bold insight.',
    'Machine Gnostics OSS helps ideas find orbit.',
    'Turn messy signals into a guided mission.',
    'Machine Gnostics OSS is data, but with a smile.',
    'Insight should be this smooth.',
    'Machine Gnostics OSS: practical intelligence with cosmic polish.',
    'Build models that behave like good satellites.',
    'Machine Gnostics OSS keeps your analytics in formation.',
    'Learn faster, steer cleaner.',
    'Machine Gnostics OSS makes complexity feel friendly.',
    'Tiny data, big clarity.',
    'Machine Gnostics OSS: the fun side of rigorous thinking.',
    'Let the signal wear the crown.',
    'Machine Gnostics OSS gives your data a cockpit.',
    'Beautiful structure for busy minds.',
    'Machine Gnostics OSS: engineered for curious orbiters.',
    'Make your next metric actually mean something.',
    'Machine Gnostics OSS turns friction into forward motion.',
    'Analytics with a little spark in the visor.',
    'Machine Gnostics OSS helps teams see what matters.',
    'Calm dashboards, sharper decisions.',
    'Machine Gnostics OSS: where neat math meets neat vibes.',
    'Put your data on a better trajectory.',
    'Machine Gnostics OSS is science with style.',
    'Machine Gnostics OSS keeps the mission readable.',
    'Make your model less noisy and more noble.',
    'Machine Gnostics OSS turns raw signals into runway.',
    'Crisp insight for cosmic-scale questions.',
    'Machine Gnostics OSS: friendly rigor for modern teams.',
    'Data should feel this clear.',
    'Machine Gnostics OSS makes the unknown less spooky.',
    'Better logic, better launch.',
    'Machine Gnostics OSS gives your pipeline a pulse.',
    'Curious minds choose Machine Gnostics OSS.',
    'Machine Gnostics OSS makes meaning pop.',
    'A better orbit for your numbers.',
    'Machine Gnostics OSS: playful, precise, and practical.',
    'Let Machine Gnostics OSS help the signal sing.',
    'Machine Gnostics OSS is your friendly gravity well for insight.',
    'Want to talk? Click the Connect button and get in touch.',
    'Try Machine Gnostics OSS by clicking the Install button.',
    'Go through the FAQ for quick answers and mission tips.',
    'We are happy to provide support for your next step.',
  ];
  const ASTRONAUT_FRIDAY_SPECIAL = 'WEEKEND ORBIT: Friday detected. Keep the vibes in stable orbit.';
  const ASTRONAUT_TAGS = ['COMMS', 'EVA LOG', 'MISSION TIP', 'ORBIT NOTE'];
  const ASTRONAUT_MOODS = ['chill', 'excited', 'curious', 'sleepy'];
  const ASTRONAUT_GESTURES = ['wave', 'thumbs', 'salute', 'visor', 'spin'];
  const ASTRONAUT_OUTFITS = [
    { name: 'mint', body: [238, 250, 248], band: [0, 212, 170], trim: [0, 126, 152], visor: [54, 92, 118] },
    { name: 'amber', body: [251, 246, 235], band: [255, 156, 64], trim: [176, 110, 24], visor: [112, 78, 50] },
    { name: 'ice', body: [242, 248, 255], band: [0, 145, 186], trim: [0, 110, 132], visor: [52, 98, 132] },
    { name: 'sunset', body: [250, 244, 240], band: [255, 136, 38], trim: [166, 96, 18], visor: [120, 88, 54] },
    { name: 'orbit', body: [244, 249, 246], band: [0, 230, 214], trim: [0, 126, 148], visor: [44, 92, 118] },
    { name: 'stellar', body: [249, 247, 252], band: [110, 236, 224], trim: [96, 118, 172], visor: [60, 72, 124] },
    { name: 'weekday', body: [245, 249, 250], band: [0, 212, 170], trim: [0, 118, 132], visor: [58, 94, 120] },
  ];
  const ASTRO_BUBBLE_TIMER_MIN = 5.6;
  const ASTRO_BUBBLE_TIMER_MAX = 7.6;
  const ASTRONAUT_PERSONALITIES = [
    { key: 'guide', label: 'Guide', defaultMood: 'chill', motion: { bob: 0.92, tilt: 0.74, wave: 0.94, leg: 0.92, surprise: 0.13 } },
    { key: 'spark', label: 'Spark', defaultMood: 'excited', motion: { bob: 1.16, tilt: 1.02, wave: 1.22, leg: 1.08, surprise: 0.22 } },
    { key: 'zen', label: 'Zen', defaultMood: 'sleepy', motion: { bob: 0.76, tilt: 0.50, wave: 0.82, leg: 0.84, surprise: 0.08 } },
    { key: 'trickster', label: 'Trickster', defaultMood: 'curious', motion: { bob: 1.08, tilt: 1.16, wave: 1.10, leg: 1.06, surprise: 0.28 } },
  ];
  const ASTRO_VISIT_KEY = 'machinegnostics.astro.visits';

  /* galaxy background — stardust band + core + distant galaxies */
  let galaxyDust    = [];
  let galaxySmudges = [];
  let galaxyCore    = { x: 0, y: 0 };
  let solarSystems  = [];

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
    /* stagger initial age so lifecycle transitions are naturally offset */
    mobHoles = Array.from({ length: 4 }, (_, i) => {
      const bh = initMobHole(newMobHole());
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

    /* mini solar systems: exactly two (left + right), gently floating */
    solarSystems = [0, 1].map(side => {
      const onLeft = side === 0;
      const sx = onLeft ? rand(W * 0.12, W * 0.34) : rand(W * 0.66, W * 0.88);
      const sy = rand(H * 0.14, H * 0.84);
      const starR = rand(2.2, 4.2);
      const planetCount = Math.floor(rand(3, 5));
      const starCol = isLight() ? [160, 118, 36] : [255, 226, 140];
      const baseOrbitSpeed = rand(0.170, 0.260);
      const planets = Array.from({ length: planetCount }, (_, i) => {
        const ring = starR * (2.6 + i * rand(1.15, 1.45));
        const speedScale = Math.max(0.34, 1.00 - i * rand(0.18, 0.28));
        return {
          orbitR: ring,
          orbitYScale: rand(0.72, 1.0),
          orbitRot: rand(0, Math.PI * 2),
          phase: rand(0, Math.PI * 2),
          speed: baseOrbitSpeed * speedScale,
          r: rand(0.65, 1.45),
          col: isLight()
            ? [Math.floor(rand(0, 120)), Math.floor(rand(100, 165)), Math.floor(rand(90, 170))]
            : [Math.floor(rand(120, 230)), Math.floor(rand(160, 245)), Math.floor(rand(170, 255))],
        };
      });
      return {
        x: sx,
        y: sy,
        driftAx: rand(12, 26),
        driftAy: rand(8, 20),
        driftSpeedX: rand(0.010, 0.028),
        driftSpeedY: rand(0.008, 0.022),
        driftPhaseX: rand(0, Math.PI * 2),
        driftPhaseY: rand(0, Math.PI * 2),
        starR,
        starCol,
        planets,
      };
    });

    initAstronaut();

    if (intro.enabled) {
      initIntro();
    }
  };

  const initAstronaut = () => {
    const size = clamp(Math.min(W, H) * 0.013, 8.4, 14.2);

    if (!astronaut) {
      const px = rand(0, 1) < 0.5 ? rand(W * 0.05, W * 0.22) : rand(W * 0.78, W * 0.95);
      let visitCount = 1;
      try {
        visitCount = Math.max(1, parseInt(localStorage.getItem(ASTRO_VISIT_KEY) || '0', 10) + 1);
        localStorage.setItem(ASTRO_VISIT_KEY, String(visitCount));
      } catch (_err) {
        visitCount = 1;
      }
      const personality = ASTRONAUT_PERSONALITIES[(visitCount - 1) % ASTRONAUT_PERSONALITIES.length];
      const dock = getAstronautDockPosition('right');
      astronaut = {
        x: dock.x,
        y: dock.y,
        vx: rand(-0.24, 0.24),
        vy: rand(-0.18, 0.18),
        targetVx: rand(-0.24, 0.24),
        targetVy: rand(-0.18, 0.18),
        targetX: dock.x,
        targetY: dock.y,
        driftTimerSec: rand(2.2, 5.5),
        bobPhase: rand(0, Math.PI * 2),
        wavePhase: rand(0, Math.PI * 2),
        spinPhase: 0,
        moodBoostSec: 0,
        mood: personality.defaultMood,
        moodTimerSec: rand(6.0, 12.0),
        gesture: 'wave',
        gestureTimerSec: rand(3.0, 6.5),
        brandTimerSec: rand(14.0, 30.0),
        marketingTimerSec: rand(12.0, 24.0),
        surpriseTimerSec: rand(9.0, 18.0),
        outfitKey: '',
        outfitTheme: null,
        outfitSpinSec: 0,
        personalityKey: personality.key,
        personalityLabel: personality.label,
        visitCount,
        movementMode: 'locked',
        dockSide: 'right',
        dockX: dock.x,
        dockY: dock.y,
        requestUnlockTimerSec: rand(10.0, 18.0),
        requestSignalSec: 0,
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
          metaPrompt: '',
          statusRows: [],
          timerSec: 0,
          cooldownSec: rand(8.5, 15.0),
          alpha: 0,
          typingElapsedSec: 0,
          typingChars: 0,
          typingSpeed: 30,
        },
      };
      return;
    }

    astronaut.size = size;
    const dock = getAstronautDockPosition(astronaut.dockSide || 'right');
    astronaut.dockX = dock.x;
    astronaut.dockY = dock.y;
    if (astronaut.movementMode === 'locked') {
      astronaut.x = dock.x;
      astronaut.y = dock.y;
      astronaut.targetX = dock.x;
      astronaut.targetY = dock.y;
      astronaut.vx = 0;
      astronaut.vy = 0;
    } else if (astronaut.movementMode === 'returning') {
      astronaut.targetX = dock.x;
      astronaut.targetY = dock.y;
    }
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
    const previousMood = astronaut.mood;
    astronaut.mood = mood;
    astronaut.moodTimerSec = durationSec;

    if (previousMood !== mood) {
      astronaut.outfitSpinSec = 1.15;
      astronaut.gesture = 'spin';
      astronaut.gestureTimerSec = Math.max(astronaut.gestureTimerSec, 1.4);
    }
  };

  const getAstronautOutfitTheme = () => {
    const day = new Date().getDay();
    const isFriday = day === 5;
    const mood = astronaut ? astronaut.mood : 'chill';
    const base = ASTRONAUT_OUTFITS[day % ASTRONAUT_OUTFITS.length];
    const moodMap = {
      chill: base,
      excited: ASTRONAUT_OUTFITS[1],
      curious: ASTRONAUT_OUTFITS[2],
      sleepy: ASTRONAUT_OUTFITS[0],
    };
    const selected = moodMap[mood] || base;
    if (isFriday) {
      return {
        name: 'friday-orbit',
        body: ASTRONAUT_OUTFITS[4].body,
        band: ASTRONAUT_OUTFITS[4].band,
        trim: ASTRONAUT_OUTFITS[4].trim,
        visor: ASTRONAUT_OUTFITS[5].visor,
      };
    }
    return selected;
  };

  const maybeUpdateAstronautOutfit = () => {
    if (!astronaut) return;
    const theme = getAstronautOutfitTheme();
    const key = `${new Date().getDay()}-${astronaut.mood}-${theme.name}`;
    if (astronaut.outfitKey !== key) {
      astronaut.outfitKey = key;
      astronaut.outfitTheme = theme;
      astronaut.outfitSpinSec = Math.max(astronaut.outfitSpinSec, 1.1);
      astronaut.gesture = 'spin';
      astronaut.gestureTimerSec = Math.max(astronaut.gestureTimerSec, 1.2);
    }
  };

  const getAstronautDockPosition = (side = 'right') => {
    const size = astronaut ? astronaut.size : Math.max(14, Math.min(W, H) * 0.022);
    const margin = Math.max(24, size * 1.95);
    const headerEl = document.querySelector('.md-header');
    const headerOffset = headerEl ? headerEl.getBoundingClientRect().height : 0;
    return {
      x: side === 'left' ? margin + size * 0.2 : W - margin - size * 0.2,
      y: Math.max(margin + size * 0.88, headerOffset + size * 1.65),
    };
  };

  const setAstronautMovementMode = (mode) => {
    if (!astronaut) return;
    astronaut.movementMode = mode;
    if (mode === 'locked' || mode === 'returning') {
      const dock = getAstronautDockPosition(astronaut.dockSide || 'right');
      astronaut.dockX = dock.x;
      astronaut.dockY = dock.y;
      astronaut.targetX = dock.x;
      astronaut.targetY = dock.y;
      astronaut.requestSignalSec = mode === 'locked' ? astronaut.requestSignalSec : 0;
    } else {
      const target = pickAstronautTarget();
      astronaut.targetX = target.x;
      astronaut.targetY = target.y;
      astronaut.driftTimerSec = rand(1.2, 3.2);
      astronaut.requestSignalSec = 0;
    }
  };

  const toggleAstronautDockLock = () => {
    if (!astronaut) return;
    if (astronaut.movementMode === 'locked') {
      setAstronautMovementMode('roaming');
      astronaut.requestUnlockTimerSec = rand(18.0, 30.0);
      astronaut.moodBoostSec = 3.4;
      setAstronautMood('excited', 4.2);
      astronaut.gesture = 'wave';
      astronaut.gestureTimerSec = rand(2.0, 3.0);
      triggerAstronautBubble('Unlock received. Astro is free to roam.', 'event');
      return;
    }

    setAstronautMovementMode('returning');
    astronaut.requestUnlockTimerSec = rand(16.0, 28.0);
    astronaut.gesture = 'salute';
    astronaut.gestureTimerSec = rand(1.6, 2.8);
    triggerAstronautBubble('Return command accepted. Floating back to dock.', 'event');
  };

  const pickAstronautGesture = () => {
    if (!astronaut) return;
    const g = ASTRONAUT_GESTURES[Math.floor(Math.random() * ASTRONAUT_GESTURES.length)];
    astronaut.gesture = g;
    astronaut.gestureTimerSec = g === 'spin' ? rand(1.6, 2.2) : rand(2.6, 5.8);
  };

  const getAstronautPersonality = () => {
    if (!astronaut) return ASTRONAUT_PERSONALITIES[0];
    return ASTRONAUT_PERSONALITIES.find((item) => item.key === astronaut.personalityKey) || ASTRONAUT_PERSONALITIES[0];
  };

  const maybeTriggerAstroSurprise = () => {
    if (!astronaut || astronaut.dragging) return;
    const personality = getAstronautPersonality();
    if (Math.random() > personality.motion.surprise) return;
    const surprises = ['spin', 'wave', 'salute', 'thumbs'];
    astronaut.gesture = surprises[Math.floor(Math.random() * surprises.length)];
    astronaut.gestureTimerSec = rand(1.8, 3.0);
    if (Math.random() < 0.38) {
      setAstronautMood(ASTRONAUT_MOODS[Math.floor(Math.random() * ASTRONAUT_MOODS.length)], rand(2.8, 4.6));
    }
    if (Math.random() < 0.24) {
      triggerAstronautBubble('Tiny surprise maneuver complete.', 'event');
    }
  };

  const pickRandom = (items) => items[Math.floor(Math.random() * items.length)];

  const getAstroTimeBucket = () => {
    const hour = new Date().getHours();
    if (hour < 5) return 'night';
    if (hour < 12) return 'morning';
    if (hour < 17) return 'afternoon';
    if (hour < 21) return 'evening';
    return 'night';
  };

  const getAstroDayKey = () => {
    return ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday'][new Date().getDay()];
  };

  const pickAstroAmbientMessage = () => {
    if (!astronaut) return pickRandom(ASTRONAUT_IDLE_PHRASES);
    if (new Date().getDay() === 5 && Math.random() < 0.18) return ASTRONAUT_FRIDAY_SPECIAL;

    const roll = Math.random();
    if (roll < 0.16) return pickRandom(ASTRONAUT_GREETING_PHRASES);
    if (roll < 0.38) return pickRandom(ASTRONAUT_TIME_PHRASES[getAstroTimeBucket()]);
    if (roll < 0.56) return pickRandom(ASTRONAUT_DAY_PHRASES[getAstroDayKey()]);
    if (roll < 0.76) return pickRandom(ASTRONAUT_MOOD_PHRASES[astronaut.mood] || ASTRONAUT_MOOD_PHRASES.chill);
    if (astronaut.visitCount > 1 && roll < 0.84) return pickRandom(ASTRONAUT_RETURN_PHRASES);
    return pickRandom(ASTRONAUT_IDLE_PHRASES);
  };

  const formatCommsLine = (message, category = 'auto') => {
    const mood = astronaut ? astronaut.mood : 'chill';
    const moodTag = mood === 'excited' ? 'BUZZING' : mood === 'curious' ? 'CURIOUS' : mood === 'sleepy' ? 'LO-FI' : 'CALM';
    let tag;
    if (category === 'click') tag = 'COMMS';
    else if (category === 'event') tag = 'ORBIT NOTE';
    else if (category === 'explosion') tag = 'EVA LOG';
    else if (category === 'sticker') tag = 'MISSION TIP';
    else if (category === 'brand') tag = 'OSS SIGNAL';
    else if (category === 'marketing') tag = 'NOTE';
    else tag = ASTRONAUT_TAGS[Math.floor(Math.random() * ASTRONAUT_TAGS.length)];
    const personality = astronaut ? astronaut.personalityLabel : 'Guide';
    const timeStamp = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    return {
      prompt: 'astro@orbit:~ %',
      statusRows: [
        { label: 'mood', value: moodTag.toLowerCase(), tone: 'mood' },
        { label: 'persona', value: personality.toLowerCase(), tone: 'persona' },
        { label: 'channel', value: tag.toLowerCase(), tone: 'tag' },
        { label: 'time', value: timeStamp, tone: 'time' },
      ],
      text: message,
    };
  };

  const beginAstronautDrag = (px, py) => {
    if (!astronaut || !astronautHit(px, py)) return false;
    if (astronaut.movementMode === 'locked' || astronaut.movementMode === 'returning') return false;
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
      text = pickAstroAmbientMessage();
    }
    const formatted = formatCommsLine(text, category);
    astronaut.bubble.text = formatted.text;
    astronaut.bubble.metaPrompt = formatted.prompt;
    astronaut.bubble.statusRows = formatted.statusRows;
    astronaut.bubble.category = category;
    astronaut.bubble.typingElapsedSec = 0;
    astronaut.bubble.typingChars = 0;
    astronaut.bubble.typingSpeed = rand(24, 32);
    astronaut.bubble.timerSec = rand(ASTRO_BUBBLE_TIMER_MIN, ASTRO_BUBBLE_TIMER_MAX) + Math.min(text.length / astronaut.bubble.typingSpeed, 1.8);
    astronaut.bubble.cooldownSec = rand(10.0, 18.0);
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
    const personality = getAstronautPersonality();
    astronaut.moodTimerSec -= dtSec;
    astronaut.gestureTimerSec -= dtSec;
    astronaut.surpriseTimerSec -= dtSec;

    if (astronaut.moodTimerSec <= 0) {
      setAstronautMood(ASTRONAUT_MOODS[Math.floor(Math.random() * ASTRONAUT_MOODS.length)]);
    }
    if (astronaut.gestureTimerSec <= 0) {
      pickAstronautGesture();
    }

    const moodMotion = astronaut.mood === 'excited' ? 1.35 : astronaut.mood === 'curious' ? 1.12 : astronaut.mood === 'sleepy' ? 0.72 : 1.0;

    maybeUpdateAstronautOutfit();
    if (astronaut.outfitSpinSec > 0) astronaut.outfitSpinSec -= dtSec;

    if (astronaut.surpriseTimerSec <= 0) {
      maybeTriggerAstroSurprise();
      astronaut.surpriseTimerSec = rand(10.0, 22.0);
    }

    if (astronaut.hover || astronaut.dragging) {
      setAstronautMood('excited', 2.2);
    }

    if (astronaut.requestSignalSec > 0) astronaut.requestSignalSec -= dtSec;
    if (astronaut.movementMode === 'locked') {
      astronaut.requestUnlockTimerSec -= dtSec;
      if (astronaut.requestUnlockTimerSec <= 0 && astronaut.bubble.timerSec <= 0.05 && astronaut.bubble.cooldownSec <= 0.5) {
        const prompt = ASTRONAUT_UNLOCK_PROMPTS[Math.floor(Math.random() * ASTRONAUT_UNLOCK_PROMPTS.length)];
        astronaut.requestSignalSec = 5.4;
        astronaut.gesture = 'visor';
        astronaut.gestureTimerSec = rand(1.8, 3.0);
        triggerAstronautBubble(prompt, 'event');
        astronaut.requestUnlockTimerSec = rand(22.0, 36.0);
      }
    }

    if (!astronaut.dragging) {
      const isLocked = astronaut.movementMode === 'locked';
      const isReturning = astronaut.movementMode === 'returning';
      const reachTarget = Math.hypot(astronaut.x - astronaut.targetX, astronaut.y - astronaut.targetY) < Math.max(18, astronaut.size * 2.2);
      if (isLocked) {
        astronaut.targetX = astronaut.dockX;
        astronaut.targetY = astronaut.dockY;
      } else if (isReturning) {
        astronaut.targetX = astronaut.dockX;
        astronaut.targetY = astronaut.dockY;
      } else {
        astronaut.driftTimerSec -= dtSec;
        if (astronaut.driftTimerSec <= 0 || reachTarget || astronautInCenterZone(astronaut.x, astronaut.y)) {
          const target = pickAstronautTarget();
          astronaut.targetX = target.x;
          astronaut.targetY = target.y;
          astronaut.driftTimerSec = rand(1.4, 3.6);
        }
      }

      const dx = astronaut.targetX - astronaut.x;
      const dy = astronaut.targetY - astronaut.y;
      const d = Math.hypot(dx, dy) + 0.001;
      let speedMood = 1.0;
      if (astronaut.mood === 'sleepy') speedMood = 0.72;
      else if (astronaut.mood === 'curious') speedMood = 1.10;
      else if (astronaut.mood === 'excited') speedMood = 1.22;

      const speedScale = isLocked ? 0.28 : isReturning ? 0.74 : 1.0;
      const desiredVx = (dx / d) * clamp(d / 180, isLocked ? 0.02 : 0.09, isLocked ? 0.18 : isReturning ? 0.34 : 0.42) * speedMood * speedScale;
      const desiredVy = (dy / d) * clamp(d / 180, isLocked ? 0.02 : 0.07, isLocked ? 0.16 : isReturning ? 0.28 : 0.34) * speedMood * speedScale;

      astronaut.targetVx = desiredVx;
      astronaut.targetVy = desiredVy;
      astronaut.vx = lerp(astronaut.vx, astronaut.targetVx, (isLocked ? 0.10 : isReturning ? 0.05 : 0.032) * frameScale);
      astronaut.vy = lerp(astronaut.vy, astronaut.targetVy, (isLocked ? 0.10 : isReturning ? 0.05 : 0.032) * frameScale);

      if (!isLocked && !isReturning && astronautInCenterZone(astronaut.x, astronaut.y)) {
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

      if (isReturning && d < Math.max(10, astronaut.size * 0.65)) {
        setAstronautMovementMode('locked');
        astronaut.x = astronaut.dockX;
        astronaut.y = astronaut.dockY;
        astronaut.vx = 0;
        astronaut.vy = 0;
        astronaut.requestUnlockTimerSec = rand(18.0, 32.0);
        astronaut.gesture = 'salute';
        astronaut.gestureTimerSec = rand(1.2, 2.2);
        triggerAstronautBubble('Dock restored. Double click if you want another orbit run.', 'event');
      }
    }

    const margin = Math.max(20, astronaut.size * 1.6);
    astronaut.x = clamp(astronaut.x, margin, W - margin);
    astronaut.y = clamp(astronaut.y, margin, H - margin);
    astronaut.bobPhase += 0.014 * frameScale * (0.82 + moodMotion * 0.28) * personality.motion.bob;
    let waveSpd = astronaut.hover || astronaut.dragging ? 0.20 : 0.11;
    if (astronaut.mood === 'excited') waveSpd *= 1.22;
    else if (astronaut.mood === 'sleepy') waveSpd *= 0.74;
    astronaut.wavePhase += waveSpd * frameScale * (0.88 + moodMotion * 0.18) * personality.motion.wave;
    astronaut.spinPhase += (astronaut.gesture === 'spin' ? 0.18 : 0.03) * frameScale * moodMotion * personality.motion.tilt;
    astronaut.suitDrift += 0.012 * frameScale * (0.90 + moodMotion * 0.10);

    astronaut.bubble.cooldownSec -= dtSec;
    if (astronaut.bubble.timerSec > 0) {
      astronaut.bubble.timerSec -= dtSec;
      astronaut.bubble.typingElapsedSec += dtSec;
      astronaut.bubble.typingChars = Math.min(
        astronaut.bubble.text.length,
        Math.floor(astronaut.bubble.typingElapsedSec * astronaut.bubble.typingSpeed)
      );
    } else if (astronaut.bubble.cooldownSec <= 0) {
      triggerAstronautBubble();
    }

    const bubbleT = astronaut.bubble.timerSec;
    astronaut.bubble.alpha = bubbleT > 0
      ? Math.min(clamp((ASTRO_BUBBLE_TIMER_MAX - bubbleT) / 0.25, 0, 1), clamp(bubbleT / 0.55, 0, 1))
      : 0;

    if (astronaut.moodBoostSec > 0) astronaut.moodBoostSec -= dtSec;

    astronaut.brandTimerSec -= dtSec;
    if (astronaut.brandTimerSec <= 0 && !astronaut.dragging) {
      const msg = ASTRONAUT_BRAND_PHRASES[Math.floor(Math.random() * ASTRONAUT_BRAND_PHRASES.length)];
      triggerAstronautBubble(msg, 'brand');
      spawnBrandMark();
      astronaut.brandTimerSec = rand(18.0, 34.0);
    }

    astronaut.marketingTimerSec -= dtSec;
    if (astronaut.marketingTimerSec <= 0 && !astronaut.dragging) {
      const msg = ASTRONAUT_MARKETING_PHRASES[Math.floor(Math.random() * ASTRONAUT_MARKETING_PHRASES.length)];
      triggerAstronautBubble(msg, 'marketing');
      astronaut.gesture = 'thumbs';
      astronaut.gestureTimerSec = rand(2.2, 3.5);
      astronaut.marketingTimerSec = rand(16.0, 30.0);
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
    const s = astronaut.size * 1.12;
    const light = isLight();
    const hour = new Date().getHours();
    const isNight = hour < 6 || hour >= 19;
    const mood = astronaut.moodBoostSec > 0 ? clamp(astronaut.moodBoostSec / 4, 0, 1) : 0;
    const gesture = astronaut.gesture || 'wave';
    const personality = getAstronautPersonality();
    const theme = astronaut.outfitTheme || getAstronautOutfitTheme();
    const bodyColor = light
      ? [Math.max(0, (theme.body || [245, 250, 252])[0] - 40), Math.max(0, (theme.body || [245, 250, 252])[1] - 48), Math.max(0, (theme.body || [245, 250, 252])[2] - 30)]
      : (theme.body || [245, 250, 252]);
    const bandColor = light
      ? [Math.min(255, (theme.band || [0, 212, 170])[0] + 18), Math.min(255, (theme.band || [0, 212, 170])[1] + 14), Math.min(255, (theme.band || [0, 212, 170])[2] + 24)]
      : (theme.band || [0, 212, 170]);
    const trimColor = light
      ? [Math.max(0, (theme.trim || [0, 126, 152])[0] - 16), Math.max(0, (theme.trim || [0, 126, 152])[1] - 24), Math.max(0, (theme.trim || [0, 126, 152])[2] - 18)]
      : (theme.trim || [0, 126, 152]);
    const visorBase = light
      ? [Math.max(0, (theme.visor || [52, 98, 132])[0] - 8), Math.max(0, (theme.visor || [52, 98, 132])[1] - 12), Math.max(0, (theme.visor || [52, 98, 132])[2] - 16)]
      : (theme.visor || [52, 98, 132]);

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
    const baseRot = Math.sin(astronaut.bobPhase * 0.8) * 0.06 * personality.motion.tilt;
    const spinRot = gesture === 'spin' ? Math.sin(astronaut.spinPhase) * 0.22 : 0;
    const outfitSpin = astronaut.outfitSpinSec > 0 ? Math.sin(astronaut.outfitSpinSec * 7.5) * 0.55 : 0;
    const headTilt = astronaut.mood === 'curious' ? Math.sin(astronaut.wavePhase * 0.7) * 0.08 : astronaut.mood === 'sleepy' ? -0.04 : 0;
    ctx.rotate(baseRot + spinRot + outfitSpin + headTilt);

    const cueAlpha = light ? 0.92 : 0.98;
    const cueColor = `rgba(${bandColor[0]},${bandColor[1]},${bandColor[2]},${cueAlpha})`;
    const cueMuted = `rgba(${trimColor[0]},${trimColor[1]},${trimColor[2]},${light ? 0.62 : 0.78})`;
    ctx.save();
    ctx.translate(0, -s * 1.18);
    ctx.strokeStyle = cueColor;
    ctx.fillStyle = cueColor;
    ctx.lineWidth = 1.2;

    if (astronaut.requestSignalSec > 0) {
      const signalA = clamp(astronaut.requestSignalSec / 5.4, 0, 1) * (0.55 + 0.45 * Math.sin(time * 7.5));
      const signalColor = light
        ? `rgba(0,136,170, ${(0.78 * signalA).toFixed(3)})`
        : `rgba(114,244,232, ${(0.88 * signalA).toFixed(3)})`;
      ctx.strokeStyle = signalColor;
      ctx.fillStyle = signalColor;
      ctx.lineWidth = 1.15;
      ctx.beginPath();
      ctx.arc(0, -4, 1.6, 0, Math.PI * 2);
      ctx.fill();
      for (const rr of [6, 10, 14]) {
        ctx.beginPath();
        ctx.arc(0, -4, rr, 1.18 * Math.PI, 1.82 * Math.PI);
        ctx.stroke();
      }
    }

    if (astronaut.mood === 'excited') {
      for (const [dx, dy, r] of [[-10, -2, 1.8], [0, -10, 2.4], [10, -2, 1.8]]) {
        ctx.beginPath();
        ctx.arc(dx, dy, r, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.beginPath();
      ctx.arc(0, -16, 7, 0.18 * Math.PI, 0.82 * Math.PI);
      ctx.stroke();
    } else if (astronaut.mood === 'curious') {
      ctx.beginPath();
      ctx.arc(0, -7, 5.4, 0.2 * Math.PI, 1.75 * Math.PI);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(5, 6, 1.9, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.moveTo(-7, 8);
      ctx.lineTo(-3, 0);
      ctx.stroke();
    } else if (astronaut.mood === 'sleepy') {
      ctx.strokeStyle = cueMuted;
      ctx.beginPath();
      ctx.moveTo(-8, -2);
      ctx.lineTo(-2, -8);
      ctx.lineTo(4, -2);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(2, 8);
      ctx.lineTo(8, 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(-10, 10, 1.8, 0, Math.PI * 2);
      ctx.fill();
    } else {
      ctx.strokeStyle = cueColor;
      ctx.beginPath();
      ctx.arc(-8, -4, 2.0, 0, Math.PI * 2);
      ctx.arc(0, -10, 2.6, 0, Math.PI * 2);
      ctx.arc(8, -4, 2.0, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.moveTo(-8, 4);
      ctx.lineTo(8, 4);
      ctx.stroke();
    }

    ctx.restore();

    /* detached suit piece drifting in space */
    const sx = s * (1.95 + Math.sin(astronaut.suitDrift) * 0.28);
    const sy = s * (-0.58 + Math.cos(astronaut.suitDrift * 0.9) * 0.20);
    ctx.save();
    ctx.translate(sx, sy);
    ctx.rotate(Math.sin(astronaut.suitDrift) * 0.38);
    const pieceGrad = ctx.createLinearGradient(-s * 0.22, -s * 0.18, s * 0.22, s * 0.18);
    pieceGrad.addColorStop(0, `rgba(${bodyColor[0]},${bodyColor[1]},${bodyColor[2]},0.88)`);
    pieceGrad.addColorStop(1, `rgba(${Math.max(0, bodyColor[0] - 70)},${Math.max(0, bodyColor[1] - 40)},${Math.max(0, bodyColor[2] - 30)},0.88)`);
    ctx.fillStyle = pieceGrad;
    roundedRectPath(ctx, -s * 0.24, -s * 0.18, s * 0.50, s * 0.36, 2.2);
    ctx.fill();
    ctx.strokeStyle = `rgba(${trimColor[0]},${trimColor[1]},${trimColor[2]},${light ? 0.36 : 0.56})`;
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

    /* torso */
    const torsoGrad = ctx.createLinearGradient(-s * 0.44, -s * 0.06, s * 0.36, s * 1.20);
    torsoGrad.addColorStop(0, `rgba(${bodyColor[0]},${bodyColor[1]},${bodyColor[2]},0.97)`);
    torsoGrad.addColorStop(0.52, `rgba(${Math.max(0, bodyColor[0] - 22)},${Math.max(0, bodyColor[1] - 16)},${Math.max(0, bodyColor[2] - 12)},0.97)`);
    torsoGrad.addColorStop(1, `rgba(${Math.max(0, bodyColor[0] - 46)},${Math.max(0, bodyColor[1] - 32)},${Math.max(0, bodyColor[2] - 24)},0.97)`);
    ctx.fillStyle = torsoGrad;
    ctx.beginPath();
    ctx.ellipse(-s * 0.02, s * 0.36, s * 0.52, s * 0.88, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = `rgba(${trimColor[0]},${trimColor[1]},${trimColor[2]},${light ? 0.72 : 0.88})`;
    ctx.lineWidth = 1.2;
    ctx.stroke();

    /* high-visibility suit bands */
    ctx.fillStyle = `rgba(${bandColor[0]},${bandColor[1]},${bandColor[2]},${light ? 0.88 : 0.92})`;
    roundedRectPath(ctx, -s * 0.34, s * 0.08, s * 0.62, s * 0.08, 1.6);
    ctx.fill();
    roundedRectPath(ctx, -s * 0.20, s * 0.78, s * 0.36, s * 0.07, 1.4);
    ctx.fill();

    ctx.strokeStyle = `rgba(${bandColor[0]},${bandColor[1]},${bandColor[2]},${light ? 0.82 : 0.92})`;
    ctx.lineWidth = 1.3;
    ctx.beginPath();
    ctx.arc(-s * 0.30, s * 0.06, s * 0.10, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(s * 0.30, s * 0.06, s * 0.10, 0, Math.PI * 2);
    ctx.stroke();

    /* chest details */
    ctx.fillStyle = `rgba(${Math.max(0, bodyColor[0] - 96)},${Math.max(0, bodyColor[1] - 82)},${Math.max(0, bodyColor[2] - 72)},0.72)`;
    roundedRectPath(ctx, -s * 0.20, s * 0.26, s * 0.26, s * 0.18, 2.2);
    ctx.fill();
    ctx.fillStyle = `rgba(${trimColor[0]},${trimColor[1]},${trimColor[2]},${light ? 0.75 : 0.86})`;
    ctx.beginPath();
    ctx.arc(-s * 0.10, s * 0.35, s * 0.03, 0, Math.PI * 2);
    ctx.fill();

    /* helmet shell */
    const helmetShell = ctx.createRadialGradient(-s * 0.06, -s * 0.50, s * 0.05, 0, -s * 0.36, s * 0.64);
    helmetShell.addColorStop(0, `rgba(${Math.min(255, bodyColor[0] + 8)},${Math.min(255, bodyColor[1] + 10)},${Math.min(255, bodyColor[2] + 10)},0.98)`);
    helmetShell.addColorStop(1, `rgba(${Math.max(0, bodyColor[0] - 52)},${Math.max(0, bodyColor[1] - 44)},${Math.max(0, bodyColor[2] - 36)},0.96)`);
    ctx.fillStyle = helmetShell;
    ctx.beginPath();
    ctx.arc(0, -s * 0.38, s * 0.60, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = `rgba(${trimColor[0]},${trimColor[1]},${trimColor[2]},${light ? 0.64 : 0.76})`;
    ctx.lineWidth = 1.0;
    ctx.stroke();

    ctx.strokeStyle = `rgba(${bandColor[0]},${bandColor[1]},${bandColor[2]},${light ? 0.72 : 0.82})`;
    ctx.lineWidth = 1.0;
    ctx.beginPath();
    ctx.arc(0, -s * 0.38, s * 0.60, 0.12 * Math.PI, 0.88 * Math.PI);
    ctx.stroke();

    /* visor */
    const visorGrad = ctx.createRadialGradient(-s * 0.12, -s * 0.52, s * 0.02, 0, -s * 0.40, s * 0.38);
    if (isNight) {
      visorGrad.addColorStop(0, light ? `rgba(${visorBase[0] + 70},${visorBase[1] + 70},${visorBase[2] + 70},0.84)` : `rgba(${visorBase[0] + 84},${visorBase[1] + 84},${visorBase[2] + 84},0.84)`);
      visorGrad.addColorStop(0.58, light ? `rgba(${visorBase[0]},${visorBase[1]},${visorBase[2]},0.88)` : `rgba(${Math.max(0, visorBase[0] - 22)},${Math.max(0, visorBase[1] - 20)},${Math.max(0, visorBase[2] - 20)},0.88)`);
      visorGrad.addColorStop(1, light ? `rgba(${Math.max(0, visorBase[0] - 32)},${Math.max(0, visorBase[1] - 36)},${Math.max(0, visorBase[2] - 42)},0.90)` : `rgba(${Math.max(0, visorBase[0] - 44)},${Math.max(0, visorBase[1] - 46)},${Math.max(0, visorBase[2] - 52)},0.92)`);
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
    ctx.strokeStyle = `rgba(${Math.min(255, bodyColor[0] + 8)},${Math.min(255, bodyColor[1] + 8)},${Math.min(255, bodyColor[2] + 8)},${light ? 0.92 : 0.94})`;
    ctx.lineWidth = 2.6;
    ctx.lineCap = 'round';
    const armLift = astronaut.mood === 'excited' ? -s * 0.05 : astronaut.mood === 'sleepy' ? s * 0.06 : astronaut.mood === 'curious' ? -s * 0.02 : 0;
    const legBend = astronaut.mood === 'excited' ? s * 0.10 : astronaut.mood === 'sleepy' ? -s * 0.04 : astronaut.mood === 'curious' ? s * 0.04 : 0;

    /* left arm baseline */
    ctx.beginPath();
    ctx.moveTo(-s * 0.22, s * 0.04 + armLift * 0.4);
    ctx.lineTo(-s * 0.60, (gesture === 'salute' ? s * 0.10 : -s * 0.02) + armLift);
    ctx.lineTo(-s * 1.02, (gesture === 'salute' ? s * 0.28 : s * 0.10) + armLift * 1.1);
    ctx.stroke();

    if (gesture === 'thumbs') {
      ctx.save();
      ctx.translate(s * 0.24, s * 0.03 + armLift * 0.45);
      ctx.rotate(-1.42 + Math.sin(astronaut.wavePhase) * 0.14);
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(s * 0.42, -s * 0.01);
      ctx.lineTo(s * 0.70, s * 0.08);
      ctx.stroke();
      ctx.fillStyle = light ? 'rgba(228,240,246,0.98)' : 'rgba(222,236,246,0.98)';
      ctx.beginPath(); ctx.arc(s * 0.72, s * 0.06, s * 0.09, 0, Math.PI * 2); ctx.fill();
      ctx.beginPath(); ctx.arc(s * 0.78, -s * 0.03, s * 0.05, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
    } else if (gesture === 'salute') {
      ctx.beginPath();
      ctx.moveTo(s * 0.23, s * 0.03 + armLift * 0.35);
      ctx.lineTo(s * 0.52, -s * 0.20);
      ctx.lineTo(s * 0.30, -s * 0.42);
      ctx.stroke();
      ctx.fillStyle = light ? 'rgba(228,240,246,0.98)' : 'rgba(222,236,246,0.98)';
      ctx.beginPath(); ctx.arc(s * 0.30, -s * 0.44, s * 0.09, 0, Math.PI * 2); ctx.fill();
    } else if (gesture === 'visor') {
      ctx.beginPath();
      ctx.moveTo(s * 0.24, s * 0.04 + armLift * 0.35);
      ctx.lineTo(s * 0.42, -s * 0.28);
      ctx.lineTo(s * 0.02, -s * 0.50);
      ctx.stroke();
      ctx.fillStyle = light ? 'rgba(228,240,246,0.98)' : 'rgba(222,236,246,0.98)';
      ctx.beginPath(); ctx.arc(-s * 0.02, -s * 0.50, s * 0.09, 0, Math.PI * 2); ctx.fill();
    } else {
      const waveA = -0.95 + Math.sin(astronaut.wavePhase) * (gesture === 'spin' ? 0.30 : 0.50);
      ctx.save();
      ctx.translate(s * 0.25, s * 0.04 + armLift * 0.35);
      ctx.rotate(waveA);
      ctx.beginPath();
      ctx.moveTo(0, 0);
      ctx.lineTo(s * 0.42, -s * 0.02);
      ctx.lineTo(s * 0.84, s * 0.16);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(s * 0.88, s * 0.16, s * 0.10, 0, Math.PI * 2);
      ctx.fillStyle = light ? 'rgba(228,240,246,0.98)' : 'rgba(222,236,246,0.98)';
      ctx.fill();
      ctx.restore();
    }

    /* legs and boots */
    ctx.beginPath();
    ctx.moveTo(-s * 0.16, s * 1.02);
    ctx.lineTo(-s * 0.30, s * 1.58 - legBend * personality.motion.leg);
    ctx.moveTo(s * 0.16, s * 1.02);
    ctx.lineTo(s * 0.30, s * 1.58 + legBend * personality.motion.leg * 0.6);
    ctx.stroke();

    ctx.fillStyle = `rgba(${Math.max(0, bodyColor[0] - 60)},${Math.max(0, bodyColor[1] - 60)},${Math.max(0, bodyColor[2] - 52)},0.96)`;
    roundedRectPath(ctx, -s * 0.36, s * (1.50 - legBend * 0.08), s * 0.28, s * 0.14, 2.2);
    ctx.fill();
    roundedRectPath(ctx, s * 0.08, s * (1.50 + legBend * 0.05), s * 0.28, s * 0.14, 2.2);
    ctx.fill();

    /* tiny RCS thruster blink */
    const thrusterA = 0.10 + 0.12 * (0.5 + 0.5 * Math.sin(astronaut.suitDrift * 3.4));
    ctx.fillStyle = light
      ? `rgba(0,145,186,${thrusterA.toFixed(3)})`
      : `rgba(132,232,255,${thrusterA.toFixed(3)})`;
    ctx.beginPath();
    ctx.arc(-s * 0.56, s * 0.44, s * 0.05, 0, Math.PI * 2);
    ctx.fill();

    const travelSpeed = Math.hypot(astronaut.vx, astronaut.vy);
    if (astronaut.movementMode !== 'locked' && travelSpeed > 0.025) {
      const dirX = astronaut.vx / (travelSpeed + 0.0001);
      const dirY = astronaut.vy / (travelSpeed + 0.0001);
      const thrust = clamp(travelSpeed / 0.34, 0.35, 1.0);
      const sideDriftX = -dirY * s * 0.10;
      const sideDriftY = dirX * s * 0.10;
      const suitExhaustX = -dirX * s * 0.18 + sideDriftX;
      const suitExhaustY = s * 0.28 - dirY * s * 0.18 + sideDriftY;
      ctx.strokeStyle = light
        ? `rgba(0,170,210, ${(0.24 + thrust * 0.24).toFixed(3)})`
        : `rgba(126,244,255, ${(0.30 + thrust * 0.30).toFixed(3)})`;
      ctx.lineWidth = s * (0.035 + thrust * 0.012);
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(suitExhaustX, suitExhaustY);
      ctx.lineTo(
        suitExhaustX - dirX * s * (0.34 + thrust * 0.20),
        suitExhaustY - dirY * s * (0.34 + thrust * 0.20)
      );
      ctx.stroke();

      for (let i = 0; i < 5; i++) {
        const puff = 1 - i / 3;
        const ox = suitExhaustX - dirX * s * (0.10 + i * 0.18 + thrust * 0.12) + Math.sin(time * 10 + i) * s * 0.035;
        const oy = suitExhaustY - dirY * s * (0.08 + i * 0.16 + thrust * 0.10) + Math.cos(time * 9 + i) * s * 0.035;
        ctx.fillStyle = light
          ? `rgba(255,176,96, ${(0.30 + puff * 0.24 * thrust).toFixed(3)})`
          : `rgba(255,204,112, ${(0.36 + puff * 0.28 * thrust).toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(ox, oy, s * (0.07 + puff * 0.06 * thrust), 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = light
          ? `rgba(0,148,182, ${(0.22 + puff * 0.14 * thrust).toFixed(3)})`
          : `rgba(110,244,255, ${(0.26 + puff * 0.16 * thrust).toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(ox - dirX * s * 0.11, oy - dirY * s * 0.11, s * (0.05 + puff * 0.04 * thrust), 0, Math.PI * 2);
        ctx.fill();
      }
    }

    ctx.restore();

    /* speech cloud */
    const ba = astronaut.bubble.alpha;
    if (ba > 0.01 && astronaut.bubble.text) {
      const text = astronaut.bubble.text;
      const metaPrompt = astronaut.bubble.metaPrompt || 'astro@orbit:~ %';
      const statusRows = Array.isArray(astronaut.bubble.statusRows) ? astronaut.bubble.statusRows : [];
      const category = astronaut.bubble.category || 'auto';
      ctx.save();
      const padX = 10;
      const padY = 8;
      const promptLineH = 10;
      const statusLineH = 9;
      const lineH = 12;
      const maxBubbleW = clamp(W * 0.22, 132, 168);
      const metaInset = category === 'brand' || category === 'marketing' ? 14 : 0;
      const maxTextW = Math.max(98, maxBubbleW - padX * 2 - metaInset);
      const promptFont = '700 8px Space Mono, monospace';
      const statusFont = '700 7.1px Space Mono, monospace';
      const bodyFont = '700 9.35px Space Mono, monospace';
      const wrapBubbleText = (input, font, limit, maxLineCount) => {
        ctx.font = font;
        const wordsIn = input.trim().split(/\s+/).filter(Boolean);
        const wrapped = [];
        let current = '';

        for (const word of wordsIn) {
          const test = current ? `${current} ${word}` : word;
          if (ctx.measureText(test).width <= limit) {
            current = test;
            continue;
          }

          if (current) wrapped.push(current);

          if (ctx.measureText(word).width > limit) {
            let chunk = '';
            for (const ch of word) {
              const chunkTest = chunk + ch;
              if (ctx.measureText(chunkTest).width <= limit) {
                chunk = chunkTest;
              } else {
                wrapped.push(chunk);
                chunk = ch;
              }
            }
            current = chunk;
          } else {
            current = word;
          }
        }
        if (current) wrapped.push(current);

        if (wrapped.length <= maxLineCount) return wrapped;

        const clipped = wrapped.slice(0, maxLineCount);
        let trimmed = clipped[maxLineCount - 1];
        while (trimmed.length > 0 && ctx.measureText(`${trimmed}...`).width > limit) {
          trimmed = trimmed.slice(0, -1);
        }
        clipped[maxLineCount - 1] = `${trimmed}...`;
        return clipped;
      };

      const maxLines = 6;
      const promptLines = wrapBubbleText(metaPrompt, promptFont, maxTextW, 2);
      const fullBubbleLines = wrapBubbleText(text, bodyFont, maxTextW, maxLines);
      const visibleText = text.slice(0, astronaut.bubble.typingChars);
      const bubbleLines = wrapBubbleText(visibleText || ' ', bodyFont, maxTextW, maxLines);

      ctx.font = promptFont;
      let textW = metaInset;
      for (const promptLine of promptLines) textW = Math.max(textW, ctx.measureText(promptLine).width + metaInset);
      ctx.font = statusFont;
      for (const row of statusRows) {
        textW = Math.max(textW, ctx.measureText(`${row.label}: ${row.value}`).width + metaInset);
      }
      ctx.font = bodyFont;
      for (const ln of fullBubbleLines) textW = Math.max(textW, ctx.measureText(ln).width);
      const tw = clamp(textW + padX * 2 + metaInset, 128, maxBubbleW);
      const promptBlockH = promptLineH * Math.max(1, promptLines.length);
      const statusBlockH = statusRows.length * statusLineH;
      const bodyOffsetY = padY + promptBlockH + statusBlockH + 8;
      const th = Math.max(68, bodyOffsetY + padY + lineH * Math.max(1, fullBubbleLines.length));
      const margin = 8;
      const gap = Math.max(18, s * 0.48);
      const astroLeft = x - s * 0.92;
      const astroRight = x + s * 0.92;
      const astroTop = y - s * 1.92;
      const astroBottom = y + s * 1.68;
      const roomRight = W - astroRight - margin;
      const roomLeft = astroLeft - margin;
      let placeRight = x >= W * 0.5;
      if (placeRight && roomRight < tw + gap && roomLeft > roomRight) placeRight = false;
      if (!placeRight && roomLeft < tw + gap && roomRight > roomLeft) placeRight = true;

      let bx = placeRight
        ? astroRight + gap
        : astroLeft - gap - tw;
      bx = clamp(bx, margin, W - tw - margin);

      let by = y - th * 0.62;
      const overlapsAstro = by + th > astroTop - 6 && by < astroBottom + 6;
      if (overlapsAstro) {
        const aboveY = astroTop - th - gap;
        const belowY = astroBottom + gap;
        const canPlaceAbove = aboveY >= margin;
        const canPlaceBelow = belowY + th <= H - margin;
        if (canPlaceAbove || (!canPlaceBelow && astroTop > H - astroBottom)) {
          by = Math.max(margin, aboveY);
        } else if (canPlaceBelow) {
          by = belowY;
        }
      }
      by = clamp(by, margin, H - th - margin);

      const bubbleMidY = by + th * 0.5;
      const tailOnRight = x >= bx + tw * 0.5;
      const tailAnchorX = tailOnRight ? bx + tw : bx;
      const tailAnchorY = clamp(bubbleMidY, by + 12, by + th - 12);
      const tailTipX = x + (tailOnRight ? -s * 0.56 : s * 0.56);
      const tailTipY = clamp(y - s * 0.18, astroTop + 12, astroBottom - 14);
      const borderColor = light
        ? `rgba(0,126,136, ${(0.60 * ba).toFixed(3)})`
        : `rgba(90,238,174, ${(0.74 * ba).toFixed(3)})`;
      const glowColor = light
        ? `rgba(0,208,180, ${(0.16 * ba).toFixed(3)})`
        : `rgba(74,255,200, ${(0.22 * ba).toFixed(3)})`;
      const panelGradient = ctx.createLinearGradient(bx, by, bx, by + th);
      if (light) {
        panelGradient.addColorStop(0, `rgba(224,246,240, ${(0.88 * ba).toFixed(3)})`);
        panelGradient.addColorStop(0.52, `rgba(196,232,226, ${(0.78 * ba).toFixed(3)})`);
        panelGradient.addColorStop(1, `rgba(168,214,206, ${(0.72 * ba).toFixed(3)})`);
      } else {
        panelGradient.addColorStop(0, `rgba(4,14,12, ${(0.94 * ba).toFixed(3)})`);
        panelGradient.addColorStop(0.54, `rgba(8,22,18, ${(0.90 * ba).toFixed(3)})`);
        panelGradient.addColorStop(1, `rgba(2,10,8, ${(0.86 * ba).toFixed(3)})`);
      }
      const sheenGradient = ctx.createLinearGradient(bx, by, bx + tw, by + th);
      sheenGradient.addColorStop(0, light
        ? `rgba(255,255,255, ${(0.20 * ba).toFixed(3)})`
        : `rgba(120,255,200, ${(0.08 * ba).toFixed(3)})`);
      sheenGradient.addColorStop(0.35, 'rgba(255,255,255, 0)');
      sheenGradient.addColorStop(1, light
        ? `rgba(0,172,146, ${(0.08 * ba).toFixed(3)})`
        : `rgba(0,210,166, ${(0.10 * ba).toFixed(3)})`);
      const headerGlow = light
        ? `rgba(0,128,118, ${(0.12 * ba).toFixed(3)})`
        : `rgba(40,255,168, ${(0.14 * ba).toFixed(3)})`;
      const scanColor = light
        ? `rgba(0,122,112, ${(0.05 * ba).toFixed(3)})`
        : `rgba(88,244,168, ${(0.06 * ba).toFixed(3)})`;
      const panelRadius = 4;

      ctx.save();
      roundedRectPath(ctx, bx, by, tw, th, panelRadius);
      ctx.clip();
      ctx.fillStyle = panelGradient;
      ctx.fillRect(bx, by, tw, th);
      ctx.fillStyle = sheenGradient;
      ctx.fillRect(bx, by, tw, th);
      ctx.fillStyle = headerGlow;
      ctx.fillRect(bx, by, tw, 14);
      ctx.fillStyle = scanColor;
      for (let sy = by + 20; sy < by + th - 6; sy += 8) {
        ctx.fillRect(bx + 6, sy, tw - 12, 1);
      }
      ctx.restore();

      ctx.strokeStyle = borderColor;
      ctx.lineWidth = 1.15;
      ctx.shadowBlur = 12;
      ctx.shadowColor = glowColor;
      roundedRectPath(ctx, bx, by, tw, th, panelRadius);
      ctx.stroke();
      ctx.shadowBlur = 0;

      ctx.strokeStyle = light
        ? `rgba(255,255,255, ${(0.34 * ba).toFixed(3)})`
        : `rgba(176,250,255, ${(0.24 * ba).toFixed(3)})`;
      ctx.lineWidth = 1;
      roundedRectPath(ctx, bx + 2, by + 2, tw - 4, th - 4, 2.5);
      ctx.stroke();

      const cornerLen = 7;
      ctx.strokeStyle = borderColor;
      ctx.lineWidth = 1.15;
      ctx.beginPath();
      ctx.moveTo(bx + 7, by + cornerLen);
      ctx.lineTo(bx + 7, by + 4);
      ctx.lineTo(bx + 7 + cornerLen, by + 4);
      ctx.moveTo(bx + tw - 7 - cornerLen, by + 4);
      ctx.lineTo(bx + tw - 7, by + 4);
      ctx.lineTo(bx + tw - 7, by + cornerLen);
      ctx.moveTo(bx + 7, by + th - cornerLen);
      ctx.lineTo(bx + 7, by + th - 4);
      ctx.lineTo(bx + 7 + cornerLen, by + th - 4);
      ctx.moveTo(bx + tw - 7 - cornerLen, by + th - 4);
      ctx.lineTo(bx + tw - 7, by + th - 4);
      ctx.lineTo(bx + tw - 7, by + th - cornerLen);
      ctx.stroke();

      ctx.fillStyle = light
        ? `rgba(0,126,136, ${(0.16 * ba).toFixed(3)})`
        : `rgba(90,238,174, ${(0.18 * ba).toFixed(3)})`;
      ctx.fillRect(bx + 8, by + 7, Math.min(18, tw * 0.22), 2);
      ctx.fillRect(bx + tw - 18, by + 7, 8, 2);
      ctx.fillRect(bx + 8, by + 17, tw - 16, 1);

      if (category === 'brand' || category === 'marketing') {
        const iconCx = bx + 13;
        const iconCy = by + 13;
        const pulse = 0.55 + 0.45 * Math.sin(time * 6.0);
        ctx.strokeStyle = category === 'brand'
          ? `rgba(0,212,170, ${(0.65 * ba * pulse).toFixed(3)})`
          : `rgba(255,156,64, ${(0.65 * ba * pulse).toFixed(3)})`;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.arc(iconCx, iconCy, 5.2, 0, Math.PI * 2);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(iconCx - 3.5, iconCy);
        ctx.lineTo(iconCx + 3.5, iconCy);
        ctx.moveTo(iconCx, iconCy - 3.5);
        ctx.lineTo(iconCx, iconCy + 3.5);
        ctx.stroke();
      }

      ctx.beginPath();
      if (tailOnRight) {
        ctx.moveTo(tailAnchorX, tailAnchorY - 8);
        ctx.lineTo(tailAnchorX, tailAnchorY + 8);
      } else {
        ctx.moveTo(tailAnchorX, tailAnchorY + 8);
        ctx.lineTo(tailAnchorX, tailAnchorY - 8);
      }
      ctx.lineTo(tailTipX, tailTipY);
      ctx.closePath();
      ctx.fillStyle = borderColor;
      ctx.shadowBlur = 8;
      ctx.shadowColor = glowColor;
      ctx.fill();
      ctx.shadowBlur = 0;

      if (category === 'marketing') {
        ctx.fillStyle = light
          ? `rgba(0,126,152, ${(0.84 * ba).toFixed(3)})`
          : `rgba(255,178,96, ${(0.86 * ba).toFixed(3)})`;
        ctx.beginPath();
        ctx.arc(bx + tw - 14, by + 14, 2.2, 0, Math.PI * 2);
        ctx.fill();
      }

      const textX = category === 'brand' || category === 'marketing' ? bx + padX + 14 : bx + padX;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      ctx.font = promptFont;
      ctx.fillStyle = light
        ? `rgba(0,94,96, ${(0.92 * ba).toFixed(3)})`
        : `rgba(168,255,202, ${(0.96 * ba).toFixed(3)})`;
      for (let i = 0; i < promptLines.length; i++) {
        ctx.fillText(promptLines[i], textX, by + padY + i * promptLineH);
      }

      const toneColor = (tone) => {
        if (tone === 'mood') {
          return astronaut.mood === 'excited'
            ? `rgba(255,166,92, ${(0.96 * ba).toFixed(3)})`
            : astronaut.mood === 'curious'
              ? `rgba(88,204,255, ${(0.96 * ba).toFixed(3)})`
              : astronaut.mood === 'sleepy'
                ? `rgba(154,174,228, ${(0.96 * ba).toFixed(3)})`
                : `rgba(96,255,184, ${(0.96 * ba).toFixed(3)})`;
        }
        if (tone === 'persona') return `rgba(255,122,214, ${(0.94 * ba).toFixed(3)})`;
        if (tone === 'tag') return `rgba(255,210,104, ${(0.94 * ba).toFixed(3)})`;
        return `rgba(118,255,214, ${(0.94 * ba).toFixed(3)})`;
      };

      const statusY = by + padY + promptBlockH + 2;
      ctx.font = statusFont;
      for (let i = 0; i < statusRows.length; i++) {
        const row = statusRows[i];
        const rowY = statusY + i * statusLineH;
        ctx.fillStyle = light
          ? `rgba(0,88,84, ${(0.82 * ba).toFixed(3)})`
          : `rgba(126,212,166, ${(0.84 * ba).toFixed(3)})`;
        const labelText = `${row.label}: `;
        ctx.fillText(labelText, textX, rowY);
        const labelW = ctx.measureText(labelText).width;
        ctx.fillStyle = toneColor(row.tone);
        ctx.fillText(row.value, textX + labelW, rowY);
      }

      ctx.fillStyle = light
        ? `rgba(0,116,108, ${(0.18 * ba).toFixed(3)})`
        : `rgba(86,240,172, ${(0.22 * ba).toFixed(3)})`;
      ctx.fillRect(textX, by + padY + promptBlockH + statusBlockH + 2, tw - (textX - bx) - padX, 1);

      ctx.fillStyle = light
        ? `rgba(0,78,74, ${(0.96 * ba).toFixed(3)})`
        : `rgba(180,255,214, ${(0.98 * ba).toFixed(3)})`;
      ctx.font = bodyFont;
      for (let i = 0; i < bubbleLines.length; i++) {
        ctx.fillText(bubbleLines[i], textX, by + bodyOffsetY + i * lineH);
      }

      if (astronaut.bubble.typingChars < text.length && Math.sin(time * 8.0) > -0.1) {
        const lastLine = bubbleLines[Math.max(0, bubbleLines.length - 1)] || '';
        const cursorY = by + bodyOffsetY + (bubbleLines.length - 1) * lineH;
        const cursorX = textX + ctx.measureText(lastLine).width + 2;
        ctx.fillRect(cursorX, cursorY + 1, 5, 9);
      }
      ctx.restore();
    }
  };

  const makeBirthParticles = (x, y, targetR, count = 26) => {
    const parts = [];
    for (let i = 0; i < count; i++) {
      const a = rand(0, Math.PI * 2);
      const d = rand(targetR * 2.6, targetR * 5.4);
      parts.push({
        x: x + Math.cos(a) * d,
        y: y + Math.sin(a) * d,
        vx: Math.cos(a + Math.PI * 0.5) * rand(0.02, 0.10),
        vy: Math.sin(a + Math.PI * 0.5) * rand(0.02, 0.10),
        alpha: rand(0.22, 0.55),
        size: rand(0.45, 1.35),
      });
    }
    return parts;
  };

  const startHoleBirth = (bh, keepPosition = true) => {
    if (!keepPosition) {
      bh.x = rand(W * 0.08, W * 0.92);
      bh.y = rand(H * 0.08, H * 0.92);
    }
    bh.state = 'birthing';
    bh.birthDurationSec = rand(2.2, 3.3);
    bh.birthTimerSec = bh.birthDurationSec;
    bh.renderAlpha = 0;
    bh.r = Math.max(1, bh.baseR * 0.08);
    bh.birthParticles = makeBirthParticles(bh.x, bh.y, bh.baseR, Math.floor(rand(22, 34)));
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
      lensPulse:     0,
      consumed:      0,
      capturedCount: 0,
      lifeSpanSec:   100,
      ageSec:        rand(0, 14),
      state:         'birthing',      /* 'birthing' | 'active' | 'fading' */
      birthTimerSec: 0,
      birthDurationSec: 0,
      birthParticles: [],
      fadeTimerSec:  0,
      fadeDurationSec: 2.8,
      renderAlpha:   1,
      ripples:       [],
      rippleTimerSec: rand(2.8, 5.2),
    };
  };

  const initMobHole = (bh) => {
    startHoleBirth(bh, true);
    return bh;
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

  const triggerHoleExplosion = (bh, options = {}) => {
    if (!bh || bh.state !== 'active') return;

    const fromClick = !!options.fromClick;

    bh.state = 'fading';
    bh.fadeDurationSec = fromClick ? 0.65 : 2.8;
    bh.fadeTimerSec = bh.fadeDurationSec;
    bh.renderAlpha = fromClick ? 0.02 : Math.min(1, bh.renderAlpha || 1);

    const rippleBurst = fromClick ? 3 : 1;
    for (let i = 0; i < rippleBurst; i++) {
      bh.ripples.push({
        r: bh.r * (1.05 + i * 0.18),
        speed: Math.min(W, H) * (fromClick ? rand(0.075, 0.110) : rand(0.050, 0.080)),
        alpha: fromClick ? (isLight() ? 0.12 : 0.18) : (isLight() ? 0.08 : 0.11),
        band: bh.r * (fromClick ? rand(2.0, 3.0) : rand(1.6, 2.4)),
      });
    }

    /* release all orbiting comets outward (slower repulsion) */
    for (const p of comets) {
      if (p.capturedBy === bh) {
        const oa = (p.orbitAngle || 0) + rand(-0.9, 0.9);
        p.vx = Math.cos(oa) * rand(0.06, 0.24);
        p.vy = Math.sin(oa) * rand(0.06, 0.24);
        p.releaseSlowSec = fromClick ? 2.4 : 1.8;
        p.capturedBy = null;
        p.trail = [];
      }
    }
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

      if (mobHoles.includes(w) && w.state === 'active') {
        const lensZone = w.r * 8.5;
        if (d < lensZone) {
          const t = clamp(1 - d / lensZone, 0, 1);
          const tx = -(ey / d);
          const ty = (ex / d);
          const swirl = w.mass * (0.8 + t * 2.2) * (0.55 + t * 0.45);
          dx += tx * swirl;
          dy += ty * swirl;
        }

        /* sparse ripple fronts: low-contrast radial deformation */
        if (w.ripples && w.ripples.length) {
          for (const rp of w.ripples) {
            const band = rp.band;
            const distToFront = Math.abs(d - rp.r);
            if (distToFront > band) continue;
            const t = 1 - distToFront / band;
            const radial = t * (rp.alpha || 0) * w.mass * 1.25;
            dx += (ex / d) * radial;
            dy += (ey / d) * radial;
          }
        }
      }
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

    const activeHoles = mobHoles.filter(bh => bh.state === 'active');
    for (let i = 0; i < activeHoles.length; i++) {
      for (let j = i + 1; j < activeHoles.length; j++) {
        const a = activeHoles[i];
        const b = activeHoles[j];
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const d = Math.hypot(dx, dy) + 0.001;
        const nx = dx / d;
        const ny = dy / d;
        const tx = -ny;
        const ty = nx;

        const influenceR = (a.r + b.r) * 8;
        if (d < influenceR) {
          const t = clamp((influenceR - d) / influenceR, 0, 1);
          const g = ((a.mass + b.mass) * 0.009 * t) / (1 + d * 0.012);
          a.vx += nx * g * frameScale;
          a.vy += ny * g * frameScale;
          b.vx -= nx * g * frameScale;
          b.vy -= ny * g * frameScale;

          const orbitKick = (0.003 + t * 0.010) * frameScale;
          a.vx += tx * orbitKick;
          a.vy += ty * orbitKick;
          b.vx -= tx * orbitKick;
          b.vy -= ty * orbitKick;

          a.lensPulse = Math.max(a.lensPulse, t * 0.9);
          b.lensPulse = Math.max(b.lensPulse, t * 0.9);
        }

        const minSep = (a.r + b.r) * 1.2;
        if (d < minSep) {
          const push = (minSep - d) * 0.5;
          a.x -= nx * push;
          a.y -= ny * push;
          b.x += nx * push;
          b.y += ny * push;
        }
      }
    }

    for (const bh of mobHoles) {

      /* ── BIRTHING: seed particles converge, then hole stabilizes ── */
      if (bh.state === 'birthing') {
        bh.birthTimerSec -= dtSec;
        const bornT = clamp(1 - (bh.birthTimerSec / Math.max(0.001, bh.birthDurationSec)), 0, 1);
        const ease = bornT * bornT * (3 - 2 * bornT);
        bh.renderAlpha = ease;
        bh.r = Math.max(1, bh.baseR * (0.08 + 0.92 * ease));

        if (bh.birthParticles && bh.birthParticles.length) {
          for (let pi = bh.birthParticles.length - 1; pi >= 0; pi--) {
            const sp = bh.birthParticles[pi];
            const ex = bh.x - sp.x;
            const ey = bh.y - sp.y;
            const d = Math.hypot(ex, ey) + 0.001;
            const nx = ex / d;
            const ny = ey / d;
            const tx = -ny;
            const ty = nx;
            const pull = (0.030 + (1 - bornT) * 0.020) * frameScale;
            sp.vx += nx * pull + tx * 0.004 * frameScale;
            sp.vy += ny * pull + ty * 0.004 * frameScale;
            sp.vx *= 0.965;
            sp.vy *= 0.965;
            sp.x += sp.vx * frameScale;
            sp.y += sp.vy * frameScale;
            sp.alpha = Math.max(0, sp.alpha - dtSec * 0.14);
            if (d < bh.r * 0.62 || sp.alpha <= 0.01) {
              bh.birthParticles.splice(pi, 1);
            }
          }
        }

        if (bh.birthTimerSec <= 0) {
          bh.state = 'active';
          bh.renderAlpha = 1;
          bh.r = bh.baseR;
          bh.birthParticles = [];
        }
        continue;
      }

      /* ── FADING: smooth decay, then clean respawn ── */
      if (bh.state === 'fading') {
        bh.fadeTimerSec -= dtSec;
        const tFade = clamp(bh.fadeTimerSec / Math.max(0.001, bh.fadeDurationSec), 0, 1);
        bh.renderAlpha = tFade;
        bh.r *= 1 + dtSec * 0.06;
        if (bh.fadeTimerSec <= 0) {
          /* respawn at fresh random position */
          const a2 = rand(0, Math.PI * 2), s2 = rand(0.12, 0.38);
          bh.x = rand(W * 0.08, W * 0.92);  bh.y = rand(H * 0.08, H * 0.92);
          bh.vx = Math.cos(a2) * s2;        bh.vy = Math.sin(a2) * s2;
          bh.turn = rand(-0.005, 0.005);
          bh.lensPulse = 0;
          bh.consumed = 0;
          bh.capturedCount = 0;
          bh.ageSec = 0;
          bh.r = bh.baseR;
          bh.renderAlpha = 0;
          bh.ripples = [];
          bh.rippleTimerSec = rand(2.8, 5.2);
          startHoleBirth(bh, true);
        }
        continue;
      }

      /* ── ACTIVE: drift + lifecycle countdown ── */
      const angle = Math.atan2(bh.vy, bh.vx) + bh.turn * frameScale;
      const spd = clamp(Math.sqrt(bh.vx * bh.vx + bh.vy * bh.vy), 0.05, 0.70);
      bh.vx = Math.cos(angle) * spd;
      bh.vy = Math.sin(angle) * spd;
      bh.x += bh.vx * frameScale;
      bh.y += bh.vy * frameScale;
      bh.phase += 0.014 * frameScale;
      bh.lensPulse = Math.max(0, (bh.lensPulse || 0) - dtSec * 0.5);
      bh.renderAlpha = lerp(bh.renderAlpha || 0, 1, 0.05);

      bh.rippleTimerSec -= dtSec;
      if (bh.rippleTimerSec <= 0) {
        bh.ripples.push({
          r: bh.r * 1.15,
          speed: Math.min(W, H) * rand(0.050, 0.080),
          alpha: isLight() ? 0.08 : 0.11,
          band: bh.r * rand(1.6, 2.4),
        });
        bh.rippleTimerSec = rand(2.8, 5.2);
      }
      for (let ri = bh.ripples.length - 1; ri >= 0; ri--) {
        const rp = bh.ripples[ri];
        rp.r += rp.speed * dtSec;
        rp.alpha = Math.max(0, rp.alpha - dtSec * 0.030);
        if (rp.alpha <= 0.004 || rp.r > Math.hypot(W, H) * 0.55) {
          bh.ripples.splice(ri, 1);
        }
      }

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

  const drawMiniSolarSystems = () => {
    const light = isLight();
    for (const sys of solarSystems) {
      const cx = sys.x + Math.sin(time * sys.driftSpeedX + sys.driftPhaseX) * sys.driftAx;
      const cy = sys.y + Math.cos(time * sys.driftSpeedY + sys.driftPhaseY) * sys.driftAy;
      const pulse = 0.90 + 0.10 * Math.sin(time * 1.25 + sys.driftPhaseX * 0.7);

      const starGlow = ctx.createRadialGradient(cx, cy, 0, cx, cy, sys.starR * (4.2 + 0.8 * pulse));
      if (light) {
        starGlow.addColorStop(0, `rgba(${sys.starCol[0]},${sys.starCol[1]},${sys.starCol[2]},${(0.23 + 0.10 * pulse).toFixed(3)})`);
        starGlow.addColorStop(1, 'transparent');
      } else {
        starGlow.addColorStop(0, `rgba(${sys.starCol[0]},${sys.starCol[1]},${sys.starCol[2]},${(0.30 + 0.12 * pulse).toFixed(3)})`);
        starGlow.addColorStop(1, 'transparent');
      }
      ctx.fillStyle = starGlow;
      ctx.beginPath();
      ctx.arc(cx, cy, sys.starR * (4.2 + 0.8 * pulse), 0, Math.PI * 2);
      ctx.fill();

      const starCoreAlpha = light ? (0.58 + 0.10 * pulse) : (0.78 + 0.10 * pulse);
      ctx.fillStyle = `rgba(${sys.starCol[0]},${sys.starCol[1]},${sys.starCol[2]},${starCoreAlpha.toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(cx, cy, sys.starR * (0.96 + 0.07 * pulse), 0, Math.PI * 2);
      ctx.fill();

      for (const p of sys.planets) {
        /* visible orbital trajectory */
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(p.orbitRot);
        ctx.scale(1, p.orbitYScale);
        ctx.strokeStyle = light ? 'rgba(0,120,130,0.18)' : 'rgba(130,210,220,0.24)';
        ctx.lineWidth = 0.62;
        ctx.beginPath();
        ctx.arc(0, 0, p.orbitR, 0, Math.PI * 2);
        ctx.stroke();
        ctx.restore();

        const a = p.phase + time * p.speed;
        const lx = Math.cos(a) * p.orbitR;
        const ly = Math.sin(a) * p.orbitR * p.orbitYScale;
        const cosR = Math.cos(p.orbitRot), sinR = Math.sin(p.orbitRot);
        const px = cx + lx * cosR - ly * sinR;
        const py = cy + lx * sinR + ly * cosR;

        ctx.fillStyle = `rgba(${p.col[0]},${p.col[1]},${p.col[2]},${light ? '0.55' : '0.72'})`;
        ctx.beginPath();
        ctx.arc(px, py, p.r, 0, Math.PI * 2);
        ctx.fill();
      }
    }
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
      let bhBoost = 0;
      for (const bh of mobHoles) {
        if (bh.state !== 'active') continue;
        const d = Math.hypot(gx - bh.x, gy - bh.y);
        const zone = bh.r * 9.5;
        if (d < zone) {
          const t = 1 - d / zone;
          bhBoost = Math.max(bhBoost, t * 0.22);
        }
      }
      const scale = Math.min(W, H) * 0.55;
      const t = clamp(Math.sqrt(minD2) / scale, 0, 1);
      const base = light
        ? lerp(0.22, 0.07, t)
        : lerp(0.18, 0.05, t);
      return clamp(base + bhBoost, 0, light ? 0.40 : 0.34);
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

    /* subtle bright grid nodes around black holes for local space-time shine */
    for (let i = 0; i <= cols; i++) {
      for (let j = 0; j <= rows; j++) {
        const gx = i * cw, gy = j * ch;
        let nodeA = 0;
        for (const bh of mobHoles) {
          if (bh.state !== 'active') continue;
          const d = Math.hypot(gx - bh.x, gy - bh.y);
          const zone = bh.r * 7.4;
          if (d < zone) {
            const t = 1 - d / zone;
            nodeA = Math.max(nodeA, (light ? 0.16 : 0.13) * t);
          }
        }
        if (nodeA < 0.02) continue;
        const { dx, dy } = warpAt(gx, gy);
        const px = gx + dx, py = gy + dy;
        ctx.fillStyle = rgba(CYAN, nodeA);
        ctx.beginPath();
        ctx.arc(px, py, light ? 0.65 : 0.55, 0, Math.PI * 2);
        ctx.fill();
      }
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

      if (bh.state === 'birthing') {
        const ba = clamp(bh.renderAlpha || 0, 0, 1);

        if (bh.birthParticles && bh.birthParticles.length) {
          for (const sp of bh.birthParticles) {
            const pa = sp.alpha * (light ? 0.75 : 1.0) * (0.65 + ba * 0.35);
            if (pa < 0.01) continue;
            ctx.fillStyle = rgba(CYAN, pa);
            ctx.beginPath();
            ctx.arc(sp.x, sp.y, sp.size, 0, Math.PI * 2);
            ctx.fill();
          }
        }

        const seedGlow = ctx.createRadialGradient(x, y, 0, x, y, r * 3.2);
        seedGlow.addColorStop(0, rgba(TEAL, (light ? 0.10 : 0.16) * ba));
        seedGlow.addColorStop(0.55, rgba(CYAN, (light ? 0.06 : 0.10) * ba));
        seedGlow.addColorStop(1, 'transparent');
        ctx.beginPath();
        ctx.arc(x, y, r * 3.2, 0, Math.PI * 2);
        ctx.fillStyle = seedGlow;
        ctx.fill();

        const seedCore = ctx.createRadialGradient(x, y, 0, x, y, r);
        seedCore.addColorStop(0, light ? 'rgba(0,0,0,0.88)' : 'rgba(0,4,10,0.84)');
        seedCore.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fillStyle = seedCore;
        ctx.fill();
        continue;
      }

      /* ── FADING: restrained decay visuals ── */
      if (bh.state === 'fading') {
        const fa = clamp(bh.renderAlpha || 0, 0, 1);
        if (fa <= 0.01) continue;
        ctx.save();
        ctx.globalAlpha = fa;
        const fadeGlow = ctx.createRadialGradient(x, y, r * 0.4, x, y, r * 6.2);
        fadeGlow.addColorStop(0, rgba(TEAL, light ? 0.10 : 0.14));
        fadeGlow.addColorStop(0.45, rgba(CYAN, light ? 0.06 : 0.09));
        fadeGlow.addColorStop(1, 'transparent');
        ctx.beginPath();
        ctx.arc(x, y, r * 6.2, 0, Math.PI * 2);
        ctx.fillStyle = fadeGlow;
        ctx.fill();
        ctx.restore();
      }

      /* ── ACTIVE ── */
      const pulse        = 0.82 + 0.18 * Math.sin(time * 0.9 + phase);
      const accreteBoost = 1 + (bh.capturedCount || 0) * 0.28; /* glow brightens as comets spiral in */
      const halfLife = bh.ageSec >= (bh.lifeSpanSec * 0.5);
      const lensBoost = 1 + (bh.lensPulse || 0) * 0.8;
      const alpha = clamp(bh.renderAlpha || 1, 0, 1);
      if (alpha <= 0.01) continue;

      ctx.save();
      ctx.globalAlpha = alpha;

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

      const einsteinA = (light ? 0.16 : 0.24) * mass * lensBoost;
      if (einsteinA > 0.02) {
        ctx.strokeStyle = rgba(CYAN, einsteinA);
        ctx.lineWidth = 0.9;
        ctx.beginPath();
        ctx.arc(x, y, r * 2.7 + Math.sin(time * 1.6 + phase) * 0.5, 0, Math.PI * 2);
        ctx.stroke();
      }

      if (bh.ripples && bh.ripples.length) {
        for (const rp of bh.ripples) {
          const ra = rp.alpha * (light ? 0.75 : 0.95);
          if (ra < 0.01) continue;
          ctx.strokeStyle = rgba(TEAL, ra);
          ctx.lineWidth = Math.max(0.18, rp.band * 0.06);
          ctx.beginPath();
          ctx.arc(x, y, rp.r, 0, Math.PI * 2);
          ctx.stroke();
        }
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

      ctx.restore();
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

        if (p.releaseSlowSec && p.releaseSlowSec > 0) {
          p.releaseSlowSec = Math.max(0, p.releaseSlowSec - 0.016);
          p.vx *= 0.965;
          p.vy *= 0.965;
        }

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

          /* ripple fronts add gentle path deformation */
          if (bh.ripples && bh.ripples.length) {
            for (const rp of bh.ripples) {
              const distToFront = Math.abs(d - rp.r);
              if (distToFront > rp.band) continue;
              const t = 1 - distToFront / rp.band;
              const rf = t * rp.alpha * bh.mass;
              p.vx += (ex / d) * rf * 0.040;
              p.vy += (ey / d) * rf * 0.040;
            }
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
    drawMiniSolarSystems();
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
        astronaut.gesture = Math.random() < 0.5 ? 'wave' : 'thumbs';
        astronaut.gestureTimerSec = rand(2.0, 3.4);
      }
      const text = ASTRONAUT_CLICK_PHRASES[Math.floor(Math.random() * ASTRONAUT_CLICK_PHRASES.length)];
      triggerAstronautBubble(text, 'click');
      return;
    }

    /* click black hole -> start smooth fade/respawn lifecycle */
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
    if (target) triggerHoleExplosion(target, { fromClick: true });

    /* each 3 clicks -> photon burst + reset 60s timer */
    burstClickCount++;
    if (burstClickCount >= 3) {
      emitPhotonBurst({ x, y });
      burstClickCount = 0;
      burstTimerSec = 60;
    }
  }, { passive: true });

  document.addEventListener('dblclick', e => {
    if (intro.enabled || !astronaut) return;
    if (!astronautHit(e.clientX, e.clientY)) return;
    toggleAstronautDockLock();
  });

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      cancelAnimationFrame(raf);
      lastVisible = false;
    } else if (!lastVisible) {
      lastVisible = true;
      lastTs = performance.now();
      if (astronaut) {
        const msg = ASTRONAUT_RETURN_PHRASES[Math.floor(Math.random() * ASTRONAUT_RETURN_PHRASES.length)];
        triggerAstronautBubble(msg, 'event');
        setAstronautMood('curious', 3.0);
        astronaut.gesture = 'salute';
        astronaut.gestureTimerSec = rand(2.1, 3.4);
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
        const msg = ASTRONAUT_SCROLL_PHRASES[Math.floor(Math.random() * ASTRONAUT_SCROLL_PHRASES.length)];
        triggerAstronautBubble(msg, 'event');
        setAstronautMood('excited', 2.4);
        astronaut.gesture = Math.random() < 0.5 ? 'spin' : 'wave';
        astronaut.gestureTimerSec = rand(1.6, 2.8);
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
