/**
 * MACHINE GNOSTICS — INTERACTIVE LAYER
 * Typing effect, scroll reveal, cursor, tab interactions
 */

document.addEventListener('DOMContentLoaded', function () {

  // ─── TYPING EFFECT (hero subtitle) ──────────────────────────────────────────
  const typedEl  = document.getElementById('mg-typed-text');
  const cursorEl = document.getElementById('mg-cursor');

  if (typedEl && cursorEl) {
    const fullText = typedEl.dataset.text || '';
    typedEl.textContent = '';
    let i = 0;

    setInterval(() => {
      cursorEl.style.opacity = cursorEl.style.opacity === '0' ? '1' : '0';
    }, 530);

    function typeChar() {
      if (i < fullText.length) {
        typedEl.textContent += fullText[i];
        i++;
        const delay = fullText[i - 1] === ',' || fullText[i - 1] === '.' ? 180 :
                      fullText[i - 1] === ' ' ? 60 : 32 + Math.random() * 25;
        setTimeout(typeChar, delay);
      } else {
        setTimeout(() => {
          if (cursorEl) cursorEl.style.transition = 'opacity 1s';
          if (cursorEl) cursorEl.style.opacity = '0';
        }, 3000);
      }
    }

    setTimeout(typeChar, 800);
  }

  // ─── SCROLL REVEAL ────────────────────────────────────────────────────────────
  const revealTargets = document.querySelectorAll(
    '.md-typeset h2, .md-typeset h3, .md-typeset p, ' +
    '.md-typeset .admonition, .md-typeset table, ' +
    '.mg-card, .mg-tag-cloud'
  );

  revealTargets.forEach((el, i) => {
    el.classList.add('mg-reveal');
    el.style.transitionDelay = `${Math.min(i * 0.04, 0.4)}s`;
  });

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('mg-visible');
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.1, rootMargin: '0px 0px -40px 0px' }
  );

  revealTargets.forEach(el => observer.observe(el));

  // ─── FLOATING PRINCIPLE TAGS ─────────────────────────────────────────────────
  document.querySelectorAll('.mg-tag').forEach((tag, i) => {
    const period  = 4000 + Math.random() * 4000;
    const offset  = Math.random() * Math.PI * 2;
    const amplitude = 4 + Math.random() * 5;

    function floatTag(timestamp) {
      const y = Math.sin((timestamp / period) * Math.PI * 2 + offset) * amplitude;
      tag.style.transform = `translateY(${y}px)`;
      requestAnimationFrame(floatTag);
    }

    setTimeout(() => requestAnimationFrame(floatTag), i * 150);
  });

  // ─── BUTTON SCAN-LINE EFFECT ─────────────────────────────────────────────────
  document.querySelectorAll('.mg-btn').forEach(btn => {
    btn.addEventListener('click', function (e) {
      const rect = btn.getBoundingClientRect();
      const ripple = document.createElement('span');
      ripple.className = 'mg-ripple';
      ripple.style.left = `${e.clientX - rect.left}px`;
      ripple.style.top  = `${e.clientY - rect.top}px`;
      btn.appendChild(ripple);
      setTimeout(() => ripple.remove(), 600);
    });
  });

});
