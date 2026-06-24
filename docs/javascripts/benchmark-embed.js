function initBenchmarkEmbeds() {
  const frames = Array.from(document.querySelectorAll('iframe[src*="benchmark_app.html"]'));
  if (!frames.length) return;

  const resizeFrame = (frame, nextHeight) => {
    if (!frame || !nextHeight || Number.isNaN(nextHeight)) return;
    const safeHeight = Math.max(960, Math.ceil(nextHeight) + 12);
    frame.style.height = `${safeHeight}px`;
  };

  frames.forEach((frame) => {
    frame.style.width = '100%';
    frame.style.display = 'block';
    frame.style.overflow = 'hidden';
    frame.setAttribute('scrolling', 'no');

    const syncDirect = () => {
      try {
        const doc = frame.contentWindow && frame.contentWindow.document;
        if (!doc) return;
        const root = doc.getElementById('mg-benchmark-root');
        const appRoot = root && root.firstElementChild ? root.firstElementChild : root;
        const nextHeight = appRoot
          ? Math.ceil(appRoot.getBoundingClientRect().height)
          : Math.max(
              doc.documentElement.scrollHeight,
              doc.body ? doc.body.scrollHeight : 0,
              doc.documentElement.offsetHeight,
              doc.body ? doc.body.offsetHeight : 0
            );
        resizeFrame(frame, nextHeight);
      } catch (_) {}
    };

    frame.addEventListener('load', () => {
      syncDirect();
      setTimeout(syncDirect, 100);
      setTimeout(syncDirect, 400);
      setTimeout(syncDirect, 1200);
    });

    setTimeout(syncDirect, 100);
  });

  window.addEventListener('message', (event) => {
    const data = event.data;
    if (!data || data.type !== 'mg-benchmark-height') return;

    frames.forEach((frame) => {
      if (frame.contentWindow === event.source) {
        resizeFrame(frame, data.height);
      }
    });
  }, { passive: true });
}

if (window.document$ && typeof window.document$.subscribe === 'function') {
  window.document$.subscribe(initBenchmarkEmbeds);
} else {
  document.addEventListener('DOMContentLoaded', initBenchmarkEmbeds);
}
