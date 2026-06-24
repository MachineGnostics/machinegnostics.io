function initBenchmarkEmbeds() {
  if (window.__mgBenchmarkEmbedsInitialized) return;
  window.__mgBenchmarkEmbedsInitialized = true;

  const frames = Array.from(document.querySelectorAll('iframe[src*="benchmark_app.html"]'));
  if (!frames.length) return;

  const resizeFrame = (frame, nextHeight) => {
    if (!frame || !nextHeight || Number.isNaN(nextHeight)) return;
    const viewportMin =
      window.innerWidth < 480 ? 520 :
      window.innerWidth < 768 ? 640 :
      960;
    const safeHeight = Math.max(viewportMin, Math.ceil(nextHeight) + 12);
    frame.style.height = `${safeHeight}px`;
  };

  frames.forEach((frame) => {
    frame.style.width = '100%';
    frame.style.display = 'block';
    frame.style.overflow = 'auto';
    frame.setAttribute('scrolling', 'yes');

    const syncDirect = () => {
      try {
        const doc = frame.contentWindow && frame.contentWindow.document;
        if (!doc) return;
        const root = doc.getElementById('mg-benchmark-root');
        const appRoot = root && root.firstElementChild ? root.firstElementChild : root;
        const nextHeight = Math.max(
          appRoot ? Math.ceil(appRoot.getBoundingClientRect().height) : 0,
          appRoot ? appRoot.scrollHeight : 0,
          root ? root.scrollHeight : 0,
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

  const syncAll = () => {
    frames.forEach((frame) => {
      try {
        const doc = frame.contentWindow && frame.contentWindow.document;
        if (!doc) return;
        const root = doc.getElementById('mg-benchmark-root');
        const appRoot = root && root.firstElementChild ? root.firstElementChild : root;
        const nextHeight = Math.max(
          appRoot ? Math.ceil(appRoot.getBoundingClientRect().height) : 0,
          appRoot ? appRoot.scrollHeight : 0,
          root ? root.scrollHeight : 0,
          doc.documentElement.scrollHeight,
          doc.body ? doc.body.scrollHeight : 0,
          doc.documentElement.offsetHeight,
          doc.body ? doc.body.offsetHeight : 0
        );
        resizeFrame(frame, nextHeight);
      } catch (_) {}
    });
  };

  window.addEventListener('resize', syncAll, { passive: true });
  window.addEventListener('orientationchange', syncAll, { passive: true });

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
