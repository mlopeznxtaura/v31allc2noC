function showJson(el, data) {
  el.textContent = JSON.stringify(data, null, 2);
}

function renderArchitecture(arch) {
  const wrap = document.getElementById('arch-flow');
  if (!arch?.layers) {
    wrap.innerHTML = '<p class="hint">Architecture metadata unavailable.</p>';
    return;
  }
  const blocks = arch.layers.map((layer, i) => {
    const nodes = (layer.nodes || [])
      .map(
        (n) =>
          `<div class="arch-node"><strong>${n.label}</strong><span>${n.desc || ''}</span></div>`,
      )
      .join('');
    const arrow = i < arch.layers.length - 1 ? '<div class="arch-arrow">▼</div>' : '';
    return `<div class="arch-block"><h3>${layer.title}</h3><div class="arch-nodes">${nodes}</div></div>${arrow}`;
  });
  wrap.innerHTML = blocks.join('');
}

async function loadInfo() {
  try {
    const info = await fetch('/api/info').then((r) => r.json());
    renderArchitecture(info.architecture);
  } catch (e) {
    document.getElementById('arch-flow').innerHTML = `<p class="hint">${e}</p>`;
  }
}

document.getElementById('btn-infer').addEventListener('click', async () => {
  const out = document.getElementById('infer-output');
  out.textContent = 'Running V31 forward pass...';
  const body = {
    text: document.getElementById('infer-text').value.trim(),
    sandbox: document.getElementById('infer-sandbox').value === 'true',
    temperature: Number(document.getElementById('infer-temp').value),
  };
  try {
    const res = await fetch('/api/infer', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    showJson(out, data);
  } catch (e) {
    out.textContent = String(e);
  }
});

document.getElementById('btn-reset').addEventListener('click', async () => {
  const out = document.getElementById('infer-output');
  try {
    const data = await fetch('/api/reset', { method: 'POST' }).then((r) => r.json());
    showJson(out, data);
  } catch (e) {
    out.textContent = String(e);
  }
});

loadInfo();
