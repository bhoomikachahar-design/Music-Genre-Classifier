/**
 * Canvas Visualizers for Waveform, MFCC Heatmap & Spectrogram
 */

class VisualizerManager {
  constructor() {
    this.waveformCanvas = document.getElementById('waveformCanvas');
    this.waveCtx = this.waveformCanvas ? this.waveformCanvas.getContext('2d') : null;
    this.heatmapCanvas = document.getElementById('heatmapCanvas');
    this.heatCtx = this.heatmapCanvas ? this.heatmapCanvas.getContext('2d') : null;

    this.currentWaveform = [];
    this.currentHeatmapData = null;
    this.playbackProgress = 0.0; // 0.0 to 1.0

    this.initCanvasResize();
  }

  initCanvasResize() {
    window.addEventListener('resize', () => {
      this.resizeCanvases();
      if (this.currentWaveform.length > 0) {
        this.drawWaveform(this.playbackProgress);
      }
      if (this.currentHeatmapData) {
        this.drawMFCCHeatmap(this.currentHeatmapData);
      }
    });
  }

  resizeCanvases() {
    if (this.waveformCanvas) {
      const rect = this.waveformCanvas.getBoundingClientRect();
      this.waveformCanvas.width = rect.width * window.devicePixelRatio || 600;
      this.waveformCanvas.height = rect.height * window.devicePixelRatio || 120;
    }
    if (this.heatmapCanvas) {
      const rect = this.heatmapCanvas.getBoundingClientRect();
      this.heatmapCanvas.width = rect.width * window.devicePixelRatio || 600;
      this.heatmapCanvas.height = rect.height * window.devicePixelRatio || 180;
    }
  }

  setWaveformData(waveformData) {
    this.currentWaveform = waveformData || [];
    this.playbackProgress = 0.0;
    this.resizeCanvases();
    this.drawWaveform(0.0);
  }

  drawWaveform(progress = 0.0) {
    if (!this.waveCtx || !this.waveformCanvas || this.currentWaveform.length === 0) return;
    this.playbackProgress = progress;

    const ctx = this.waveCtx;
    const w = this.waveformCanvas.width;
    const h = this.waveformCanvas.height;
    const midY = h / 2;

    ctx.clearRect(0, 0, w, h);

    // Background gradient
    const bgGrad = ctx.createLinearGradient(0, 0, 0, h);
    bgGrad.addColorStop(0, '#0f172a');
    bgGrad.addColorStop(1, '#1e293b');
    ctx.fillStyle = bgGrad;
    ctx.fillRect(0, 0, w, h);

    // Center subtle grid line
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, midY);
    ctx.lineTo(w, midY);
    ctx.stroke();

    const data = this.currentWaveform;
    const numBars = data.length;
    const barWidth = Math.max(2, (w / numBars) * 0.85);
    const barSpacing = w / numBars;
    const currentX = progress * w;

    // Draw bars
    for (let i = 0; i < numBars; i++) {
      const x = i * barSpacing;
      const amp = Math.min(1.0, Math.abs(data[i])) * (midY * 0.88);
      const isPlayed = x <= currentX;

      // Color based on played or unplayed
      if (isPlayed) {
        ctx.fillStyle = '#06b6d4'; // bright cyan played
      } else {
        ctx.fillStyle = '#475569'; // slate unplayed
      }

      // Rounded vertical bar
      const topY = midY - amp;
      const barH = amp * 2;
      ctx.fillRect(x, topY, barWidth, Math.max(2, barH));
    }

    // Draw playback scrubber line
    if (progress > 0 && progress < 1) {
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2 * window.devicePixelRatio;
      ctx.shadowColor = '#38bdf8';
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.moveTo(currentX, 0);
      ctx.lineTo(currentX, h);
      ctx.stroke();
      ctx.shadowBlur = 0; // reset
    }
  }

  drawMFCCHeatmap(heatmapPayload) {
    if (!this.heatCtx || !this.heatmapCanvas || !heatmapPayload || !heatmapPayload.matrix) return;
    this.currentHeatmapData = heatmapPayload;

    const ctx = this.heatCtx;
    const w = this.heatmapCanvas.width;
    const h = this.heatmapCanvas.height;
    ctx.clearRect(0, 0, w, h);

    const matrix = heatmapPayload.matrix;
    const rows = matrix.length; // 20
    const cols = matrix[0] ? matrix[0].length : 1;

    // Find min and max for normalization
    let minVal = Infinity;
    let maxVal = -Infinity;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const val = matrix[r][c];
        if (val < minVal) minVal = val;
        if (val > maxVal) maxVal = val;
      }
    }
    const range = (maxVal - minVal) || 1.0;

    const leftMargin = 55 * window.devicePixelRatio;
    const bottomMargin = 20 * window.devicePixelRatio;
    const plotW = w - leftMargin - (15 * window.devicePixelRatio);
    const plotH = h - bottomMargin;

    const cellW = plotW / cols;
    const cellH = plotH / rows;

    // Draw Heatmap Cells
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const norm = (matrix[r][c] - minVal) / range;
        ctx.fillStyle = this.getColorForNorm(norm);
        ctx.fillRect(leftMargin + (c * cellW), r * cellH, Math.ceil(cellW), Math.ceil(cellH));
      }
    }

    // Row Labels (MFCC 1, 5, 10, 15, 20)
    ctx.fillStyle = '#94a3b8';
    ctx.font = `${9 * window.devicePixelRatio}px Inter, sans-serif`;
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';

    for (let r = 0; r < rows; r += 4) {
      const yPos = (r * cellH) + (cellH / 2);
      ctx.fillText(`MFCC ${r + 1}`, leftMargin - (6 * window.devicePixelRatio), yPos);
    }
    // last row
    ctx.fillText(`MFCC 20`, leftMargin - (6 * window.devicePixelRatio), ((rows - 1) * cellH) + (cellH / 2));

    // Time Axis labels
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    ctx.fillText('0s', leftMargin, plotH + (4 * window.devicePixelRatio));
    ctx.fillText('Time ->', leftMargin + (plotW / 2), plotH + (4 * window.devicePixelRatio));
    ctx.fillText('30s (End)', leftMargin + plotW, plotH + (4 * window.devicePixelRatio));
  }

  // Magma-style colormap (Deep Navy -> Purple -> Magenta -> Orange -> Yellow)
  getColorForNorm(norm) {
    const t = Math.max(0, Math.min(1, norm));
    let r, g, b;

    if (t < 0.25) {
      const f = t / 0.25;
      r = Math.floor(15 + f * 50);
      g = Math.floor(15 + f * 20);
      b = Math.floor(45 + f * 90);
    } else if (t < 0.5) {
      const f = (t - 0.25) / 0.25;
      r = Math.floor(65 + f * 105);
      g = Math.floor(35 + f * 20);
      b = Math.floor(135 + f * 30);
    } else if (t < 0.75) {
      const f = (t - 0.5) / 0.25;
      r = Math.floor(170 + f * 65);
      g = Math.floor(55 + f * 85);
      b = Math.floor(165 - f * 115);
    } else {
      const f = (t - 0.75) / 0.25;
      r = Math.floor(235 + f * 20);
      g = Math.floor(140 + f * 115);
      b = Math.floor(50 + f * 100);
    }
    return `rgb(${r}, ${g}, ${b})`;
  }
}

window.VisualizerManager = VisualizerManager;
