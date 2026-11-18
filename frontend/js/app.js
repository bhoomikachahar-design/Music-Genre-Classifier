/**
 * Main Frontend Application Controller
 * Handles state, API calls, tab switching, and dynamic rendering.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Initialize Components
  const visualizer = new VisualizerManager();
  const player = new AudioPlayerController(visualizer);

  let activeModel = 'ensemble';
  let probabilityChart = null;
  let featureImportanceChart = null;
  let cachedMetrics = null;

  // DOM Elements
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabPanes = document.querySelectorAll('.tab-pane');
  const modelSelect = document.getElementById('modelSelect');
  const fileInput = document.getElementById('audioFileInput');
  const dropzone = document.getElementById('uploadDropzone');
  const loadingOverlay = document.getElementById('loadingOverlay');
  const loadingText = document.getElementById('loadingText');
  const resultsContainer = document.getElementById('resultsContainer');
  const emptyState = document.getElementById('emptyState');

  // Genre badge color themes
  const genreColors = {
    blues: 'bg-blue-900/60 text-blue-300 border-blue-500/30',
    classical: 'bg-amber-900/60 text-amber-300 border-amber-500/30',
    country: 'bg-yellow-900/60 text-yellow-300 border-yellow-500/30',
    disco: 'bg-fuchsia-900/60 text-fuchsia-300 border-fuchsia-500/30',
    hiphop: 'bg-emerald-900/60 text-emerald-300 border-emerald-500/30',
    jazz: 'bg-indigo-900/60 text-indigo-300 border-indigo-500/30',
    metal: 'bg-red-900/60 text-red-300 border-red-500/30',
    pop: 'bg-pink-900/60 text-pink-300 border-pink-500/30',
    reggae: 'bg-lime-900/60 text-lime-300 border-lime-500/30',
    rock: 'bg-orange-900/60 text-orange-300 border-orange-500/30'
  };

  // ----------------- TAB SWITCHING -----------------
  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      const targetTab = btn.getAttribute('data-tab');
      tabBtns.forEach(b => {
        b.classList.remove('bg-cyan-500/20', 'text-cyan-400', 'border-cyan-500');
        b.classList.add('text-slate-400', 'border-transparent');
      });
      btn.classList.add('bg-cyan-500/20', 'text-cyan-400', 'border-cyan-500');
      btn.classList.remove('text-slate-400', 'border-transparent');

      tabPanes.forEach(pane => {
        if (pane.id === targetTab) {
          pane.classList.remove('hidden');
        } else {
          pane.classList.add('hidden');
        }
      });

      if (targetTab === 'metricsTab') {
        loadMetricsTab();
      } else if (targetTab === 'historyTab') {
        loadHistoryTab();
      } else if (targetTab === 'pipelineTab') {
        loadPipelineTab();
      }
    });
  });

  // Model Selector change
  if (modelSelect) {
    modelSelect.addEventListener('change', (e) => {
      activeModel = e.target.value;
      const modelBadge = document.getElementById('activeModelBadge');
      if (modelBadge) {
        const labels = {
          ensemble: 'Hybrid Ensemble (82% Acc)',
          random_forest: 'Random Forest (79% Acc)',
          svm: 'RBF-SVM (76% Acc)'
        };
        modelBadge.textContent = labels[activeModel] || activeModel;
      }
    });
  }

  // ----------------- FILE UPLOAD HANDLING -----------------
  if (dropzone && fileInput) {
    dropzone.addEventListener('click', () => fileInput.click());

    ['dragenter', 'dragover'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(eventName => {
      dropzone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropzone.classList.remove('dragover');
      });
    });

    dropzone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files.length > 0) {
        processUploadedFile(files[0]);
      }
    });

    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length > 0) {
        processUploadedFile(e.target.files[0]);
      }
    });
  }

  // Process File with Backend API
  async function processUploadedFile(file) {
    const validExtensions = ['.wav', '.mp3', '.ogg', '.flac', '.m4a'];
    const fileName = file.name.toLowerCase();
    const isValid = validExtensions.some(ext => fileName.endsWith(ext));

    if (!isValid) {
      alert("Unsupported format. Please upload a WAV, MP3, OGG, or FLAC file.");
      return;
    }

    showLoading(true, "Extracting 58 audio features via Librosa...");

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`/api/predict?model=${activeModel}`, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail || "Audio analysis failed.");
      }

      const data = await response.json();
      displayResults(data);
    } catch (err) {
      alert(`Error analyzing audio: ${err.message}`);
    } finally {
      showLoading(false);
    }
  }

  // ----------------- SAMPLE CLIPS PRESETS -----------------
  document.querySelectorAll('.sample-btn').forEach(btn => {
    btn.addEventListener('click', async () => {
      const sampleId = btn.getAttribute('data-sample');
      showLoading(true, `Analyzing demo sample: ${sampleId.toUpperCase()}...`);

      try {
        const response = await fetch(`/api/predict-sample/${sampleId}?model=${activeModel}`, {
          method: 'POST'
        });

        if (!response.ok) {
          const err = await response.json();
          throw new Error(err.detail || "Sample analysis failed.");
        }

        const data = await response.json();
        displayResults(data);
      } catch (err) {
        alert(`Error loading sample: ${err.message}`);
      } finally {
        showLoading(false);
      }
    });
  });

  // ----------------- DISPLAY PREDICTION RESULTS -----------------
  function displayResults(data) {
    if (emptyState) emptyState.classList.add('hidden');
    if (resultsContainer) resultsContainer.classList.remove('hidden');

    const pred = data.prediction;
    const vis = data.visualizations;
    const fileInfo = data.file_info;
    const feats = data.extracted_features;

    // Load audio into player
    player.loadAudio(fileInfo.audio_url);

    // Track Title
    const titleEl = document.getElementById('currentTrackTitle');
    if (titleEl) titleEl.textContent = fileInfo.original_name;

    // Top Genre Badge & Confidence
    const badgeEl = document.getElementById('predictedGenreBadge');
    if (badgeEl) {
      badgeEl.textContent = pred.predicted_genre.toUpperCase();
      badgeEl.className = `genre-tag px-4 py-1.5 rounded-full border text-lg font-bold tracking-wider ${genreColors[pred.predicted_genre] || 'bg-cyan-900/60 text-cyan-300'}`;
    }

    const confEl = document.getElementById('predictedConfidenceText');
    if (confEl) {
      confEl.textContent = `${pred.confidence_percentage}% Confidence`;
    }

    // Genre Profile Card
    const profileTitle = document.getElementById('profileTitle');
    const profileDesc = document.getElementById('profileDescription');
    const profileFeats = document.getElementById('profileKeyFeatures');
    const profileInst = document.getElementById('profileInstruments');

    if (profileTitle) profileTitle.textContent = `${pred.profile.title} Characteristics`;
    if (profileDesc) profileDesc.textContent = pred.profile.description;
    if (profileFeats) profileFeats.textContent = pred.profile.key_features;
    if (profileInst) {
      profileInst.innerHTML = pred.profile.typical_instruments
        .map(i => `<span class="px-2.5 py-1 text-xs rounded-md bg-slate-700/80 text-slate-300 border border-slate-600">${i}</span>`)
        .join('');
    }

    // Acoustic Metrics Dashboard
    document.getElementById('metricDuration').textContent = `${vis.summary.duration}s`;
    document.getElementById('metricTempo').textContent = `${vis.summary.estimated_tempo_bpm} BPM`;
    document.getElementById('metricRms').textContent = `${vis.summary.rms_energy_db} dB`;
    document.getElementById('metricCentroid').textContent = `${Math.round(feats.spectral_centroid_mean || 0)} Hz`;
    document.getElementById('metricRolloff').textContent = `${Math.round(feats.rolloff_mean || 0)} Hz`;
    document.getElementById('metricZcr').textContent = `${(feats.zero_crossing_rate_mean || 0).toFixed(4)}`;

    // Render Waveform
    visualizer.setWaveformData(vis.waveform);

    // Render MFCC Heatmap
    visualizer.drawMFCCHeatmap(vis.mfcc_heatmap);

    // Render Probability Distribution Chart
    renderProbabilityChart(pred.probabilities);

    // Scroll to results smoothly
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  // ----------------- PROBABILITY BAR CHART -----------------
  function renderProbabilityChart(probList) {
    const ctx = document.getElementById('probChart');
    if (!ctx) return;

    const labels = probList.map(p => p.genre.toUpperCase());
    const dataValues = probList.map(p => p.percentage);

    if (probabilityChart) {
      probabilityChart.destroy();
    }

    probabilityChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: labels,
        datasets: [{
          label: 'Probability (%)',
          data: dataValues,
          backgroundColor: dataValues.map((_, i) => i === 0 ? 'rgba(6, 182, 212, 0.85)' : 'rgba(51, 65, 85, 0.7)'),
          borderColor: dataValues.map((_, i) => i === 0 ? '#06b6d4' : '#64748b'),
          borderWidth: 1,
          borderRadius: 6
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (item) => ` ${item.raw}% Likelihood`
            }
          }
        },
        scales: {
          x: {
            beginAtZero: true,
            max: 100,
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#94a3b8' }
          },
          y: {
            grid: { display: false },
            ticks: { color: '#f8fafc', font: { weight: '600' } }
          }
        }
      }
    });
  }

  // ----------------- METRICS TAB -----------------
  async function loadMetricsTab() {
    if (cachedMetrics) {
      renderMetricsUI(cachedMetrics);
      return;
    }

    try {
      const res = await fetch('/api/models');
      const data = await res.json();
      cachedMetrics = data;
      renderMetricsUI(data);
    } catch (e) {
      console.error("Failed to load metrics:", e);
    }
  }

  function renderMetricsUI(data) {
    const models = data.models;
    if (!models) return;

    // Overview Cards
    document.getElementById('rfAcc').textContent = `${(models.random_forest.metrics.accuracy * 100).toFixed(1)}%`;
    document.getElementById('rfF1').textContent = models.random_forest.metrics.macro_f1.toFixed(3);

    document.getElementById('svmAcc').textContent = `${(models.svm.metrics.accuracy * 100).toFixed(1)}%`;
    document.getElementById('svmF1').textContent = models.svm.metrics.macro_f1.toFixed(3);

    document.getElementById('ensAcc').textContent = `${(models.ensemble.metrics.accuracy * 100).toFixed(1)}%`;
    document.getElementById('ensF1').textContent = models.ensemble.metrics.macro_f1.toFixed(3);

    // Confusion Matrix (Displaying Random Forest / Ensemble)
    renderConfusionMatrix(models.ensemble.metrics.confusion_matrix, data.dataset_info.classes);

    // Per-class Breakdown Table
    renderClassTable(models.ensemble.metrics.per_class);

    // Top Features Chart
    renderFeatureImportanceChart(data.top_features);
  }

  function renderConfusionMatrix(matrix, classes) {
    const table = document.getElementById('cmTable');
    if (!table) return;

    let html = '<thead><tr><th class="text-slate-400">Actual \\ Pred</th>';
    classes.forEach(c => {
      html += `<th class="text-cyan-400 font-semibold uppercase text-xs">${c.slice(0, 3)}</th>`;
    });
    html += '</tr></thead><tbody>';

    matrix.forEach((row, rIdx) => {
      html += `<tr><td class="text-cyan-400 font-semibold uppercase text-xs text-left">${classes[rIdx].slice(0, 3)}</td>`;
      const rowSum = row.reduce((a, b) => a + b, 0) || 1;
      row.forEach((val, cIdx) => {
        const isDiag = rIdx === cIdx;
        const intensity = (val / rowSum);
        const bg = isDiag
          ? `rgba(6, 182, 212, ${0.2 + intensity * 0.7})`
          : val > 0 ? `rgba(239, 68, 68, ${0.15 + (val / 5) * 0.4})` : 'transparent';
        const textColor = isDiag ? '#38bdf8' : (val > 0 ? '#fca5a5' : '#64748b');

        html += `<td class="cm-cell border border-slate-700/50" style="background-color: ${bg}; color: ${textColor};" title="Actual: ${classes[rIdx]}, Predicted: ${classes[cIdx]} (${val} samples)">${val}</td>`;
      });
      html += '</tr>';
    });
    html += '</tbody>';
    table.innerHTML = html;
  }

  function renderClassTable(perClass) {
    const tbody = document.getElementById('perClassTbody');
    if (!tbody) return;

    let html = '';
    Object.entries(perClass).forEach(([genre, stats]) => {
      html += `
        <tr class="border-b border-slate-700/50 hover:bg-slate-800/40">
          <td class="py-2 px-3 font-semibold uppercase text-slate-200">${genre}</td>
          <td class="py-2 px-3 text-cyan-400">${(stats.precision * 100).toFixed(1)}%</td>
          <td class="py-2 px-3 text-teal-400">${(stats.recall * 100).toFixed(1)}%</td>
          <td class="py-2 px-3 text-amber-400 font-semibold">${(stats.f1_score * 100).toFixed(1)}%</td>
          <td class="py-2 px-3 text-slate-400">${stats.support}</td>
        </tr>
      `;
    });
    tbody.innerHTML = html;
  }

  function renderFeatureImportanceChart(features) {
    const ctx = document.getElementById('featureImportanceChart');
    if (!ctx || !features) return;

    if (featureImportanceChart) featureImportanceChart.destroy();

    const top12 = features.slice(0, 12);
    featureImportanceChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: top12.map(f => f.feature),
        datasets: [{
          label: 'Gini Importance Score',
          data: top12.map(f => f.importance),
          backgroundColor: 'rgba(139, 92, 246, 0.75)',
          borderColor: '#8b5cf6',
          borderWidth: 1,
          borderRadius: 6
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            ticks: { color: '#94a3b8', font: { size: 10 } },
            grid: { display: false }
          },
          y: {
            ticks: { color: '#94a3b8' },
            grid: { color: 'rgba(255, 255, 255, 0.05)' }
          }
        },
        plugins: {
          legend: { display: false }
        }
      }
    });
  }

  // ----------------- HISTORY TAB -----------------
  async function loadHistoryTab() {
    try {
      const res = await fetch('/api/history');
      const history = await res.json();
      const tbody = document.getElementById('historyTbody');
      if (!tbody) return;

      if (history.length === 0) {
        tbody.innerHTML = `<tr><td colspan="6" class="py-8 text-center text-slate-500">No predictions recorded yet. Upload or try a sample audio file!</td></tr>`;
        return;
      }

      let html = '';
      history.forEach((h, idx) => {
        html += `
          <tr class="border-b border-slate-700/50 hover:bg-slate-800/40">
            <td class="py-3 px-4 text-slate-400">${idx + 1}</td>
            <td class="py-3 px-4 font-medium text-slate-200">${h.filename}</td>
            <td class="py-3 px-4"><span class="genre-tag ${genreColors[h.predicted_genre] || 'bg-slate-700'}">${h.predicted_genre.toUpperCase()}</span></td>
            <td class="py-3 px-4 font-bold text-cyan-400">${h.confidence_percentage}%</td>
            <td class="py-3 px-4 text-slate-400">${h.duration}s / ${h.tempo_bpm} BPM</td>
            <td class="py-3 px-4 text-xs text-slate-500">${h.timestamp}</td>
          </tr>
        `;
      });
      tbody.innerHTML = html;
    } catch (e) {
      console.error("Failed to load history:", e);
    }
  }

  // Export buttons
  const exportCsvBtn = document.getElementById('exportCsvBtn');
  const exportJsonBtn = document.getElementById('exportJsonBtn');
  if (exportCsvBtn) {
    exportCsvBtn.addEventListener('click', () => {
      window.location.href = '/api/history/export?format=csv';
    });
  }
  if (exportJsonBtn) {
    exportJsonBtn.addEventListener('click', () => {
      window.location.href = '/api/history/export?format=json';
    });
  }

  // ----------------- PIPELINE TAB -----------------
  async function loadPipelineTab() {
    // Already populated in HTML, but can trigger any visual flourishes
  }

  // Helper
  function showLoading(show, message = "Processing...") {
    if (!loadingOverlay) return;
    if (show) {
      loadingText.textContent = message;
      loadingOverlay.classList.remove('hidden');
    } else {
      loadingOverlay.classList.add('hidden');
    }
  }
});
