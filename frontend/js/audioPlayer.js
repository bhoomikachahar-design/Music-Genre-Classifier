/**
 * Audio Player Controller
 * Coordinates HTML5 Audio playback with Waveform Canvas scrubbing
 */

class AudioPlayerController {
  constructor(visualizer) {
    this.visualizer = visualizer;
    this.audio = new Audio();
    this.isPlaying = false;
    this.currentUrl = null;

    // DOM Elements
    this.playBtn = document.getElementById('playPauseBtn');
    this.playIcon = document.getElementById('playIcon');
    this.pauseIcon = document.getElementById('pauseIcon');
    this.currentTimeLabel = document.getElementById('currentTimeLabel');
    this.durationLabel = document.getElementById('durationLabel');
    this.volumeSlider = document.getElementById('volumeSlider');
    this.waveformCanvas = document.getElementById('waveformCanvas');

    this.bindEvents();
  }

  bindEvents() {
    if (this.playBtn) {
      this.playBtn.addEventListener('click', () => this.togglePlay());
    }

    if (this.volumeSlider) {
      this.volumeSlider.addEventListener('input', (e) => {
        this.audio.volume = parseFloat(e.target.value);
      });
    }

    this.audio.addEventListener('timeupdate', () => {
      this.onTimeUpdate();
    });

    this.audio.addEventListener('ended', () => {
      this.isPlaying = false;
      this.updatePlayState();
      this.visualizer.drawWaveform(0.0);
    });

    // Waveform click to seek
    if (this.waveformCanvas) {
      this.waveformCanvas.addEventListener('click', (e) => {
        const rect = this.waveformCanvas.getBoundingClientRect();
        const clickX = e.clientX - rect.left;
        const progress = Math.max(0, Math.min(1, clickX / rect.width));

        if (this.audio.duration && !isNaN(this.audio.duration)) {
          this.audio.currentTime = progress * this.audio.duration;
          this.visualizer.drawWaveform(progress);
        }
      });
    }
  }

  loadAudio(url) {
    this.currentUrl = url;
    this.audio.src = url;
    this.audio.load();
    this.isPlaying = false;
    this.updatePlayState();

    this.audio.addEventListener('loadedmetadata', () => {
      if (this.durationLabel) {
        this.durationLabel.textContent = this.formatTime(this.audio.duration);
      }
    }, { once: true });
  }

  togglePlay() {
    if (!this.currentUrl) return;

    if (this.isPlaying) {
      this.audio.pause();
      this.isPlaying = false;
    } else {
      this.audio.play().then(() => {
        this.isPlaying = true;
        this.updatePlayState();
      }).catch(err => {
        console.error("Playback error:", err);
      });
    }
    this.updatePlayState();
  }

  updatePlayState() {
    if (!this.playIcon || !this.pauseIcon) return;
    if (this.isPlaying) {
      this.playIcon.classList.add('hidden');
      this.pauseIcon.classList.remove('hidden');
    } else {
      this.playIcon.classList.remove('hidden');
      this.pauseIcon.classList.add('hidden');
    }
  }

  onTimeUpdate() {
    if (!this.audio.duration || isNaN(this.audio.duration)) return;

    const progress = this.audio.currentTime / this.audio.duration;
    if (this.currentTimeLabel) {
      this.currentTimeLabel.textContent = this.formatTime(this.audio.currentTime);
    }

    // Update Waveform Visualizer progress line
    this.visualizer.drawWaveform(progress);
  }

  formatTime(seconds) {
    if (!seconds || isNaN(seconds)) return "00:00";
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins < 10 ? '0' : ''}${mins}:${secs < 10 ? '0' : ''}${secs}`;
  }
}

window.AudioPlayerController = AudioPlayerController;
