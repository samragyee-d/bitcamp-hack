document.addEventListener('DOMContentLoaded', function() {
  const previewVideo = document.getElementById('preview');
  const startBtn = document.getElementById('startBtn');
  const pauseBtn = document.getElementById('pauseBtn');
  const stopBtn = document.getElementById('stopBtn');
  const timerDisplay = document.getElementById('timer');
  let stream;
  let recordingTime = 0;
  let recordingInterval;

  // Access user's webcam
  navigator.mediaDevices.getUserMedia({ video: true })
    .then(function(streamObj) {
      stream = streamObj;
      previewVideo.srcObject = stream;

      // Only autoplay after metadata is loaded
      previewVideo.onloadedmetadata = function() {
        previewVideo.play();
      };
    })
    .catch(function(err) {
      console.error('Error accessing webcam:', err);
      alert('Unable to access webcam. Please check your permissions.');
    });

  // Start button
  startBtn.addEventListener('click', function() {
    if (previewVideo.paused) {
      previewVideo.play();
      startBtn.disabled = true;
      pauseBtn.disabled = false;
      stopBtn.disabled = false;
      startTimer();
    }
  });

  // Pause button
  pauseBtn.addEventListener('click', function() {
    if (!previewVideo.paused) {
      previewVideo.pause();
      pauseBtn.disabled = true;
      startBtn.disabled = false;
    }
  });

  // Stop button
  stopBtn.addEventListener('click', function() {
    previewVideo.pause();
    previewVideo.currentTime = 0;
    startBtn.disabled = false;
    pauseBtn.disabled = true;
    stopBtn.disabled = true;
    stopTimer();
  });

  // Timer functions
  function startTimer() {
    recordingInterval = setInterval(function() {
      recordingTime++;
      timerDisplay.textContent = 'Recording Time: ' + recordingTime + 's';
    }, 1000);
  }

  function stopTimer() {
    clearInterval(recordingInterval);
    recordingTime = 0;
    timerDisplay.textContent = 'Recording Time: 0s';
  }
});
