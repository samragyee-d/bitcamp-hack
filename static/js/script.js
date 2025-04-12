let mediaRecorder;
let recordedChunks = [];
let stream;
let isRecording = false;
let isPaused = false;
let recordingTime = 0;
let timer;
const timerDisplay = document.getElementById("timer");
const frames = [];  // Array to store frames
const canvas = document.createElement('canvas');  // Canvas to capture frames
const context = canvas.getContext('2d');  // Context to draw frames
const videoElement = document.getElementById('preview'); // Your video element where the webcam is displayed

function formatTime(seconds) {
  const hrs = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;
  return `${hrs.toString().padStart(2, '0')}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function updateTimer() {
  timerDisplay.textContent = `Recording Time: ${formatTime(recordingTime++)}`;
}

// Function to capture frames from the video feed
function captureFrame() {
  if (videoElement && canvas) {
    // Set the canvas size to match the video element
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;

    // Draw the current frame from the video element to the canvas
    context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);

    // Convert the canvas to a data URL (JPEG or PNG)
    const frameData = canvas.toDataURL('image/jpeg');

    // Store the captured frame (you can later use it as needed)
    frames.push(frameData);
  }
}

async function initializeWebcam() {
  try {
    console.log("Initializing webcam...");
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: true });
    console.log("Webcam stream obtained:", stream);
    const videoElement = document.getElementById('preview');
    console.log("Preview video element:", videoElement);
    videoElement.srcObject = stream;
    console.log("Stream assigned to preview element.");

    mediaRecorder = new MediaRecorder(stream);
    mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        recordedChunks.push(event.data);
      }
    };

    mediaRecorder.onstop = () => {
      const blob = new Blob(recordedChunks, { type: 'video/webm' });
      const videoURL = URL.createObjectURL(blob);
      const replayElement = document.getElementById('replay');
      replayElement.src = videoURL;
      replayElement.style.display = 'block';
    };

  } catch (error) {
    console.error("Error accessing webcam: ", error);
  }
}
// Call the function to initialize the webcam when the page loads
initializeWebcam();

// Start recording
document.getElementById("startBtn").addEventListener("click", () => {
  if (!isRecording) {
    recordedChunks = []; // Clear previous recording chunks
    mediaRecorder.start();
    isRecording = true;
    isPaused = false;
    recordingTime = 0;

    timerDisplay.style.display = "block";
    timerDisplay.textContent = `Recording Time: 00:00:00`;
    timer = setInterval(updateTimer, 1000);

    // Start capturing frames every 100ms
    setInterval(captureFrame, 100);
  } else if (isPaused) {
    mediaRecorder.resume();
    isPaused = false;
    timer = setInterval(updateTimer, 1000);
  }
});

// Pause recording
document.getElementById("pauseBtn").addEventListener("click", () => {
  if (isRecording && !isPaused) {
    mediaRecorder.pause();
    isPaused = true;
    clearInterval(timer);
  }
});

// Stop recording
document.getElementById("stopBtn").addEventListener("click", () => {
  if (isRecording) {
    mediaRecorder.stop();
    isRecording = false;
    isPaused = false;
    clearInterval(timer);
  }
});
