## Inspiration

We’ve all faced it: you sit down to study, and 10 minutes later you’re deep in a meme rabbit hole or doomscrolling TikTok. Or worse yet, you’ve actually managed to lock in and you find yourself getting angry and unproductive. In today’s world of constant digital distraction, staying focused isn’t just about willpower — it’s about having the right tools.

With online learning and self-study becoming the norm, we envisioned a smarter, emotionally-aware companion that supports—not scolds—you during your study sessions. That’s how **EvaStudy** was born.

- 📱 Detects distractions like phone usage in real time  
- 💬 Recognizes emotional fatigue and offers personalized support  
- 🎥 Boosts accountability with session recordings and insights

EvaStudy blends intelligent detection with compassionate feedback to help make every minute of studying more effective—and more human.

---

## What it does

EvaStudy is an AI-powered web app that transforms how you study by combining multi-modal machine learning with conversational GenAI.

### 🎥 Monitor Sessions
- Real-time phone detection using YOLOv5
- Facial expression analysis to record emotion in a log and recommend breaks accordingly
- Webcam recording for self-review and accountability

### 🤖 Deliver Smart Feedback via Chat Bot
- Live nudges if you're distracted (via Gemini)
- Encouragement when frustration is detected
- Break suggestions based on emotional data

📹 *Why record at 4× speed?*  
By saving sped-up versions of study sessions, EvaStudy makes it easier to **review long sessions quickly**, reflect on productivity patterns, and stay consistent. This also taps into the growing trend of “study-with-me” style content that helps users **feel connected, motivated, and mindful**—even during solo study time.

---

## Business Model

We aim to empower focused learning—at scale.

### Target Audience:
- High school & college students
- Working professionals seeking productivity tools
- EdTech platforms looking to enhance engagement

### Revenue Streams:
- **Individual Plans:** Monthly/yearly subscriptions with core features  
- **Institutional Licenses:** Bulk discounts for schools and universities  
- **Premium Add-Ons:** Personalized coaching, advanced analytics  
- **Partnerships:** Integrations with learning platforms and referral models

### Projections:
- **Year 1:** 5,000 users | $300K revenue  
- **Year 2:** 15,000 users | $1M revenue | Breaking Even 
- **Year 3:** 40,000 users | $3M revenue

We're currently seeking **$500,000 in funding** to accelerate development, expand our reach, and build a community around focus and wellness in learning.

---

## How we built it

### Frontend:
- Flask with Jinja2 templating
- OpenCV video feed + custom CSS UI

### Backend Highlights:
- **Object Detection:** YOLOv5s model from Torch Hub identifies phones  
- **Emotion Recognition:** FER-2013 CNN via pre-trained model (AI-Gajendra)  
- **Live Feedback:** Gemini API crafts contextual nudges like “Put down your phone”  
- **Cooldown Logic:** Tracks recent emotions to prevent repetitive alerts  
- **Video Recording:** Captures sessions at 4× speed without audio (every 4th frame)  
- **Storage:** MySQL backend handles video metadata and user session logs

### Stack Overview:
- Python, TensorFlow/Keras, Torch, OpenCV  
- Google Gemini API, MySQL, Flask

---

## Challenges

- Balancing performance vs. accuracy in real-time video  
- Avoiding false positives in face and phone detection
- Avoiding overrepresentation in emotion detection  
- Coordinating multiple AI models with shared session state  
- Designing interventions that *support* rather than annoy  
- Ensuring secure, isolated video recordings for each user and routing related to SQL, Flask and JavaScript

---

## Accomplishments

- Seamless integration of emotion + object detection 
- Built a real-time system that actually respects user experience  
- Developed non-intrusive reminders based on emotion context  
- Created full video review tools with annotation overlays

---

## What we learned

- Tuning AI models for real-world use requires iteration  
- Multimodal systems (CV + NLP + user flow) require careful coordination  
- Frontend UX must evolve alongside backend logic to maintain trust  
- Python + OpenCV can deliver real-time performance with the right optimizations

---

## What’s next for EvaStudy

### 🚀 Deployment and more Features
- Deploy to AWS Cloud or Cloudflare for increased security, usability
- Calendar and Pomodoro integration  
- Facial recognition for personalized emotion baselines  
- AI companion with customizable tone/personalities  

### 👥 Community & Collaboration
- Group study mode with shared stats  
- Leaderboards and friendly focus competitions

### 📱 Platform Integration
- Embed in LMS systems like Canvas or Moodle  
- API access for EdTech platforms and coaching apps

---

## Final Note

> At EvaStudy, we're not just building software—we're building better habits, healthier minds, and a future where focus is your superpower.


