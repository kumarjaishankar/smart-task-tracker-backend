# Smart Task Tracker – Backend

A FastAPI-based backend for the Smart Task Tracker app, providing a RESTful API for task management with AI-powered features.

---

## �� Live API

- **Backend API:** [https://smart-task-tracker-backend-production.up.railway.app](https://smart-task-tracker-backend-production.up.railway.app)

---

## 🛠️ Setup Steps

### Prerequisites

- Python 3.8+
- pip
- Git

### Local Setup

```bash
git clone https://github.com/kumarjaishankar/smart-task-tracker-backend.git
cd smart-task-tracker-backend
pip install -r requirements.txt
python main.py
```

> The API will run at [http://127.0.0.1:8000](http://127.0.0.1:8000).

---

## ✨ Features

- Create, read, update, and delete tasks
- Task summary endpoint (total, completed, percent completed)
- **AI-powered task enhancement and suggestions**
- **Productivity insights and analytics**
- **Smart category and priority detection**
- SQLAlchemy ORM with SQLite database
- CORS enabled for frontend integration
- FastAPI auto-generated docs at `/docs`

---

## �� API Endpoints

### Core Task Management
- `POST /tasks/` – Create a new task
- `GET /tasks/` – List all tasks
- `GET /tasks/summary` – Get task summary stats
- `GET /tasks/{task_id}` – Get a specific task
- `PUT /tasks/{task_id}` – Update a task
- `DELETE /tasks/{task_id}` – Delete a task

### AI-Powered Features
- `POST /ai/enhance-task` – Get AI suggestions for task enhancement
- `GET /ai/productivity-insights` – Get productivity analytics and recommendations

---

## 🤖 AI Integration Features

### Task Enhancement
- **Smart Title Enhancement:** AI suggests improved task titles
- **Category Detection:** Automatically categorizes tasks based on content
- **Priority Assessment:** Analyzes task urgency and suggests priority levels
- **Time Estimation:** Provides estimated completion times
- **Task Breakdown:** Suggests subtasks for complex tasks

### Productivity Analytics
- **Completion Rate Analysis:** Tracks overall productivity metrics
- **Category Distribution:** Shows which categories you focus on most
- **Priority Analysis:** Analyzes your priority management patterns
- **Smart Recommendations:** Provides personalized productivity tips
- **Productivity Score:** Calculates overall productivity rating

### Fallback System
- **Offline Intelligence:** Works even when AI APIs are unavailable
- **Rule-based Suggestions:** Uses keyword analysis for smart defaults
- **Reliable Performance:** Ensures consistent functionality

---

## 📂 Repository

- [Backend GitHub Repo](https://github.com/kumarjaishankar/smart-task-tracker-backend)

---

## ��️ Project Structure

- `main.py` – FastAPI app, endpoints, and AI integration logic
- `models.py` – SQLAlchemy models
- `database.py` – Database connection and session
- `requirements.txt` – Python dependencies

---

## 📝 Documentation

- Interactive API docs: `/docs` (when running locally or on Railway)
- Database: SQLite (default, can be swapped for other DBs with SQLAlchemy)
- AI Features: Uses Hugging Face Inference API for task enhancement

---

## 🌐 Deployment

- Deployed on Railway: [https://smart-task-tracker-backend-production.up.railway.app](https://smart-task-tracker-backend-production.up.railway.app)
