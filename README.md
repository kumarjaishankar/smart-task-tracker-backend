# Smart Task Tracker – Backend

A FastAPI-based backend for the Smart Task Tracker app, providing a RESTful API for task management.

---

## 🚀 Live API

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
- SQLAlchemy ORM with SQLite database
- CORS enabled for frontend integration
- FastAPI auto-generated docs at `/docs`

---

## 📑 API Endpoints

- `POST /tasks/` – Create a new task
- `GET /tasks/` – List all tasks
- `GET /tasks/summary` – Get task summary stats
- `GET /tasks/{task_id}` – Get a specific task
- `PUT /tasks/{task_id}` – Update a task
- `DELETE /tasks/{task_id}` – Delete a task

---

## 📂 Repository

- [Backend GitHub Repo](https://github.com/kumarjaishankar/smart-task-tracker-backend)

---

## 🗂️ Project Structure

- `main.py` – FastAPI app and endpoints
- `models.py` – SQLAlchemy models
- `database.py` – Database connection and session
- `requirements.txt` – Python dependencies

---

## 📝 Documentation

- Interactive API docs: `/docs` (when running locally or on Railway)
- Database: SQLite (default, can be swapped for other DBs with SQLAlchemy)

---

## 🌐 Deployment

- Deployed on Railway: [https://smart-task-tracker-backend-production.up.railway.app](https://smart-task-tracker-backend-production.up.railway.app)
