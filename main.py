from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import models
import database
from pydantic import BaseModel
from datetime import datetime
import uvicorn
import requests
import json
import os
from typing import Dict, Any

models.Base.metadata.create_all(bind=database.engine)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://smart-task-tracker-frontend.vercel.app",
        "https://smart-task-tracker-frontend-beta.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# AI Helper Functions
def enhance_task_with_ai(title: str, description: str = "") -> Dict[str, Any]:
    """Enhance task with AI suggestions using free APIs"""
    try:
        # Using Hugging Face Inference API (free tier)
        API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
        headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY', 'hf_demo')}"}
        
        # Create a prompt for task enhancement
        prompt = f"Task: {title}\nDescription: {description}\n\nPlease provide:\n1. Enhanced title\n2. Suggested category\n3. Priority level (Low/Medium/High)\n4. Estimated completion time (in hours)\n5. Task breakdown (if complex)"
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": 200,
                "temperature": 0.7
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            enhanced_text = result[0].get('generated_text', '')
            
            # Parse the AI response (simplified parsing)
            suggestions = {
                "enhanced_title": title,  # Fallback to original
                "suggested_category": "General",
                "suggested_priority": "Medium",
                "estimated_time": 1,
                "task_breakdown": [],
                "ai_insights": enhanced_text[:100] + "..." if len(enhanced_text) > 100 else enhanced_text
            }
            
            # Simple keyword-based category detection
            task_text = f"{title} {description}".lower()
            if any(word in task_text for word in ['work', 'office', 'job', 'meeting', 'project']):
                suggestions["suggested_category"] = "Work"
            elif any(word in task_text for word in ['personal', 'family', 'home', 'life']):
                suggestions["suggested_category"] = "Personal"
            elif any(word in task_text for word in ['study', 'learn', 'course', 'education']):
                suggestions["suggested_category"] = "Learning"
            elif any(word in task_text for word in ['health', 'exercise', 'fitness', 'gym']):
                suggestions["suggested_category"] = "Health"
            
            # Priority detection based on urgency words
            urgency_words = ['urgent', 'asap', 'emergency', 'critical', 'deadline']
            if any(word in task_text for word in urgency_words):
                suggestions["suggested_priority"] = "High"
            elif any(word in task_text for word in ['important', 'priority']):
                suggestions["suggested_priority"] = "Medium"
            else:
                suggestions["suggested_priority"] = "Low"
            
            return suggestions
        else:
            # Fallback to rule-based suggestions
            return get_fallback_suggestions(title, description)
            
    except Exception as e:
        print(f"AI API error: {e}")
        return get_fallback_suggestions(title, description)

def get_fallback_suggestions(title: str, description: str) -> Dict[str, Any]:
    """Fallback suggestions when AI API is unavailable"""
    task_text = f"{title} {description}".lower()
    
    # Category detection
    category = "General"
    if any(word in task_text for word in ['work', 'office', 'job', 'meeting', 'project']):
        category = "Work"
    elif any(word in task_text for word in ['personal', 'family', 'home', 'life']):
        category = "Personal"
    elif any(word in task_text for word in ['study', 'learn', 'course', 'education']):
        category = "Learning"
    elif any(word in task_text for word in ['health', 'exercise', 'fitness', 'gym']):
        category = "Health"
    
    # Priority detection
    priority = "Medium"
    urgency_words = ['urgent', 'asap', 'emergency', 'critical', 'deadline']
    if any(word in task_text for word in urgency_words):
        priority = "High"
    elif any(word in task_text for word in ['important', 'priority']):
        priority = "Medium"
    else:
        priority = "Low"
    
    return {
        "enhanced_title": title,
        "suggested_category": category,
        "suggested_priority": priority,
        "estimated_time": 1,
        "task_breakdown": [],
        "ai_insights": "AI suggestions temporarily unavailable. Using smart defaults."
    }

def get_productivity_insights(tasks: List[models.Task]) -> Dict[str, Any]:
    """Generate productivity insights from task data"""
    if not tasks:
        return {"message": "No tasks available for analysis"}
    
    total_tasks = len(tasks)
    completed_tasks = len([t for t in tasks if t.completed])
    completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
    
    # Category analysis
    categories = {}
    for task in tasks:
        cat = task.category or "Uncategorized"
        categories[cat] = categories.get(cat, 0) + 1
    
    # Priority analysis
    priorities = {"High": 0, "Medium": 0, "Low": 0}
    for task in tasks:
        priority = task.priority or "Medium"
        priorities[priority] += 1
    
    # Most productive category
    most_productive_category = max(categories.items(), key=lambda x: x[1])[0] if categories else "None"
    
    insights = {
        "total_tasks": total_tasks,
        "completed_tasks": completed_tasks,
        "completion_rate": round(completion_rate, 1),
        "category_distribution": categories,
        "priority_distribution": priorities,
        "most_productive_category": most_productive_category,
        "productivity_score": min(100, completion_rate + (priorities["High"] * 10)),
        "recommendations": []
    }
    
    # Generate recommendations
    if completion_rate < 50:
        insights["recommendations"].append("Consider breaking down large tasks into smaller, manageable pieces")
    if priorities["High"] > total_tasks * 0.3:
        insights["recommendations"].append("You have many high-priority tasks. Consider delegating some tasks")
    if len(categories) > 5:
        insights["recommendations"].append("You're working across many categories. Consider focusing on 2-3 main areas")
    
    return insights

# Pydantic Schemas
class TaskBase(BaseModel):
    title: str
    description: Optional[str] = None
    category: Optional[str] = None
    due_date: Optional[datetime] = None
    priority: Optional[str] = "Medium"

class TaskCreate(TaskBase):
    pass

class TaskUpdate(TaskBase):
    completed: Optional[bool] = None

class TaskOut(TaskBase):
    id: int
    completed: bool
    created_at: Optional[datetime]
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True

class AISuggestions(BaseModel):
    enhanced_title: str
    suggested_category: str
    suggested_priority: str
    estimated_time: int
    task_breakdown: List[str]
    ai_insights: str

class ProductivityInsights(BaseModel):
    total_tasks: int
    completed_tasks: int
    completion_rate: float
    category_distribution: Dict[str, int]
    priority_distribution: Dict[str, int]
    most_productive_category: str
    productivity_score: float
    recommendations: List[str]

class TaskEnhancementRequest(BaseModel):
    title: str
    description: str = ""

# CRUD Endpoints
@app.post("/tasks/", response_model=TaskOut)
def create_task(task: TaskCreate, db: Session = Depends(get_db)):
    db_task = models.Task(**task.model_dump())
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    return db_task

@app.get("/tasks/", response_model=List[TaskOut])
def read_tasks(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return db.query(models.Task).offset(skip).limit(limit).all()

# Summary endpoint 
@app.get("/tasks/summary")
def get_summary(db: Session = Depends(get_db)):
    total = db.query(models.Task).count()
    completed = db.query(models.Task).filter(models.Task.completed == True).count()
    percent_completed = (completed / total * 100) if total else 0
    return {
        "total": total,
        "completed": completed,
        "percent_completed": percent_completed
    }

# AI Enhancement endpoint
@app.post("/ai/enhance-task", response_model=AISuggestions)
def enhance_task(request: TaskEnhancementRequest):
    """Get AI suggestions for task enhancement"""
    return enhance_task_with_ai(request.title, request.description)

# Productivity Insights endpoint
@app.get("/ai/productivity-insights", response_model=ProductivityInsights)
def get_insights(db: Session = Depends(get_db)):
    """Get AI-powered productivity insights"""
    tasks = db.query(models.Task).all()
    return get_productivity_insights(tasks)

@app.get("/tasks/{task_id}", response_model=TaskOut)
def read_task(task_id: int, db: Session = Depends(get_db)):
    task = db.query(models.Task).filter(models.Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task

@app.put("/tasks/{task_id}", response_model=TaskOut)
def update_task(task_id: int, task: TaskUpdate, db: Session = Depends(get_db)):
    db_task = db.query(models.Task).filter(models.Task.id == task_id).first()
    if not db_task:
        raise HTTPException(status_code=404, detail="Task not found")
    for key, value in task.model_dump(exclude_unset=True).items():
        setattr(db_task, key, value)
    db.commit()
    db.refresh(db_task)
    return db_task

@app.delete("/tasks/{task_id}")
def delete_task(task_id: int, db: Session = Depends(get_db)):
    db_task = db.query(models.Task).filter(models.Task.id == task_id).first()
    if not db_task:
        raise HTTPException(status_code=404, detail="Task not found")
    db.delete(db_task)
    db.commit()
    return {"ok": True}

@app.get("/")
def read_root():
    return {"message": "Backend is running!"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)