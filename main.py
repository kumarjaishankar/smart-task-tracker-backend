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
        
        # Create a better prompt for task enhancement
        prompt = f"""Task Enhancement Request:
Original Task: {title}
Description: {description}

Please enhance this task by:
1. Making the title more specific and actionable
2. Suggesting the best category
3. Determining appropriate priority level
4. Estimating completion time

Example enhancement:
Original: "meeting"
Enhanced: "Schedule team meeting for project review"
Category: Work
Priority: Medium
Time: 1 hour

Please provide your enhancement:"""
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": 300,
                "temperature": 0.8,
                "do_sample": True
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            enhanced_text = result[0].get('generated_text', '')
            
            # Try to extract enhanced title from AI response
            enhanced_title = extract_enhanced_title(title, enhanced_text)
            
            # Parse the AI response and extract suggestions
            suggestions = {
                "enhanced_title": enhanced_title,
                "suggested_category": "General",
                "suggested_priority": "Medium",
                "estimated_time": 1,
                "task_breakdown": [],
                "ai_insights": enhanced_text[:150] + "..." if len(enhanced_text) > 150 else enhanced_text
            }
            
            # Enhanced keyword-based category detection
            task_text = f"{title} {description}".lower()
            if any(word in task_text for word in ['work', 'office', 'job', 'meeting', 'project', 'client', 'business', 'team']):
                suggestions["suggested_category"] = "Work"
            elif any(word in task_text for word in ['personal', 'family', 'home', 'life', 'house', 'clean', 'organize']):
                suggestions["suggested_category"] = "Personal"
            elif any(word in task_text for word in ['study', 'learn', 'course', 'education', 'read', 'research', 'skill']):
                suggestions["suggested_category"] = "Learning"
            elif any(word in task_text for word in ['health', 'exercise', 'fitness', 'gym', 'workout', 'diet', 'meditation']):
                suggestions["suggested_category"] = "Health"
            elif any(word in task_text for word in ['plan', 'schedule', 'organize', 'prepare', 'review']):
                suggestions["suggested_category"] = "Planning"
            
            # Enhanced priority detection
            urgency_words = ['urgent', 'asap', 'emergency', 'critical', 'deadline', 'today', 'now']
            importance_words = ['important', 'priority', 'key', 'essential', 'crucial']
            
            if any(word in task_text for word in urgency_words):
                suggestions["suggested_priority"] = "High"
            elif any(word in task_text for word in importance_words):
                suggestions["suggested_priority"] = "Medium"
            else:
                suggestions["suggested_priority"] = "Low"
            
            # Estimate time based on task complexity
            time_estimate = estimate_task_time(title, description)
            suggestions["estimated_time"] = time_estimate
            
            return suggestions
        else:
            # Fallback to rule-based suggestions
            return get_fallback_suggestions(title, description)
            
    except Exception as e:
        print(f"AI API error: {e}")
        return get_fallback_suggestions(title, description)

def extract_enhanced_title(original_title: str, ai_response: str) -> str:
    """Extract enhanced title from AI response or create a better one"""
    # If AI response is too short or unclear, create an enhanced title manually
    if len(ai_response) < 20 or "enhanced" not in ai_response.lower():
        return create_enhanced_title(original_title)
    
    # Try to extract title from AI response
    lines = ai_response.split('\n')
    for line in lines:
        if 'enhanced' in line.lower() and ':' in line:
            enhanced_part = line.split(':')[-1].strip()
            if len(enhanced_part) > 3:
                return enhanced_part
    
    # Fallback to manual enhancement
    return create_enhanced_title(original_title)

def create_enhanced_title(original_title: str) -> str:
    """Create an enhanced title manually based on common patterns"""
    title = original_title.strip().lower()
    
    # Common task enhancements
    enhancements = {
        'meeting': 'Schedule and prepare for team meeting',
        'call': 'Make important phone call',
        'email': 'Draft and send email',
        'review': 'Review and analyze documents',
        'plan': 'Create detailed plan',
        'study': 'Study and review materials',
        'exercise': 'Complete workout routine',
        'clean': 'Clean and organize space',
        'shop': 'Go shopping for essentials',
        'cook': 'Prepare and cook meal',
        'read': 'Read and take notes',
        'write': 'Write and edit content',
        'research': 'Research and gather information',
        'organize': 'Organize and sort items',
        'prepare': 'Prepare and set up',
        'check': 'Check and verify information',
        'update': 'Update and maintain records',
        'create': 'Create and develop content',
        'design': 'Design and create layout',
        'build': 'Build and construct project'
    }
    
    # Find matching enhancement
    for key, enhancement in enhancements.items():
        if key in title:
            return enhancement
    
    # If no specific match, make it more actionable
    if len(title) < 10:
        return f"Complete {title} task"
    elif not any(word in title for word in ['complete', 'finish', 'do', 'make', 'create', 'prepare']):
        return f"Complete {title}"
    else:
        return original_title.title()

def estimate_task_time(title: str, description: str = "") -> int:
    """Estimate task completion time in hours"""
    task_text = f"{title} {description}".lower()
    
    # Time estimation based on keywords and complexity
    if any(word in task_text for word in ['quick', 'simple', 'small', 'brief']):
        return 1
    elif any(word in task_text for word in ['meeting', 'call', 'email', 'review']):
        return 1
    elif any(word in task_text for word in ['project', 'plan', 'organize', 'prepare']):
        return 2
    elif any(word in task_text for word in ['study', 'research', 'analysis', 'report']):
        return 3
    elif any(word in task_text for word in ['complex', 'major', 'large', 'extensive']):
        return 4
    else:
        return 1

def get_fallback_suggestions(title: str, description: str) -> Dict[str, Any]:
    """Fallback suggestions when AI API is unavailable"""
    task_text = f"{title} {description}".lower()
    
    # Enhanced category detection
    category = "General"
    if any(word in task_text for word in ['work', 'office', 'job', 'meeting', 'project', 'client', 'business', 'team']):
        category = "Work"
    elif any(word in task_text for word in ['personal', 'family', 'home', 'life', 'house', 'clean', 'organize']):
        category = "Personal"
    elif any(word in task_text for word in ['study', 'learn', 'course', 'education', 'read', 'research', 'skill']):
        category = "Learning"
    elif any(word in task_text for word in ['health', 'exercise', 'fitness', 'gym', 'workout', 'diet', 'meditation']):
        category = "Health"
    elif any(word in task_text for word in ['plan', 'schedule', 'organize', 'prepare', 'review']):
        category = "Planning"
    
    # Enhanced priority detection
    priority = "Medium"
    urgency_words = ['urgent', 'asap', 'emergency', 'critical', 'deadline', 'today', 'now']
    importance_words = ['important', 'priority', 'key', 'essential', 'crucial']
    
    if any(word in task_text for word in urgency_words):
        priority = "High"
    elif any(word in task_text for word in importance_words):
        priority = "Medium"
    else:
        priority = "Low"
    
    # Create enhanced title
    enhanced_title = create_enhanced_title(title)
    
    # Estimate time
    time_estimate = estimate_task_time(title, description)
    
    return {
        "enhanced_title": enhanced_title,
        "suggested_category": category,
        "suggested_priority": priority,
        "estimated_time": time_estimate,
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