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
import re
from typing import Dict, Any
from collections import defaultdict, Counter
import math

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

# Enhanced Offline Intelligence System
class OfflineIntelligence:
    def __init__(self):
        # Enhanced category patterns with confidence scores
        self.category_patterns = {
            "Work": {
                "keywords": ['work', 'office', 'job', 'meeting', 'project', 'client', 'business', 'team', 'presentation', 'report', 'deadline', 'conference', 'interview', 'proposal', 'strategy', 'analysis', 'review', 'planning', 'coordination', 'collaboration'],
                "confidence": 0.9,
                "weight": 1.2
            },
            "Personal": {
                "keywords": ['personal', 'family', 'home', 'life', 'house', 'clean', 'organize', 'shopping', 'cooking', 'laundry', 'maintenance', 'decorate', 'garden', 'pet', 'child', 'parent', 'relationship', 'social', 'party', 'celebration'],
                "confidence": 0.85,
                "weight": 1.0
            },
            "Learning": {
                "keywords": ['study', 'learn', 'course', 'education', 'read', 'research', 'skill', 'training', 'workshop', 'seminar', 'tutorial', 'practice', 'exam', 'assignment', 'homework', 'certification', 'degree', 'knowledge', 'understanding', 'mastery'],
                "confidence": 0.9,
                "weight": 1.1
            },
            "Health": {
                "keywords": ['health', 'exercise', 'fitness', 'gym', 'workout', 'diet', 'meditation', 'doctor', 'appointment', 'wellness', 'nutrition', 'sleep', 'mental', 'therapy', 'recovery', 'checkup', 'medicine', 'treatment', 'rehabilitation'],
                "confidence": 0.95,
                "weight": 1.3
            },
            "Planning": {
                "keywords": ['plan', 'schedule', 'organize', 'prepare', 'review', 'strategy', 'goal', 'objective', 'timeline', 'roadmap', 'budget', 'forecast', 'analysis', 'assessment', 'evaluation', 'coordination', 'management'],
                "confidence": 0.8,
                "weight": 1.0
            },
            "Finance": {
                "keywords": ['finance', 'money', 'budget', 'investment', 'banking', 'tax', 'expense', 'income', 'savings', 'loan', 'credit', 'payment', 'billing', 'accounting', 'financial', 'economic', 'trading', 'portfolio'],
                "confidence": 0.9,
                "weight": 1.2
            },
            "Travel": {
                "keywords": ['travel', 'trip', 'vacation', 'booking', 'flight', 'hotel', 'reservation', 'itinerary', 'destination', 'tour', 'explore', 'visit', 'sightseeing', 'adventure', 'journey', 'transportation'],
                "confidence": 0.85,
                "weight": 1.1
            }
        }
        
        # Enhanced priority detection with context
        self.priority_patterns = {
            "High": {
                "urgency_words": ['urgent', 'asap', 'emergency', 'critical', 'deadline', 'today', 'now', 'immediate', 'pressing', 'crucial', 'vital', 'essential'],
                "time_indicators": ['today', 'tomorrow', 'this week', 'immediately', 'right away'],
                "importance_indicators": ['must', 'need to', 'have to', 'required', 'mandatory', 'obligatory'],
                "confidence": 0.9
            },
            "Medium": {
                "importance_words": ['important', 'priority', 'key', 'significant', 'valuable', 'worthwhile', 'beneficial'],
                "time_indicators": ['this week', 'soon', 'shortly', 'in a few days'],
                "confidence": 0.8
            },
            "Low": {
                "casual_words": ['maybe', 'sometime', 'when possible', 'if time', 'optional', 'nice to have', 'when convenient'],
                "time_indicators": ['later', 'sometime', 'when free', 'no rush'],
                "confidence": 0.7
            }
        }
        
        # Task complexity analysis
        self.complexity_indicators = {
            "simple": ['quick', 'simple', 'small', 'brief', 'basic', 'easy', 'straightforward'],
            "moderate": ['review', 'check', 'update', 'organize', 'prepare', 'plan'],
            "complex": ['project', 'analysis', 'research', 'development', 'creation', 'design', 'build', 'implement', 'integrate', 'optimize']
        }
        
        # Time estimation patterns
        self.time_patterns = {
            "quick_tasks": ['email', 'call', 'message', 'note', 'reminder', 'check'],
            "short_tasks": ['meeting', 'review', 'update', 'organize', 'prepare'],
            "medium_tasks": ['project', 'plan', 'analysis', 'research', 'design'],
            "long_tasks": ['development', 'implementation', 'creation', 'building', 'integration']
        }
    
    def analyze_task_complexity(self, title: str, description: str = "") -> Dict[str, Any]:
        """Advanced task complexity analysis"""
        text = f"{title} {description}".lower()
        
        # Count complexity indicators
        complexity_scores = {}
        for level, indicators in self.complexity_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text)
            complexity_scores[level] = score
        
        # Determine overall complexity
        if complexity_scores["complex"] > 0:
            complexity = "complex"
            estimated_hours = max(4, complexity_scores["complex"] * 2)
        elif complexity_scores["moderate"] > 0:
            complexity = "moderate"
            estimated_hours = max(2, complexity_scores["moderate"] * 1.5)
        else:
            complexity = "simple"
            estimated_hours = max(1, complexity_scores["simple"] * 0.5)
        
        return {
            "complexity": complexity,
            "estimated_hours": estimated_hours,
            "scores": complexity_scores
        }
    
    def smart_category_detection(self, title: str, description: str = "") -> Dict[str, Any]:
        """Enhanced category detection with confidence scoring"""
        text = f"{title} {description}".lower()
        
        category_scores = {}
        for category, pattern in self.category_patterns.items():
            score = 0
            keyword_matches = sum(1 for keyword in pattern["keywords"] if keyword in text)
            score = keyword_matches * pattern["weight"]
            
            # Bonus for exact matches
            if any(keyword in text.split() for keyword in pattern["keywords"]):
                score *= 1.5
            
            category_scores[category] = {
                "score": score,
                "confidence": min(0.95, score * pattern["confidence"] / 10),
                "matches": keyword_matches
            }
        
        # Find best category
        best_category = max(category_scores.items(), key=lambda x: x[1]["score"])
        
        return {
            "category": best_category[0],
            "confidence": best_category[1]["confidence"],
            "all_scores": category_scores
        }
    
    def smart_priority_detection(self, title: str, description: str = "") -> Dict[str, Any]:
        """Enhanced priority detection with context analysis"""
        text = f"{title} {description}".lower()
        
        priority_scores = {}
        for priority, pattern in self.priority_patterns.items():
            score = 0
            
            # Check urgency words
            urgency_matches = sum(1 for word in pattern.get("urgency_words", []) if word in text)
            score += urgency_matches * 3
            
            # Check time indicators
            time_matches = sum(1 for indicator in pattern.get("time_indicators", []) if indicator in text)
            score += time_matches * 2
            
            # Check importance indicators
            importance_matches = sum(1 for indicator in pattern.get("importance_indicators", []) if indicator in text)
            score += importance_matches * 2
            
            # Check casual words (negative scoring for low priority)
            if priority == "Low":
                casual_matches = sum(1 for word in pattern.get("casual_words", []) if word in text)
                score += casual_matches * 2
            
            priority_scores[priority] = {
                "score": score,
                "confidence": min(0.95, score * pattern["confidence"] / 5)
            }
        
        # Find best priority
        best_priority = max(priority_scores.items(), key=lambda x: x[1]["score"])
        
        return {
            "priority": best_priority[0],
            "confidence": best_priority[1]["confidence"],
            "all_scores": priority_scores
        }
    
    def generate_smart_suggestions(self, title: str, description: str = "") -> Dict[str, Any]:
        """Generate comprehensive smart suggestions using offline intelligence"""
        # Analyze task complexity
        complexity_analysis = self.analyze_task_complexity(title, description)
        
        # Smart category detection
        category_analysis = self.smart_category_detection(title, description)
        
        # Smart priority detection
        priority_analysis = self.smart_priority_detection(title, description)
        
        # Enhanced title creation
        enhanced_title = self.create_enhanced_title(title, category_analysis["category"])
        
        # Generate task breakdown
        task_breakdown = self.generate_task_breakdown(title, description, complexity_analysis["complexity"])
        
        # Generate insights
        insights = self.generate_insights(title, description, category_analysis, priority_analysis, complexity_analysis)
        
        return {
            "enhanced_title": enhanced_title,
            "suggested_category": category_analysis["category"],
            "suggested_priority": priority_analysis["priority"],
            "estimated_time": complexity_analysis["estimated_hours"],
            "task_breakdown": task_breakdown,
            "ai_insights": insights,
            "confidence_scores": {
                "category_confidence": category_analysis["confidence"],
                "priority_confidence": priority_analysis["confidence"],
                "complexity": complexity_analysis["complexity"]
            }
        }
    
    def create_enhanced_title(self, original_title: str, category: str) -> str:
        """Create enhanced title based on category and patterns"""
        title = original_title.strip().lower()
        
        # Category-specific enhancements
        category_enhancements = {
            "Work": {
                "meeting": "Schedule and prepare for team meeting",
                "call": "Make important business call",
                "email": "Draft and send professional email",
                "review": "Review and analyze work documents",
                "project": "Plan and execute project milestone",
                "report": "Create and submit detailed report"
            },
            "Personal": {
                "clean": "Clean and organize living space",
                "shop": "Go shopping for household essentials",
                "cook": "Prepare and cook healthy meal",
                "organize": "Organize personal belongings",
                "maintenance": "Complete home maintenance task"
            },
            "Learning": {
                "study": "Study and review course materials",
                "read": "Read and take detailed notes",
                "practice": "Practice and improve skills",
                "research": "Research and gather information",
                "learn": "Learn new concept or skill"
            },
            "Health": {
                "exercise": "Complete workout routine",
                "meditation": "Practice mindfulness meditation",
                "diet": "Plan and prepare healthy meals",
                "doctor": "Schedule and attend medical appointment",
                "wellness": "Focus on overall wellness"
            }
        }
        
        # Get category-specific patterns
        patterns = category_enhancements.get(category, {})
        
        # Find matching enhancement
        for key, enhancement in patterns.items():
            if key in title:
                return enhancement
        
        # Generic enhancement if no category-specific match
        if len(title) < 10:
            return f"Complete {title} task"
        elif not any(word in title for word in ['complete', 'finish', 'do', 'make', 'create', 'prepare']):
            return f"Complete {title}"
        else:
            return original_title.title()
    
    def generate_task_breakdown(self, title: str, description: str, complexity: str) -> List[str]:
        """Generate subtasks based on complexity"""
        breakdown = []
        
        if complexity == "complex":
            breakdown = [
                "Research and gather information",
                "Plan and organize approach",
                "Execute main task",
                "Review and refine results",
                "Document and summarize"
            ]
        elif complexity == "moderate":
            breakdown = [
                "Prepare and organize",
                "Complete main task",
                "Review and finalize"
            ]
        else:
            breakdown = [
                "Complete task efficiently"
            ]
        
        return breakdown
    
    def generate_insights(self, title: str, description: str, category_analysis: Dict, priority_analysis: Dict, complexity_analysis: Dict) -> str:
        """Generate intelligent insights about the task"""
        insights = []
        
        # Category insights
        if category_analysis["confidence"] > 0.8:
            insights.append(f"Task appears to be {category_analysis['category']}-related with high confidence")
        
        # Priority insights
        if priority_analysis["priority"] == "High":
            insights.append("This task requires immediate attention")
        elif priority_analysis["priority"] == "Low":
            insights.append("This task can be scheduled for later")
        
        # Complexity insights
        if complexity_analysis["complexity"] == "complex":
            insights.append("Consider breaking this complex task into smaller steps")
        elif complexity_analysis["complexity"] == "simple":
            insights.append("This task should be quick to complete")
        
        # Time insights
        if complexity_analysis["estimated_hours"] > 4:
            insights.append(f"Estimated time: {complexity_analysis['estimated_hours']} hours - plan accordingly")
        
        return ". ".join(insights) if insights else "Task analyzed using offline intelligence system."

# Initialize offline intelligence
offline_intelligence = OfflineIntelligence()

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
            # Fallback to enhanced offline intelligence
            return get_enhanced_fallback_suggestions(title, description)
            
    except Exception as e:
        print(f"AI API error: {e}")
        return get_enhanced_fallback_suggestions(title, description)

def get_enhanced_fallback_suggestions(title: str, description: str) -> Dict[str, Any]:
    """Enhanced fallback suggestions using offline intelligence"""
    return offline_intelligence.generate_smart_suggestions(title, description)

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

def get_productivity_insights(tasks: List[models.Task]) -> Dict[str, Any]:
    """Generate enhanced productivity insights using offline intelligence"""
    if not tasks:
        return {"message": "No tasks available for analysis"}
    
    total_tasks = len(tasks)
    completed_tasks = len([t for t in tasks if t.completed])
    completion_rate = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
    
    # Enhanced category analysis with confidence
    categories = {}
    category_completion_rates = {}
    for task in tasks:
        cat = task.category or "Uncategorized"
        categories[cat] = categories.get(cat, 0) + 1
        
        # Track completion rates per category
        if cat not in category_completion_rates:
            category_completion_rates[cat] = {"total": 0, "completed": 0}
        category_completion_rates[cat]["total"] += 1
        if task.completed:
            category_completion_rates[cat]["completed"] += 1
    
    # Enhanced priority analysis
    priorities = {"High": 0, "Medium": 0, "Low": 0}
    priority_completion_rates = {"High": 0, "Medium": 0, "Low": 0}
    for task in tasks:
        priority = task.priority or "Medium"
        priorities[priority] += 1
        if task.completed:
            priority_completion_rates[priority] += 1
    
    # Calculate completion rates
    for priority in priority_completion_rates:
        if priorities[priority] > 0:
            priority_completion_rates[priority] = (priority_completion_rates[priority] / priorities[priority]) * 100
    
    # Find most productive category
    most_productive_category = "None"
    best_completion_rate = 0
    for category, stats in category_completion_rates.items():
        if stats["total"] >= 2:  # Only consider categories with at least 2 tasks
            completion_rate = (stats["completed"] / stats["total"]) * 100
            if completion_rate > best_completion_rate:
                best_completion_rate = completion_rate
                most_productive_category = category
    
    # Enhanced productivity score calculation
    base_score = completion_rate
    priority_bonus = (priorities["High"] / total_tasks) * 20 if total_tasks > 0 else 0
    category_bonus = min(20, len(categories) * 2)  # Bonus for working across categories
    productivity_score = min(100, base_score + priority_bonus + category_bonus)
    
    # Generate intelligent recommendations
    recommendations = []
    
    # Completion rate recommendations
    if completion_rate < 30:
        recommendations.append("Your completion rate is low. Consider setting smaller, more achievable goals")
    elif completion_rate < 60:
        recommendations.append("Try breaking down complex tasks into smaller, manageable pieces")
    elif completion_rate > 80:
        recommendations.append("Excellent productivity! Consider taking on more challenging tasks")
    
    # Priority management recommendations
    high_priority_ratio = priorities["High"] / total_tasks if total_tasks > 0 else 0
    if high_priority_ratio > 0.4:
        recommendations.append("You have many high-priority tasks. Consider delegating or rescheduling some")
    elif high_priority_ratio < 0.1:
        recommendations.append("Consider adding more high-priority tasks to focus on important goals")
    
    # Category balance recommendations
    if len(categories) > 6:
        recommendations.append("You're working across many categories. Consider focusing on 2-3 main areas for better results")
    elif len(categories) < 3:
        recommendations.append("Consider diversifying your tasks across different life areas for better balance")
    
    # Time management insights
    if priorities["High"] > 0:
        high_priority_completion = priority_completion_rates["High"]
        if high_priority_completion < 50:
            recommendations.append("Focus on completing high-priority tasks first to improve overall productivity")
    
    # Category-specific insights
    if most_productive_category != "None":
        recommendations.append(f"Your most productive area is {most_productive_category}. Leverage this strength")
    
    # Add default recommendation if none generated
    if not recommendations:
        recommendations.append("Keep up the good work! Your productivity patterns look balanced")
    
    insights = {
        "total_tasks": total_tasks,
        "completed_tasks": completed_tasks,
        "completion_rate": round(completion_rate, 1),
        "category_distribution": categories,
        "priority_distribution": priorities,
        "most_productive_category": most_productive_category,
        "productivity_score": round(productivity_score, 1),
        "recommendations": recommendations,
        "detailed_insights": {
            "category_completion_rates": category_completion_rates,
            "priority_completion_rates": priority_completion_rates,
            "high_priority_ratio": round(high_priority_ratio * 100, 1),
            "category_diversity": len(categories)
        }
    }
    
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

# Enhanced Offline Intelligence endpoint
@app.post("/ai/offline-enhance", response_model=AISuggestions)
def offline_enhance_task(request: TaskEnhancementRequest):
    """Get enhanced offline intelligence suggestions (no external API required)"""
    return offline_intelligence.generate_smart_suggestions(request.title, request.description)

# Offline Intelligence Status endpoint
@app.get("/ai/offline-status")
def get_offline_status():
    """Get information about offline intelligence capabilities"""
    return {
        "offline_intelligence": True,
        "capabilities": {
            "smart_category_detection": True,
            "priority_analysis": True,
            "complexity_analysis": True,
            "time_estimation": True,
            "task_breakdown": True,
            "productivity_insights": True
        },
        "categories_supported": list(offline_intelligence.category_patterns.keys()),
        "confidence_threshold": 0.8,
        "description": "Enhanced offline intelligence system with ML-like decision trees and pattern recognition"
    }

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