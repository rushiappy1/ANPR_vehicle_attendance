import os
import json
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func, and_

# DATABASE IMPORTS - YOUR LOCALHOST SETUP
from database import get_db, VehicleRegistration, VehicleAttendance, engine
import database  # Auto-create tables on startup

# Import your pipeline
try:
    from main_pipeline import run_pipeline
except ImportError:
    print("[WARNING] main_pipeline not found - using dummy for testing")
    def run_pipeline(image_path: str, plate_text: str):
        return {
            "input_image": os.path.basename(image_path),
            "expected_plate": plate_text,
            "yolo_crop_base_name": "test_crop.jpg",
            "crnn_predictions": [{"image": "test.jpg", "prediction": plate_text}]
        }

# CONFIG & SETUP
app = FastAPI(title="Vehicle Attendance System - SQL DATABASE")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories
os.makedirs("static", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

# DATABASE HELPER FUNCTIONS

def normalize_vehicle_number(vehicle: str) -> str:
    """Normalize: strip whitespace, uppercase"""
    return str(vehicle).strip().upper()

def is_vehicle_registered(db: Session, vehicle_number: str) -> bool:
    """Check if vehicle exists in DATABASE - CASE INSENSITIVE"""
    normalized = normalize_vehicle_number(vehicle_number)
    print(f"[DB] Checking registration for: '{normalized}'")
    exists = db.query(VehicleRegistration).filter(
        func.upper(VehicleRegistration.vehicle_number) == normalized
    ).first() is not None
    print(f"[DB] Result: {'REGISTERED' if exists else ' NOT REGISTERED'}")
    return exists

def get_vehicle_owner(db: Session, vehicle_number: str) -> str:
    """Get owner from DATABASE"""
    normalized = normalize_vehicle_number(vehicle_number)
    vehicle = db.query(VehicleRegistration).filter(
        func.upper(VehicleRegistration.vehicle_number) == normalized
    ).first()
    owner = vehicle.owner_name if vehicle else "Unknown"
    print(f"[DB] Owner for '{normalized}': '{owner}'")
    return owner

def save_registration(db: Session, vehicle_number: str, owner_name: str, image_file: str) -> bool:
    """Save to DATABASE"""
    try:
        normalized = normalize_vehicle_number(vehicle_number)
        registration = VehicleRegistration(
            vehicle_number=normalized,
            owner_name=owner_name,
            registration_date=datetime.now().strftime('%Y-%m-%d'),
            image_filename=image_file
        )
        db.add(registration)
        db.commit()
        db.refresh(registration)
        print(f"[DB] Registration saved: {normalized} (ID: {registration.id})")
        return True
    except Exception as e:
        db.rollback()
        print(f"[DB ERROR] Registration failed: {e}")
        return False

def save_attendance(db: Session, vehicle_number: str, owner_name: str, image_file: str) -> bool:
    """Save attendance to DATABASE"""
    try:
        normalized = normalize_vehicle_number(vehicle_number)
        attendance = VehicleAttendance(
            vehicle_number=normalized,
            owner_name=owner_name,
            attendance_date=datetime.now().strftime('%Y-%m-%d'),
            image_filename=image_file
        )
        db.add(attendance)
        db.commit()
        db.refresh(attendance)
        print(f"[DB] Attendance saved: {normalized} (ID: {attendance.id})")
        return True
    except Exception as e:
        db.rollback()
        print(f"[DB ERROR] Attendance failed: {e}")
        return False

def load_html_file(filename: str) -> str:
    """Load HTML with multiple fallback paths"""
    ui_paths = [
        f"ui/{filename}",
        f"/home/roshan/ANPR_PIP_LINE/ui/{filename}",
        f"./ui/{filename}",
        os.path.join("ui", filename),
    ]
    
    for path in ui_paths:
        try:
            if os.path.exists(path):
                with open(path, 'r', encoding='utf-8') as f:
                    return f.read()
        except Exception as e:
            print(f"[DEBUG] Failed to load {path}: {e}")
    
    return f"<h1>Error: {filename} not found</h1><p>Paths tried: {ui_paths}</p>"

# PAGE ROUTES

@app.get("/", response_class=HTMLResponse)
async def home():
    return load_html_file("home.html")

@app.get("/register", response_class=HTMLResponse)
async def register_page():
    return load_html_file("register.html")

@app.get("/attendance", response_class=HTMLResponse)
async def attendance_page():
    return load_html_file("attendance.html")

# API ENDPOINTS - REGISTRATION
@app.post("/api/register")
async def register_vehicle(
    image: UploadFile = File(...),
    vehicle_number: str = Form(...),
    owner_name: str = Form(default="Unknown"),
    db: Session = Depends(get_db)
):
    """Register new vehicle - FULL DATABASE VERSION"""
    try:
        normalized_vehicle = normalize_vehicle_number(vehicle_number)
        
        if not normalized_vehicle:
            raise HTTPException(status_code=400, detail="Invalid vehicle number")
        
        # Save uploaded image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"register_{timestamp}_{normalized_vehicle}.jpg"
        image_path = f"uploads/{filename}"
        
        content = await image.read()
        with open(image_path, 'wb') as f:
            f.write(content)
        
        print(f"[REGISTER] Saved image: {image_path}")
        
        # Check if already registered
        if is_vehicle_registered(db, normalized_vehicle):
            raise HTTPException(
                status_code=400, 
                detail=f"Vehicle {normalized_vehicle} already registered!"
            )
        
        # Run your YOLO+CRNN pipeline
        print(f"[PIPELINE] Processing {image_path}")
        result = run_pipeline(image_path, normalized_vehicle)
        
        if result is None:
            raise HTTPException(status_code=500, detail="Pipeline failed")
        
        # Save to DATABASE
        if save_registration(db, normalized_vehicle, owner_name, filename):
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": f"Vehicle {normalized_vehicle} registered successfully!",
                    "vehicle_number": normalized_vehicle,
                    "owner_name": owner_name,
                    "image_filename": filename,
                    "pipeline_result": result
                }
            )
        else:
            raise HTTPException(status_code=500, detail="Database save failed")
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Registration failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Server error: {str(e)}"}
        )

# API ENDPOINTS - ATTENDANCE
@app.post("/api/mark-attendance")
async def mark_attendance(
    image: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Mark attendance - FULL DATABASE VERSION"""
    try:
        # Save image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"attendance_{timestamp}.jpg"
        image_path = f"uploads/{filename}"
        
        content = await image.read()
        with open(image_path, 'wb') as f:
            f.write(content)
        
        print(f"[ATTENDANCE] Processing: {image_path}")
        
        # Run pipeline to detect plate
        result = run_pipeline(image_path, "UNKNOWN")
        if not result or not result.get('crnn_predictions'):
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "No vehicle plate detected"}
            )
        
        # Extract detected vehicle number
        detected_vehicle = normalize_vehicle_number(
            result['crnn_predictions'][0].get('prediction', '')
        )
        print(f"[ATTENDANCE] Detected vehicle: '{detected_vehicle}'")
        
        if not detected_vehicle or len(detected_vehicle) < 5:
            return JSONResponse(
                status_code=400,
                content={"success": False, "message": "Invalid vehicle number detected"}
            )
        
        # Check if registered
        if not is_vehicle_registered(db, detected_vehicle):
            return JSONResponse(
                status_code=403,
                content={
                    "success": False,
                    "message": f"Vehicle {detected_vehicle} is NOT registered!",
                    "detected_vehicle": detected_vehicle,
                    "action": "Please register this vehicle first"
                }
            )
        
        # Get owner and save attendance
        owner_name = get_vehicle_owner(db, detected_vehicle)
        if save_attendance(db, detected_vehicle, owner_name, filename):
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": f"Attendance marked for {detected_vehicle}!",
                    "vehicle_number": detected_vehicle,
                    "owner_name": owner_name,
                    "image_filename": filename,
                    "timestamp": datetime.now().isoformat()
                }
            )
        else:
            return JSONResponse(
                status_code=500,
                content={"success": False, "message": "Failed to save attendance"}
            )
            
    except Exception as e:
        print(f"[ERROR] Attendance failed: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "message": f"Server error: {str(e)}"}
        )

# REPORTS & DEBUG ENDPOINTS
@app.get("/api/registered-vehicles")
async def get_registered_vehicles(db: Session = Depends(get_db), skip: int = 0, limit: int = 100):
    """Get all registered vehicles from DATABASE"""
    vehicles = db.query(VehicleRegistration).offset(skip).limit(limit).all()
    return {
        "success": True,
        "count": len(vehicles),
        "vehicles": [
            {
                "id": v.id,
                "vehicle_number": v.vehicle_number,
                "owner_name": v.owner_name,
                "registration_date": v.registration_date,
                "image_filename": v.image_filename,
                "created_at": v.created_at.isoformat() if v.created_at else None
            }
            for v in vehicles
        ]
    }

@app.get("/api/today-attendance")
async def get_today_attendance(db: Session = Depends(get_db)):
    """Today's attendance from DATABASE"""
    today = datetime.now().strftime('%Y-%m-%d')
    attendances = db.query(VehicleAttendance).filter(
        VehicleAttendance.attendance_date == today
    ).order_by(VehicleAttendance.attendance_time.desc()).all()
    
    return {
        "success": True,
        "date": today,
        "count": len(attendances),
        "attendances": [
            {
                "id": a.id,
                "vehicle_number": a.vehicle_number,
                "owner_name": a.owner_name,
                "attendance_time": a.attendance_time.strftime('%H:%M:%S') if a.attendance_time else None,
                "image_filename": a.image_filename,
                "confidence_score": a.confidence_score
            }
            for a in attendances
        ]
    }

@app.get("/api/debug-db")
async def debug_database(db: Session = Depends(get_db)):
    """Debug endpoint - shows DB contents"""
    reg_count = db.query(VehicleRegistration).count()
    att_count = db.query(VehicleAttendance).count()
    
    sample_reg = db.query(VehicleRegistration).limit(5).all()
    sample_att = db.query(VehicleAttendance).limit(5).all()
    
    return {
        "success": True,
        "stats": {
            "registrations": reg_count,
            "attendance_records": att_count,
            "database_file": os.path.exists("anpr.db")
        },
        "sample_registrations": [
            {"vehicle_number": v.vehicle_number, "owner_name": v.owner_name}
            for v in sample_reg
        ],
        "sample_attendance": [
            {"vehicle_number": a.vehicle_number, "time": str(a.attendance_time)}
            for a in sample_att
        ]
    }

@app.get("/api/health")
async def health_check(db: Session = Depends(get_db)):
    """System health check"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "database": {
            "connected": True,
            "file_exists": os.path.exists("anpr.db")
        },
        "storage": {
            "uploads_dir": os.path.exists("uploads"),
            "static_dir": os.path.exists("static")
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting ANPR Attendance System with SQL Database...")
    print("Database: ./anpr.db")
    print("Access: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
