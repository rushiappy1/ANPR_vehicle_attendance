from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.sql import func
from .database import Base

class VehicleRegistration(Base):
    __tablename__ = "vehicle_registrations"
    
    id = Column(Integer, primary_key=True, index=True)
    vehicle_number = Column(String(20), unique=True, index=True)
    owner_name = Column(String(100))
    registration_time = Column(DateTime, server_default=func.now())
    registration_date = Column(String(10))
    image_filename = Column(String(255))
    created_at = Column(DateTime, server_default=func.now())

class VehicleAttendance(Base):
    __tablename__ = "vehicle_attendance"
    
    id = Column(Integer, primary_key=True, index=True)
    vehicle_number = Column(String(20), index=True)
    owner_name = Column(String(100))
    attendance_time = Column(DateTime, server_default=func.now())
    attendance_date = Column(String(10))
    image_filename = Column(String(255))
    confidence_score = Column(String(10), default="0.85")
    created_at = Column(DateTime, server_default=func.now())
