import os
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

DB_URL = os.getenv("DB_URL", "sqlite:///data/requests.db")
engine = create_engine(DB_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Request(Base):
    __tablename__ = "requests"
    id = Column(Integer, primary_key=True, index=True)
    request_id = Column(String, unique=True, index=True)
    user_name = Column(String)
    application = Column(String)
    version = Column(String, nullable=True)
    remarks = Column(String, nullable=True)
    status = Column(String, default="Pending")
    approver = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)

def create_request(session, user_name, application, version=None, remarks=None):
    import uuid
    rid = str(uuid.uuid4())[:8]
    req = Request(request_id=rid, user_name=user_name, application=application, version=version, remarks=remarks, status="Pending")
    session.add(req)
    session.commit()
    session.refresh(req)
    return req

def get_all_requests(session):
    return session.query(Request).order_by(Request.created_at.desc()).all()

def get_request_by_id(session, rid):
    return session.query(Request).filter_by(request_id=rid).first()

def update_request_status(session, req, status, approver=None):
    req.status = status
    if approver:
        req.approver = approver
    session.add(req)
    session.commit()
