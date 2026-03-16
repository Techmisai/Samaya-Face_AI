import os
import uuid
import pickle
import shutil
import cv2
import logging
import asyncio
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import face_recognition
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    DB_PATH = Path('./db')
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}
    FACE_RECOGNITION_MODEL = "cnn"  # or "hog" for faster but less accurate
    SIMILARITY_THRESHOLD = 0.5
    MAX_USERS = 1000
    TEMP_DIR = Path('./temp')

# Pydantic models
class UserResponse(BaseModel):
    status: str
    message: str
    user_id: Optional[str] = None

class VerifyResponse(BaseModel):
    status: str
    user: Optional[str] = None
    match_percentage: Optional[float] = None
    message: Optional[str] = None

class UserInfo(BaseModel):
    user_name: str
    created_at: str
    image_path: str

# Custom exceptions
class FaceRecognitionError(Exception):
    pass

class NoFaceDetectedError(FaceRecognitionError):
    pass

class MultipleFacesError(FaceRecognitionError):
    pass

class DatabaseError(FaceRecognitionError):
    pass

# Database manager
class FaceDatabase:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.mkdir(exist_ok=True)
        self._ensure_db_integrity()
    
    def _ensure_db_integrity(self):
        """Ensure database integrity on startup"""
        pickle_files = list(self.db_path.glob('*.pickle'))
        image_files = list(self.db_path.glob('*.png'))
        
        # Remove orphaned files
        pickle_names = {f.stem for f in pickle_files}
        image_names = {f.stem for f in image_files}
        
        orphaned_pickles = pickle_names - image_names
        orphaned_images = image_names - pickle_names
        
        for orphan in orphaned_pickles:
            (self.db_path / f"{orphan}.pickle").unlink(missing_ok=True)
            logger.warning(f"Removed orphaned pickle file: {orphan}.pickle")
        
        for orphan in orphaned_images:
            (self.db_path / f"{orphan}.png").unlink(missing_ok=True)
            logger.warning(f"Removed orphaned image file: {orphan}.png")
    
    def user_exists(self, user_name: str) -> bool:
        pickle_path = self.db_path / f"{user_name}.pickle"
        return pickle_path.exists()
    
    def get_all_users(self) -> List[str]:
        return [f.stem for f in self.db_path.glob('*.pickle')]
    
    def save_user(self, user_name: str, embeddings: np.ndarray, image_data: bytes):
        if len(self.get_all_users()) >= Config.MAX_USERS:
            raise DatabaseError("Maximum number of users reached")
        
        pickle_path = self.db_path / f"{user_name}.pickle"
        image_path = self.db_path / f"{user_name}.png"
        
        try:
            # Save embeddings
            with open(pickle_path, 'wb') as f:
                pickle.dump(embeddings, f)
            
            # Save image
            with open(image_path, 'wb') as f:
                f.write(image_data)
                
            logger.info(f"User {user_name} saved successfully")
        except Exception as e:
            # Cleanup on failure
            pickle_path.unlink(missing_ok=True)
            image_path.unlink(missing_ok=True)
            raise DatabaseError(f"Failed to save user: {str(e)}")
    
    def delete_user(self, user_name: str):
        pickle_path = self.db_path / f"{user_name}.pickle"
        image_path = self.db_path / f"{user_name}.png"
        
        if not pickle_path.exists():
            raise HTTPException(status_code=404, detail="User not found")
        
        pickle_path.unlink(missing_ok=True)
        image_path.unlink(missing_ok=True)
        logger.info(f"User {user_name} deleted successfully")
    
    def load_user_embeddings(self, user_name: str) -> np.ndarray:
        pickle_path = self.db_path / f"{user_name}.pickle"
        try:
            with open(pickle_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            raise DatabaseError(f"Failed to load embeddings for {user_name}: {str(e)}")

# Face recognition service
class FaceRecognitionService:
    def __init__(self, db: FaceDatabase):
        self.db = db
    
    def extract_face_embeddings(self, image: np.ndarray, enforce_single_face: bool = True) -> np.ndarray:
        """Extract face embeddings from image"""
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
            
            embeddings = face_recognition.face_encodings(image_rgb, model=Config.FACE_RECOGNITION_MODEL)
            
            if len(embeddings) == 0:
                raise NoFaceDetectedError("No face detected in the image")
            
            if enforce_single_face and len(embeddings) > 1:
                raise MultipleFacesError("Multiple faces detected. Please use an image with a single face")
            
            return embeddings[0]
            
        except Exception as e:
            if isinstance(e, (NoFaceDetectedError, MultipleFacesError)):
                raise
            raise FaceRecognitionError(f"Error processing image: {str(e)}")
    
    def recognize_face(self, image: np.ndarray) -> Tuple[str, bool, float]:
        """Recognize face in image"""
        try:
            unknown_embedding = self.extract_face_embeddings(image, enforce_single_face=False)
        except NoFaceDetectedError:
            return 'no_persons_found', False, 0.0
        except Exception as e:
            logger.error(f"Error in face recognition: {str(e)}")
            return 'error', False, 0.0
        
        users = self.db.get_all_users()
        if not users:
            return 'unknown_person', False, 0.0
        
        best_match_distance = float('inf')
        best_match_name = 'unknown_person'
        
        for user_name in users:
            try:
                embeddings = self.db.load_user_embeddings(user_name)
                # Handle both single embedding and list of embeddings
                if isinstance(embeddings, list):
                    embedding = embeddings[0]
                else:
                    embedding = embeddings
                
                distance = face_recognition.face_distance([embedding], unknown_embedding)[0]
                
                if distance < best_match_distance:
                    best_match_distance = distance
                    best_match_name = user_name
                    
            except Exception as e:
                logger.error(f"Error processing user {user_name}: {str(e)}")
                continue
        
        match_percentage = max(0, (1 - best_match_distance) * 100)
        is_match = best_match_distance < Config.SIMILARITY_THRESHOLD
        
        if is_match:
            return best_match_name, True, match_percentage
        else:
            return 'unknown_person', False, match_percentage

# File validation utilities
class FileValidator:
    @staticmethod
    def validate_image_file(file: UploadFile) -> None:
        # Check file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in Config.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file format. Allowed: {', '.join(Config.ALLOWED_EXTENSIONS)}"
            )
        
        # Check file size (this is approximate since we haven't read the file yet)
        if hasattr(file, 'size') and file.size > Config.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {Config.MAX_FILE_SIZE // (1024*1024)}MB"
            )
    
    @staticmethod
    def validate_user_name(user_name: str) -> None:
        if not user_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User name is required"
            )
        
        if len(user_name) < 2 or len(user_name) > 50:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User name must be between 2 and 50 characters"
            )
        
        # Check for invalid characters
        if not user_name.replace('_', '').replace('-', '').replace(' ', '').isalnum():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User name can only contain letters, numbers, spaces, hyphens, and underscores"
            )

# Temporary file manager
class TempFileManager:
    def __init__(self):
        Config.TEMP_DIR.mkdir(exist_ok=True)
    
    async def save_temp_file(self, file: UploadFile) -> Path:
        """Save uploaded file temporarily and return path"""
        temp_filename = f"{uuid.uuid4()}{Path(file.filename).suffix}"
        temp_path = Config.TEMP_DIR / temp_filename
        
        try:
            contents = await file.read()
            if len(contents) > Config.MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"File too large. Maximum size: {Config.MAX_FILE_SIZE // (1024*1024)}MB"
                )
            
            with open(temp_path, "wb") as f:
                f.write(contents)
            
            return temp_path
        except Exception as e:
            temp_path.unlink(missing_ok=True)
            raise
    
    def cleanup_temp_file(self, temp_path: Path):
        """Clean up temporary file"""
        temp_path.unlink(missing_ok=True)

# Initialize components
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting Face Recognition API...")
    
    # Cleanup temp directory
    if Config.TEMP_DIR.exists():
        shutil.rmtree(Config.TEMP_DIR)
    Config.TEMP_DIR.mkdir(exist_ok=True)
    
    yield
    
    # Shutdown
    logger.info("Shutting down Face Recognition API...")
    # Cleanup temp files
    if Config.TEMP_DIR.exists():
        shutil.rmtree(Config.TEMP_DIR)

# FastAPI app setup
app = FastAPI(
    title="Face Recognition System",
    description="Enhanced API for registering, verifying, deleting, and listing users using face recognition.",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
db = FaceDatabase(Config.DB_PATH)
face_service = FaceRecognitionService(db)
temp_manager = TempFileManager()
file_validator = FileValidator()

# API Routes
@app.post("/register_new_user", response_model=UserResponse)
async def register_new_user(file: UploadFile = File(...), user_name: str = Form(...)):
    """Register a new user with face recognition"""
    try:
        # Validate inputs
        file_validator.validate_image_file(file)
        file_validator.validate_user_name(user_name)
        
        # Check if user already exists
        if db.user_exists(user_name):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="User already exists"
            )
        
        # Save temporary file
        temp_path = await temp_manager.save_temp_file(file)
        
        try:
            # Read and process image
            image = cv2.imread(str(temp_path))
            if image is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid image file"
                )
            
            # Extract face embeddings
            embeddings = face_service.extract_face_embeddings(image)
            
            # Read file contents for storage
            with open(temp_path, 'rb') as f:
                image_data = f.read()
            
            # Save user to database
            db.save_user(user_name, embeddings, image_data)
            
            return UserResponse(
                status="success",
                message="User registered successfully",
                user_id=user_name
            )
            
        finally:
            temp_manager.cleanup_temp_file(temp_path)
            
    except NoFaceDetectedError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No face detected in the image"
        )
    except MultipleFacesError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Multiple faces detected. Please use an image with a single face"
        )
    except DatabaseError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error in register_new_user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.post("/verify_user", response_model=VerifyResponse)
async def verify_user(file: UploadFile = File(...)):
    """Verify a user using face recognition"""
    try:
        # Validate file
        file_validator.validate_image_file(file)
        
        # Save temporary file
        temp_path = await temp_manager.save_temp_file(file)
        
        try:
            # Read and process image
            image = cv2.imread(str(temp_path))
            if image is None:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid image file"
                )
            
            # Recognize face
            user_name, match_status, match_percentage = face_service.recognize_face(image)
            
            if match_status:
                return VerifyResponse(
                    status="success",
                    user=user_name,
                    match_percentage=round(match_percentage, 2)
                )
            else:
                return VerifyResponse(
                    status="error",
                    message="No matching face found",
                    match_percentage=round(match_percentage, 2)
                )
                
        finally:
            temp_manager.cleanup_temp_file(temp_path)
            
    except Exception as e:
        logger.error(f"Error in verify_user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.get("/users", response_model=List[str])
def get_all_users():
    """Get list of all registered users"""
    try:
        return db.get_all_users()
    except Exception as e:
        logger.error(f"Error in get_all_users: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.delete("/users/{user_name}", response_model=UserResponse)
def delete_user(user_name: str):
    """Delete a registered user"""
    try:
        file_validator.validate_user_name(user_name)
        db.delete_user(user_name)
        
        return UserResponse(
            status="success",
            message=f"User '{user_name}' deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in delete_user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "total_users": len(db.get_all_users()),
        "version": "2.0.0"
    }

@app.get("/config")
def get_config():
    """Get current configuration"""
    return {
        "max_file_size_mb": Config.MAX_FILE_SIZE // (1024 * 1024),
        "allowed_extensions": list(Config.ALLOWED_EXTENSIONS),
        "similarity_threshold": Config.SIMILARITY_THRESHOLD,
        "max_users": Config.MAX_USERS,
        "model": Config.FACE_RECOGNITION_MODEL
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.detail}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )