import os
import uuid
import pickle
import shutil
import cv2
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import face_recognition
import numpy as np

logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger=logging.getLogger(__name__)

class Config:
    DB_PATH=Path('./db')
    MAX_FILE_SIZE=10*1024*1024
    ALLOWED_EXTENSIONS={'.jpg','.jpeg','.png','.bmp'}
    FACE_RECOGNITION_MODEL="hog"
    SIMILARITY_THRESHOLD=0.5
    MAX_USERS=1000
    TEMP_DIR=Path('./temp')

class UserResponse(BaseModel):
    status:str
    message:str
    user_id:Optional[str]=None

class VerifyResponse(BaseModel):
    status:str
    user:Optional[str]=None
    match_percentage:Optional[float]=None
    message:Optional[str]=None

class UserInfo(BaseModel):
    user_name:str
    created_at:str
    image_path:str

class FaceRecognitionError(Exception):
    pass

class NoFaceDetectedError(FaceRecognitionError):
    pass

class MultipleFacesError(FaceRecognitionError):
    pass

class DatabaseError(FaceRecognitionError):
    pass

class FaceDatabase:
    def __init__(self,db_path:Path):
        self.db_path=db_path
        self.db_path.mkdir(exist_ok=True)
        self._ensure_db_integrity()

    def _ensure_db_integrity(self):
        pickle_files=list(self.db_path.glob('*.pickle'))
        image_files=list(self.db_path.glob('*.png'))

        pickle_names={f.stem for f in pickle_files}
        image_names={f.stem for f in image_files}

        orphaned_pickles=pickle_names-image_names
        orphaned_images=image_names-pickle_names

        for orphan in orphaned_pickles:
            (self.db_path/f"{orphan}.pickle").unlink(missing_ok=True)

        for orphan in orphaned_images:
            (self.db_path/f"{orphan}.png").unlink(missing_ok=True)

    def user_exists(self,user_name:str)->bool:
        return (self.db_path/f"{user_name}.pickle").exists()

    def get_all_users(self)->List[str]:
        return [f.stem for f in self.db_path.glob('*.pickle')]

    def save_user(self,user_name:str,embeddings:np.ndarray,image_data:bytes):
        if len(self.get_all_users())>=Config.MAX_USERS:
            raise DatabaseError("Maximum users reached")

        pickle_path=self.db_path/f"{user_name}.pickle"
        image_path=self.db_path/f"{user_name}.png"

        with open(pickle_path,'wb') as f:
            pickle.dump(embeddings,f)

        with open(image_path,'wb') as f:
            f.write(image_data)

    def delete_user(self,user_name:str):
        pickle_path=self.db_path/f"{user_name}.pickle"
        image_path=self.db_path/f"{user_name}.png"

        if not pickle_path.exists():
            raise HTTPException(status_code=404,detail="User not found")

        pickle_path.unlink(missing_ok=True)
        image_path.unlink(missing_ok=True)

    def load_user_embeddings(self,user_name:str)->np.ndarray:
        with open(self.db_path/f"{user_name}.pickle",'rb') as f:
            return pickle.load(f)

class FaceRecognitionService:
    def __init__(self,db:FaceDatabase):
        self.db=db
        self.user_names=[]
        self.embeddings_matrix=[]

        users=self.db.get_all_users()

        for user_name in users:
            try:
                emb=self.db.load_user_embeddings(user_name)

                if isinstance(emb,list):
                    emb=emb[0]

                self.user_names.append(user_name)
                self.embeddings_matrix.append(emb)

            except Exception as e:
                logger.error(f"Embedding load failed for {user_name}")

        if len(self.embeddings_matrix)>0:
            self.embeddings_matrix=np.vstack(self.embeddings_matrix)

        logger.info(f"Loaded {len(self.user_names)} embeddings")

    def extract_face_embeddings(self,image:np.ndarray,enforce_single_face=True)->np.ndarray:

        if len(image.shape)==3 and image.shape[2]==3:
            image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

        image=cv2.resize(image,(0,0),fx=0.5,fy=0.5)

        face_locations=face_recognition.face_locations(
            image,
            model=Config.FACE_RECOGNITION_MODEL
        )

        embeddings=face_recognition.face_encodings(
            image,
            face_locations,
            model="small"
        )

        if len(embeddings)==0:
            raise NoFaceDetectedError()

        if enforce_single_face and len(embeddings)>1:
            raise MultipleFacesError()

        return embeddings[0]

    def recognize_face(self,image:np.ndarray)->Tuple[str,bool,float]:

        try:
            unknown_embedding=self.extract_face_embeddings(image,False)
        except NoFaceDetectedError:
            return 'no_persons_found',False,0.0

        if len(self.user_names)==0:
            return 'unknown_person',False,0.0

        distances=face_recognition.face_distance(self.embeddings_matrix,unknown_embedding)

        best_index=np.argmin(distances)
        best_distance=distances[best_index]

        match_percentage=max(0,(1-best_distance)*100)

        if best_distance<Config.SIMILARITY_THRESHOLD:
            return self.user_names[best_index],True,match_percentage

        return 'unknown_person',False,match_percentage

class FileValidator:
    @staticmethod
    def validate_image_file(file:UploadFile):

        ext=Path(file.filename).suffix.lower()

        if ext not in Config.ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400,detail="Invalid file format")

class TempFileManager:
    def __init__(self):
        Config.TEMP_DIR.mkdir(exist_ok=True)

    async def save_temp_file(self,file:UploadFile)->Path:

        temp_filename=f"{uuid.uuid4()}{Path(file.filename).suffix}"
        temp_path=Config.TEMP_DIR/temp_filename

        contents=await file.read()

        if len(contents)>Config.MAX_FILE_SIZE:
            raise HTTPException(status_code=413,detail="File too large")

        with open(temp_path,"wb") as f:
            f.write(contents)

        return temp_path

    def cleanup_temp_file(self,temp_path:Path):
        temp_path.unlink(missing_ok=True)

@asynccontextmanager
async def lifespan(app:FastAPI):

    logger.info("Starting Face Recognition API")

    if Config.TEMP_DIR.exists():
        shutil.rmtree(Config.TEMP_DIR)

    Config.TEMP_DIR.mkdir(exist_ok=True)

    yield

    if Config.TEMP_DIR.exists():
        shutil.rmtree(Config.TEMP_DIR)

app=FastAPI(
    title="Face Recognition System",
    description="Face recognition API",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db=FaceDatabase(Config.DB_PATH)
face_service=FaceRecognitionService(db)
temp_manager=TempFileManager()
file_validator=FileValidator()

@app.post("/register_new_user",response_model=UserResponse)
async def register_new_user(file:UploadFile=File(...),user_name:str=Form(...)):

    file_validator.validate_image_file(file)

    if db.user_exists(user_name):
        raise HTTPException(status_code=409,detail="User exists")

    temp_path=await temp_manager.save_temp_file(file)

    try:

        image=cv2.imread(str(temp_path))

        if image is None:
            raise HTTPException(status_code=400,detail="Invalid image")

        embeddings=face_service.extract_face_embeddings(image)

        with open(temp_path,'rb') as f:
            image_data=f.read()

        db.save_user(user_name,embeddings,image_data)

        return UserResponse(
            status="success",
            message="User registered successfully",
            user_id=user_name
        )

    finally:
        temp_manager.cleanup_temp_file(temp_path)

@app.post("/verify_user",response_model=VerifyResponse)
async def verify_user(file:UploadFile=File(...)):

    file_validator.validate_image_file(file)

    temp_path=await temp_manager.save_temp_file(file)

    try:

        image=cv2.imread(str(temp_path))

        if image is None:
            raise HTTPException(status_code=400,detail="Invalid image")

        user_name,match_status,match_percentage=face_service.recognize_face(image)

        if match_status:

            return VerifyResponse(
                status="success",
                user=user_name,
                match_percentage=round(match_percentage,2)
            )

        return VerifyResponse(
            status="error",
            message="No matching face found",
            match_percentage=round(match_percentage,2)
        )

    finally:
        temp_manager.cleanup_temp_file(temp_path)

@app.get("/users",response_model=List[str])
def get_all_users():
    return db.get_all_users()

@app.delete("/users/{user_name}",response_model=UserResponse)
def delete_user(user_name:str):

    db.delete_user(user_name)

    return UserResponse(
        status="success",
        message=f"{user_name} deleted"
    )

@app.get("/health")
def health_check():
    return {
        "status":"healthy",
        "total_users":len(db.get_all_users()),
        "version":"2.0.0"
    }

@app.get("/config")
def get_config():
    return {
        "max_file_size_mb":Config.MAX_FILE_SIZE//(1024*1024),
        "allowed_extensions":list(Config.ALLOWED_EXTENSIONS),
        "similarity_threshold":Config.SIMILARITY_THRESHOLD,
        "max_users":Config.MAX_USERS,
        "model":Config.FACE_RECOGNITION_MODEL
    }

@app.exception_handler(HTTPException)
async def http_exception_handler(request,exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"status":"error","message":exc.detail}
    )

if __name__=="__main__":
    import uvicorn
    uvicorn.run(app,host="0.0.0.0",port=8000,log_level="info")