# For API operations and standards
from fastapi import APIRouter, UploadFile, Response, status, HTTPException
# Our detector objects
from detectors import mediapipe
# For encoding images
import cv2
# For response schemas
from schemas.mediapipe import ImageAnalysisResponse

# A new router object that we can add endpoints to.
# Note that the prefix is /yolo, so all endpoints from
# here on will be relative to /yolo
router = APIRouter(tags=["Image Upload and analysis"], prefix="/api")

# A cache of annotated images. Note that this would typically
# be some sort of persistent storage (think maybe postgres + S3)
# but for simplicity, we can keep things in memory
images = []

@router.post("/retrieve_rgb_season",
    status_code=status.HTTP_201_CREATED,
    responses={
        201: {"description": "Successfully Analyzed Image."}
    },
    response_model=ImageAnalysisResponse,
)


async def mediapipe_image_analysis(image_data: dict) -> ImageAnalysisResponse:

    try:
        image_url = image_data.get("image_url")
        if not image_url:
            raise HTTPException(status_code=400, detail="Missing image_url field in request body")
        dt = mediapipe.ImageAnalysis(image_url=image_url)
        output = await dt()
        response = ImageAnalysisResponse(
            categorize_info=output["categorize_info"],
            color=output["color"],
            lh=output["lh"],
            season=output["season"]
        )
        return response
    except IndexError:
        raise HTTPException(status_code=404, detail="Image cannot be analyzed") 
    
