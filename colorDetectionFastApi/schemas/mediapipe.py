from pydantic import BaseModel
from typing import List, Tuple, Dict, Union

class ImageAnalysisResponse(BaseModel):
    categorize_info: Dict[str, Union[float, str, List[float]]]
    color: Dict[str, Dict[str, Union[str, List[float]]]]
    lh: Dict[str, List[float]]
    season: str