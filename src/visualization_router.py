from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlmodel import Session

router = APIRouter(prefix="/visualization", tags=["visualization"])


@router.get("/visualize/feature/{feature_name}")
async def visualize_feature_distribution(
    feature_name: str,
    session: Session = Depends(get_session)
):
