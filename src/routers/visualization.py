from fastapi import APIRouter, Depends
from sqlmodel import Session

from src.db.session import get_session

router = APIRouter(prefix="/visualization", tags=["visualization"])


@router.get("/visualize/feature/{feature_name}")
async def visualize_feature_distribution(
    feature_name: str,
    session: Session = Depends(get_session)
):
    pass
