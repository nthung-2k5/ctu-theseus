"""/api/integrations: the caller's saved credentials for outside services (Hugging Face)."""

from fastapi import APIRouter

from omnitrain.deps import SessionDep, UserId
from omnitrain.schemas.integrations import ConnectHuggingFaceBody, HuggingFaceStatus
from omnitrain.services import integrations

router = APIRouter(prefix="/integrations", tags=["integrations"])


@router.get("/huggingface", response_model=HuggingFaceStatus)
async def get_huggingface(user_id: UserId, session: SessionDep) -> HuggingFaceStatus:
    return await integrations.huggingface_status(session, user_id)


@router.put("/huggingface", response_model=HuggingFaceStatus)
async def connect_huggingface(body: ConnectHuggingFaceBody, user_id: UserId, session: SessionDep) -> HuggingFaceStatus:
    """Check the token with Hugging Face, then save it encrypted (replacing any earlier one)."""
    return await integrations.connect_huggingface(session, user_id, body.token)


@router.delete("/huggingface", status_code=204)
async def disconnect_huggingface(user_id: UserId, session: SessionDep) -> None:
    """Forget the saved token. A private model already imported works until its files must be fetched again."""
    await integrations.disconnect_huggingface(session, user_id)
