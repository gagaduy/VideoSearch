from pydantic import BaseModel


class JobRead(BaseModel):
    id: int
    status: str
    stage: str

    model_config = {"from_attributes": True}
