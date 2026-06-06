from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str
    object_labels: list[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    query: str
    expanded_queries: list[str]
    results: list[dict[str, object]]
