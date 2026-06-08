from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str
    object_labels: list[str] = Field(default_factory=list)


class QuestionSearchRequest(BaseModel):
    question: str


class SearchResponse(BaseModel):
    query: str
    expanded_queries: list[str]
    results: list[dict[str, object]]
    mode: str | None = None
    parsed_query: dict[str, object] | None = None
