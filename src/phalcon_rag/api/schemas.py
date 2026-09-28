from typing import Annotated

from pydantic import BaseModel, StringConstraints


class SourceResponse(BaseModel):
    id: str
    type: str
    file: str
    section: str | None = None
    method: str | None = None


class AskRequest(BaseModel):
    question: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=2000,
        ),
    ]


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
