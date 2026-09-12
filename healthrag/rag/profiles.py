"""Optional domain configuration; no changes to the retrieval engine are needed."""
from importlib.resources import files
from pydantic import BaseModel, ConfigDict, Field


class ResearchProfile(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str = Field(pattern=r'^[a-z][a-z0-9_-]{0,63}$')
    description: str = Field(max_length=300)
    retrieval_context: str = Field(default='', max_length=300)
    answer_guidance: str = Field(default='', max_length=1000)


def load_profile(name='general'):
    if name not in {'general', 'braf_oncology'}:
        raise ValueError(f'Unknown profile: {name}')
    return ResearchProfile.model_validate_json(
        files('healthrag.rag').joinpath('profiles', f'{name}.json').read_text()
    )
