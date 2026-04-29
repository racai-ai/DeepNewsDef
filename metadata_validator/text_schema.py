"""Pydantic models for text metadata validation."""

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


class TextSource(BaseModel):
    model_config = ConfigDict(extra="allow")

    article_url: str
    article_title: str
    article_date: str
    name: str
    category: str
    author: str
    language: str
    country: str
    from_dataset: str
    dataset_id: str


class TextGeneration(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    quantization: str
    temperature: float
    prompt: str
    generation_type: Literal["complete", "based_on_real_article", "based_on_topic"]
    paragraphs: List


class TextPerturbation(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: Literal[
        "numbers", "names", "dates", "locations",
        "quotes", "events", "sentiment", "paraphrasing",
    ]
    parameters: Optional[dict] = None
    paragraphs: Optional[List] = None


class TextAnnotations(BaseModel):
    model_config = ConfigDict(extra="allow")

    word_count: int
    char_count: int
    sentences: int
    file_size: int
    paragraph_count: int
    text_romania: Literal["yes", "no", ""]
    text_romania_type: Literal["provided", "automatic", "manual", ""]
    text_moldova: Literal["yes", "no", ""]
    text_moldova_type: Literal["provided", "automatic", "manual", ""]
    lang_ro: Literal["yes", "no", ""]
    lang_ro_type: Literal["provided", "automatic", "manual", ""]
    lang_ru: Literal["yes", "no", ""]
    lang_ru_type: Literal["provided", "automatic", "manual", ""]
    lang_other: Literal["yes", "no", ""]
    lang_other_type: Literal["provided", "automatic", "manual", ""]
    is_public_figure: Literal["yes", "no", ""]
    is_public_figure_type: Literal["provided", "automatic", "manual"]
    domain: str
    subdomain: str
    summary: str
    summary_type: Literal["provided", "automatic", "manual", ""]


class TextMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    # All fields required
    id: str
    real_fake: Literal["real", "fake"]
    based_on: str
    perturbations_applied: str
    year: int
    filename: str
    creation_date: str
    title: str
    summary: str

    # Nested (all required)
    source: TextSource
    generation: TextGeneration
    perturbations_list: List[TextPerturbation] = Field(
        alias="perturbations_data"
    )
    annotations: TextAnnotations

    # Warnings collected by cross-field validation
    warnings_: List[str] = []

    @model_validator(mode="after")
    def cross_field_checks(self) -> "TextMetadata":
        warnings = []

        # Check: fake text should have generation data
        if self.real_fake == "fake":
            if not self.generation.model:
                warnings.append(
                    "real_fake is 'fake' but generation.model is missing/empty."
                )

        # Check: perturbations_applied status includes 'perturbations' but no records
        perturbations_status = self.perturbations_applied or ""
        if "perturbations" in perturbations_status:
            if len(self.perturbations_list) == 0:
                warnings.append(
                    "perturbations status includes 'perturbations' but "
                    "no perturbation records provided."
                )

        self.warnings_ = warnings
        return self
