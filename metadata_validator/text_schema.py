"""Pydantic models for text metadata validation."""

from datetime import date, datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator, ValidationInfo


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
    is_public_figure_type: Literal["provided", "automatic", "manual", ""]
    domain: str
    subdomain: str
    summary: str
    summary_type: Literal["provided", "automatic", "manual", ""]

    @field_validator("is_public_figure_type")
    @classmethod
    def validate_public_figure_type(cls, v: str, info: ValidationInfo) -> str:
        # is_public_figure_type can be empty only if is_public_figure is empty
        is_public_figure = info.data.get("is_public_figure", "")
        if v == "" and is_public_figure != "":
            raise ValueError("is_public_figure_type can only be empty if is_public_figure is empty")
        return v

    @field_validator("text_romania_type")
    @classmethod
    def validate_text_romania_type(cls, v: str, info: ValidationInfo) -> str:
        text_romania = info.data.get("text_romania", "")
        if v == "" and text_romania != "":
            raise ValueError("text_romania_type can only be empty if text_romania is empty")
        return v

    @field_validator("text_moldova_type")
    @classmethod
    def validate_text_moldova_type(cls, v: str, info: ValidationInfo) -> str:
        text_moldova = info.data.get("text_moldova", "")
        if v == "" and text_moldova != "":
            raise ValueError("text_moldova_type can only be empty if text_moldova is empty")
        return v

    @field_validator("lang_ro_type")
    @classmethod
    def validate_lang_ro_type(cls, v: str, info: ValidationInfo) -> str:
        lang_ro = info.data.get("lang_ro", "")
        if v == "" and lang_ro != "":
            raise ValueError("lang_ro_type can only be empty if lang_ro is empty")
        return v

    @field_validator("lang_ru_type")
    @classmethod
    def validate_lang_ru_type(cls, v: str, info: ValidationInfo) -> str:
        lang_ru = info.data.get("lang_ru", "")
        if v == "" and lang_ru != "":
            raise ValueError("lang_ru_type can only be empty if lang_ru is empty")
        return v

    @field_validator("lang_other_type")
    @classmethod
    def validate_lang_other_type(cls, v: str, info: ValidationInfo) -> str:
        lang_other = info.data.get("lang_other", "")
        if v == "" and lang_other != "":
            raise ValueError("lang_other_type can only be empty if lang_other is empty")
        return v

    @field_validator("summary_type")
    @classmethod
    def validate_summary_type(cls, v: str, info: ValidationInfo) -> str:
        summary = info.data.get("summary", "")
        if v == "" and summary != "":
            raise ValueError("summary_type can only be empty if summary is empty")
        return v


class TextMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    # All fields required
    id: str
    real_fake: Literal["real", "fake"]
    based_on: str
    perturbations_applied: str
    filename: str
    creation_date: datetime
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
