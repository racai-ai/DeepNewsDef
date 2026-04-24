"""Pydantic models for image metadata validation."""

import os
from typing import List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, model_validator


class ImageSource(BaseModel):
    model_config = ConfigDict(extra="allow")

    page_url: str
    page_date: str
    page_title: str
    page_content: str
    source_name: str
    page_category: str
    from_dataset: str
    dataset_id: str


class ImageGeneration(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    quantization: str
    prompt: str
    neg_prompt: str
    category: str
    steps: str
    guidance_scale: str
    type: str
    additional_image: str


class ImageFilter(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str
    x: Optional[int] = None
    y: Optional[int] = None
    w: Optional[int] = None
    h: Optional[int] = None
    size_x: Optional[int] = None
    size_y: Optional[int] = None
    radius: Optional[float] = None
    amount: Optional[float] = None
    threshold: Optional[float] = None


class ImageAnnotations(BaseModel):
    model_config = ConfigDict(extra="allow")

    no_of_persons: Union[int, str]
    no_of_persons_type: Literal["provided", "automatic", "manual"]
    is_public_figure: Literal["yes", "no", ""]
    is_public_figure_type: Literal["provided", "automatic", "manual"]
    text_content_language: str
    text_content_language_type: Literal["provided", "automatic", "manual"]
    text_romania: Literal["yes", "no", ""]
    text_romania_type: Literal["provided", "automatic", "manual", ""]
    text_moldova: Literal["yes", "no", ""]
    text_moldova_type: Literal["provided", "automatic", "manual", ""]
    image_romania: Literal["yes", "no", ""]
    image_romania_type: Literal["provided", "automatic", "manual"]
    image_moldova: Literal["yes", "no", ""]
    image_moldova_type: Literal["provided", "automatic", "manual"]
    title: str
    title_type: Literal["provided", "automatic", "manual"]
    description: str
    description_type: Literal["provided", "automatic", "manual"]


class ImageMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    # All fields required
    id: str
    real_fake: Literal["real", "fake"]
    based_on: str
    manipulations: str
    filename: str
    extension: str
    mime_type: str
    creation_date: str
    url: str
    height: int
    width: int
    channels: int
    bpp: int
    color: Literal["color", "grayscale"]
    compression_level: int
    filesize: int

    # Nested (all required)
    source: ImageSource
    generation: ImageGeneration
    filters: List[ImageFilter]
    annotations: ImageAnnotations

    # Warnings collected by cross-field validation
    warnings_: List[str] = []

    @model_validator(mode="after")
    def cross_field_checks(self) -> "ImageMetadata":
        warnings = []
        manipulations = self.manipulations or ""

        # Check: fake + generation manipulation but no generation data
        if self.real_fake == "fake" and "generation" in manipulations:
            if not self.generation.model:
                warnings.append(
                    "real_fake is 'fake' with 'generation' manipulation, "
                    "but generation data or generation.model is missing/empty."
                )

        # Check: fake + filters manipulation but no filters data
        if self.real_fake == "fake" and "filters" in manipulations:
            if len(self.filters) == 0:
                warnings.append(
                    "real_fake is 'fake' with 'filters' manipulation, "
                    "but filters list is missing or empty."
                )

        # Check: id should match filename without extension
        if self.id and self.filename:
            name_without_ext = os.path.splitext(self.filename)[0]
            if self.id != name_without_ext:
                warnings.append(
                    f"id '{self.id}' does not match filename without extension "
                    f"'{name_without_ext}'."
                )

        self.warnings_ = warnings
        return self
