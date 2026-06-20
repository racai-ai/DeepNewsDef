import uuid
import datetime
import json
from typing import Dict, Any, List
from pydantic import BaseModel, Field

from metadata_validator.text_schema import (
    TextMetadata, TextSource, TextGeneration, TextAnnotations, TextPerturbation
)
from synthetic_news_text_generator.llm_interface import GeneratedArticle, LLMInterface

class LLMExtraction(BaseModel):
    summary: str = Field(description="A 1-2 sentence summary of the text.")
    is_public_figure: str = Field(description="yes or no, if the text mentions a known public figure.")

class Pipeline:
    def __init__(self, llm_interface: LLMInterface, model_name: str = "gpt-4o"):
        self.llm = llm_interface
        self.model_name = model_name

    def extract_metrics(self, paragraphs: List[str]) -> Dict[str, int]:
        text_content = "\n".join(paragraphs)
        char_count = len(text_content)
        word_count = len(text_content.split())
        sentences = text_content.count('.') + text_content.count('!') + text_content.count('?')
        paragraph_count = len(paragraphs)
        file_size = len(text_content.encode('utf-8'))
        return {
            "char_count": char_count,
            "word_count": word_count,
            "sentences": sentences,
            "paragraph_count": paragraph_count,
            "file_size": file_size
        }

    def extract_llm_metadata(self, paragraphs: List[str]) -> LLMExtraction:
        system_prompt = "Analyze the provided text and extract metadata."
        user_prompt = "Text:\n" + "\n".join(paragraphs) + "\n\nExtract summary and whether it features a public figure."
        try:
            response = self.llm.client.beta.chat.completions.parse(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format=LLMExtraction
            )
            return response.choices[0].message.parsed
        except Exception:
            return LLMExtraction(summary="No summary available.", is_public_figure="no")
        
    def detect_language(self, paragraphs: List[str]) -> Dict[str, str]:
        # Keep it simple for the first step: assume only Romanian text is generated
        return {
            "lang_ro": "yes",
            "lang_ru": "no"
        }

    def process(self, generated_article: GeneratedArticle, context: Dict[str, Any]) -> TextMetadata:
        """
        Process the generated article and context building into TextMetadata
        context should contain:
            - country (Romania or Republic of Moldova)
            - generation_type (complete, based_on_real_article, based_on_topic)
            - prompt
            - model, quantization, temperature
            - topic, domain, subdomain (optional)
        """
        paragraphs = generated_article.paragraphs
        metrics = self.extract_metrics(paragraphs)
        lang_meta = self.detect_language(paragraphs)
        llm_extraction = self.extract_llm_metadata(paragraphs)
        
        summary = llm_extraction.summary
        is_public_figure = llm_extraction.is_public_figure.lower()
        if is_public_figure not in ["yes", "no"]:
            is_public_figure = "no"

        article_id = str(uuid.uuid4())
        timestamp = datetime.date.today()
        
        country = context.get('country', "Romania")
        text_romania = "yes" if "romania" in country.lower() else "no"
        text_moldova = "yes" if "moldova" in country.lower() else "no"

        source = TextSource(
            article_url="",
            article_title=generated_article.title,
            article_date=timestamp.isoformat(),
            name="synthetic_news_generator",
            category=context.get('topic', ""),
            author=context.get('persona', "synthetic"),
            language="ro",
            country=country,
            from_dataset="synthetic",
            dataset_id="V1"
        )

        generation = TextGeneration(
            model=context.get('model', 'gpt-4o'),
            quantization=context.get('quantization', 'none'),
            temperature=float(context.get('temperature', 0.7)),
            top_p=float(context.get('top_p', 1.0)),
            frequency_penalty=float(context.get('frequency_penalty', 0.0)),
            presence_penalty=float(context.get('presence_penalty', 0.0)),
            max_tokens=context.get('max_tokens', 2000),
            prompt=context.get('prompt', ''),
            generation_type=context.get('generation_type', 'complete'),
            paragraphs=paragraphs
        )

        annotations = TextAnnotations(
            word_count=metrics['word_count'],
            char_count=metrics['char_count'],
            sentences=metrics['sentences'],
            file_size=metrics['file_size'],
            paragraph_count=metrics['paragraph_count'],
            text_romania=text_romania,
            text_romania_type="automatic",
            text_moldova=text_moldova,
            text_moldova_type="automatic",
            lang_ro=lang_meta['lang_ro'],
            lang_ro_type="automatic",
            lang_ru=lang_meta['lang_ru'],
            lang_ru_type="automatic",
            lang_other="no",
            lang_other_type="automatic",
            is_public_figure=is_public_figure, # type: ignore
            is_public_figure_type="automatic",
            domain=context.get('domain', ''),
            subdomain=context.get('subdomain', ''),
            summary=summary,
            summary_type="automatic"
        )

        metadata_dict = {
            "id": article_id,
            "real_fake": "fake",
            "based_on": context.get('generation_type', 'complete'),
            "perturbations_applied": "",
            "year": timestamp.year,
            "filename": f"{article_id}.txt",
            "creation_date": timestamp.isoformat(),
            "title": generated_article.title,
            "summary": summary,
            "source": source.model_dump(),
            "generation": generation.model_dump(),
            "perturbations_data": [],
            "annotations": annotations.model_dump()
        }

        # Validate through Pydantic Schema
        validated_metadata = TextMetadata.model_validate(metadata_dict)
        return validated_metadata
