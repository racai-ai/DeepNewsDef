import json
from pydantic import BaseModel, Field
from typing import List

class GeneratedArticle(BaseModel):
    title: str = Field(description="The title or subject of the generated text")
    paragraphs: List[str] = Field(description="A list of paragraphs making up the body of the generated text")

class LLMInterface:
    def __init__(self, api_key: str, endpoint: str = None):
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=endpoint)
    
    def generate_article(self, model: str, system_prompt: str, user_prompt: str, temperature: float = 0.7, max_tokens: int = None, top_p: float = 1.0, frequency_penalty: float = 0.0, presence_penalty: float = 0.0, **kwargs) -> GeneratedArticle:
        try:
            response = self.client.beta.chat.completions.parse(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                response_format=GeneratedArticle,
                **kwargs
            )
            return response.choices[0].message.parsed
        except Exception as e:
            # Fallback to standard json mode if structured outputs not fully supported
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                response_format={"type": "json_object"},
                **kwargs
            )
            content = response.choices[0].message.content
            return GeneratedArticle.model_validate_json(content)
