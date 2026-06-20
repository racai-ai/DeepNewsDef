import yaml
from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict, Union, Any, Optional

class LLMConfig(BaseModel):
    name: str
    display_name: Optional[str] = None
    endpoint: str
    temperature_range: List[float]
    top_p_range: List[float] = [1.0]
    frequency_penalty_range: List[float] = [0.0]
    presence_penalty_range: List[float] = [0.0]
    max_tokens: int = 2000
    quantization: str = "none"

class TopicConfig(BaseModel):
    domain: str
    subdomains: List[str]

class GeneratorConfig(BaseModel):
    llms: List[LLMConfig]
    target_countries: List[str]
    viewpoints: List[str]
    lengths: List[str]
    platforms: List[str]
    personas: List[str]
    topics: List[TopicConfig]
    grammar_levels: List[str]
    manipulation_levels: List[str]
    propaganda_techniques: List[str]
    target_audiences: List[str]
    urgency_levels: List[str]
    formatting_styles: List[str]
    output_directory: str
    use_wikipedia_seed: Union[bool, str] = True

def load_config(file_path: str = "config.yaml") -> GeneratorConfig:
    path = Path(file_path)
    if not path.is_absolute():
        path = Path(__file__).parent / file_path
    
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        
    return GeneratorConfig(**data)

if __name__ == "__main__":
    config = load_config()
    print("Config loaded successfully!")
    print(f"Loaded {len(config.llms)} LLMs and {len(config.topics)} topics.")
