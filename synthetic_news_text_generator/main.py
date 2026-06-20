import os
import argparse
import random
import yaml
import json
from synthetic_news_text_generator.config_parser import load_config
from synthetic_news_text_generator.wiki_scraper import fetch_random_wikipedia_extract
from synthetic_news_text_generator.prompt_engine import PromptEngine
from synthetic_news_text_generator.llm_interface import LLMInterface
from synthetic_news_text_generator.pipeline import Pipeline

def main():
    parser = argparse.ArgumentParser(description="Synthetic News Text Generator")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config.yaml")
    parser.add_argument("--count", type=int, default=1, help="Number of articles to generate")
    args = parser.parse_args()

    config = load_config(args.config)
    api_key = os.getenv("OPENAI_API_KEY", "ollama_dummy_key")
    llm_interface = LLMInterface(api_key=api_key)
    pipeline = Pipeline(llm_interface)

    output_dir = config.output_directory
    os.makedirs(output_dir, exist_ok=True)

    for i in range(args.count):
        print(f"Generating article {i+1}/{args.count}...")

        # Select random parameters
        country = random.choice(config.target_countries)
        persona = random.choice(config.personas)
        length = random.choice(config.lengths)
        platform = random.choice(config.platforms)
        viewpoint = random.choice(config.viewpoints)
        grammar = random.choice(config.grammar_levels) if hasattr(config, 'grammar_levels') and config.grammar_levels else "impecabilă, fără greșeli"
        manipulation = random.choice(config.manipulation_levels) if hasattr(config, 'manipulation_levels') and config.manipulation_levels else "lipsită de manipulare, pur informativă și obiectivă"
        propaganda = random.choice(config.propaganda_techniques) if hasattr(config, 'propaganda_techniques') and config.propaganda_techniques else "Niciuna"
        audience = random.choice(config.target_audiences) if hasattr(config, 'target_audiences') and config.target_audiences else "Public general"
        urgency = random.choice(config.urgency_levels) if hasattr(config, 'urgency_levels') and config.urgency_levels else "Știre de rutină"
        formatting = random.choice(config.formatting_styles) if hasattr(config, 'formatting_styles') and config.formatting_styles else "Standard"
        
        topic_info = random.choice(config.topics)
        domain = topic_info.domain
        subdomain = random.choice(topic_info.subdomains)
        topic_str = f"{domain} - {subdomain}"

        llm_config = random.choice(config.llms)
        model_name = llm_config.name
        model_display_name = getattr(llm_config, "display_name", None) or model_name
        # Override the endpoint for this LLM config
        if llm_config.endpoint and llm_config.endpoint != "api":
            llm_interface = LLMInterface(api_key=api_key, endpoint=llm_config.endpoint)
            pipeline = Pipeline(llm_interface)

        temperature = random.choice(llm_config.temperature_range)
        top_p = random.choice(getattr(llm_config, "top_p_range", [1.0]) or [1.0])
        frequency_penalty = random.choice(getattr(llm_config, "frequency_penalty_range", [0.0]) or [0.0])
        presence_penalty = random.choice(getattr(llm_config, "presence_penalty_range", [0.0]) or [0.0])
        max_tokens = getattr(llm_config, "max_tokens", 2000)

        current_generation_types = ["complete", "based_on_topic"]
        use_seed = getattr(config, 'use_wikipedia_seed', True)
        if use_seed == "random":
            use_seed = random.choice([True, False])
        if use_seed:
            current_generation_types.append("based_on_real_article")
            
        generation_type = random.choice(current_generation_types)
        seed = ""
        seed_url = ""
        if generation_type == "based_on_real_article":
            try:
                seed, seed_url = fetch_random_wikipedia_extract()
                if not seed:
                    raise Exception("Empty extract returned")
            except Exception as e:
                print(f"Warning: Fetching wiki failed: {e}. Falling back to 'complete'.")
                generation_type = "complete"

        # Build prompt
        if generation_type == "complete":
            prompt = PromptEngine.build_complete_prompt(persona, country, length, platform, topic_str, viewpoint, grammar, manipulation, propaganda, audience, urgency, formatting)
        elif generation_type == "based_on_real_article":
            prompt = PromptEngine.build_based_on_real_article_prompt(persona, country, length, platform, viewpoint, seed, grammar, manipulation, propaganda, audience, urgency, formatting)
        elif generation_type == "based_on_topic":
            prompt = PromptEngine.build_based_on_topic_prompt(persona, country, length, topic_str, viewpoint, grammar, manipulation, propaganda, audience, urgency, formatting)
        else:
            prompt = PromptEngine.build_complete_prompt(persona, country, length, platform, topic_str, viewpoint, grammar, manipulation, propaganda, audience, urgency, formatting)
            
        system_prompt = PromptEngine.get_system_prompt()

        try:
            # Generate article
            article = llm_interface.generate_article(
                model=model_name,
                system_prompt=system_prompt,
                user_prompt=prompt, 
                temperature=temperature,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty
            )
            
            # Post-process and metadata extraction
            context = {
                "country": country,
                "generation_type": generation_type,
                "prompt": prompt,
                "model": model_display_name,
                "quantization": getattr(llm_config, "quantization", "none"),
                "temperature": temperature,
                "top_p": top_p,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "max_tokens": max_tokens,
                "topic": topic_str,
                "persona": persona
            }
            
            metadata = pipeline.process(article, context)

            # Modify metadata before saving
            metadata.generation.paragraphs = []
            
            # Save to file
            output_file_json = os.path.join(output_dir, f"{metadata.id}.json")
            with open(output_file_json, 'w', encoding='utf-8') as f:
                f.write(metadata.model_dump_json(by_alias=True, indent=4))
                
            output_file_txt = os.path.join(output_dir, f"{metadata.id}.txt")
            with open(output_file_txt, 'w', encoding='utf-8') as f:
                f.write(article.title + "\n\n" + "\n\n".join(article.paragraphs))
            
            print(f"Saved generated article to {output_file_txt} and {output_file_json}")

        except Exception as e:
            print(f"Error generating article: {e}")

if __name__ == "__main__":
    main()
