# Specification: Romanian Synthetic News Text Generator

## 1. Overview
The Synthetic News Text Generator is a highly modular tool designed to generate diverse, synthetic Romanian and Moldovan language news articles. It uses a combination of seed content (Wikipedia), a pool of bespoke author personas, target geographies, and topical themes to prompt Large Language Models (LLMs). The tool aims to produce robust datasets for **deep generated text detection models**, requiring strict adherence to the `TextMetadata` schema and varied text artifacts.

## 2. Core Components

### 2.1 Wikipedia Seed Module
Provides factual seeds solely to increase output diversity. It programmatically extracts random parts of the Romanian Wikipedia (`ro.wikipedia.org`).
*   **Usage**: Enables the `based_on_real_article` generation type, acting as thematic inspiration.

### 2.2 Persona Pool
A structured dataset of "writing styles" customized for diverse text generation to train detectors.
*   **Standard Personas**: Tabloid journalism, formal investigative, corporate PR, casual opinion blogger.
*   **Detector-Specific Personas**: "Translation Bot" (Simulates low-quality automated translation from English into Romanian, a common vector for bulk-generated content farming).
*   **Social & Platform Personas**: "Social Media Influencer" (emojis, direct address), "Reddit User" (slang, structured formatting, cynical tone), "Telegram Admin" (bullet points, urgent tone, propaganda-adjacent), "TikTok Creator" (high-energy hook, short sentences, captions format).

### 2.3 Geography Target (Romania vs Moldova)
Explicit injection of the target country to ensure generated articles represent local context and dialectic nuances (e.g., regionalisms for Moldova).
*   **Targets**: "Romania" or "Republic of Moldova".
*   **Usage**: Modifies the persona context to target a distinct audience, mapping to `TextSource.country` and `TextAnnotations` flags.

### 2.4 Viewpoint Modulation
A module to enforce positive, neutral, or negative sentiment to increase the psychological range of the generated data.
*   **Attributes**: "Positive", "Neutral", "Negative".
Platform & Format Simulation
To ensure coverage over the broad ways synthetic text is distributed today, the generator simulates distinct publishing contexts, directly mapping to the schemas:
*   **Formats**: `Standard News Portal`, `Telegram Broadcast`, `Reddit Thread/Post`, `TikTok Text/Script`, `Facebook/Instagram Post`.
*   **Usage**: Appended dynamically to the prompt to force the LLM to output text using platform-specific conventions (e.g., bullet points for Telegram, hashtags/emojis for Instagram, colloquial hooks for Reddit).

### 2.6 Article Length Control (Crucial for Detection)
Detectors often perform differently based on text length. Ensuring high variance here is mandatory.
*   **Lengths**: `Short` (~100-150 words, breaking news format or social post), `Medium` (~350 words, standard reporting/Reddit post), `Long` (600+ words, editorial/investigative).

### 2.7 LLM Engine & Parameters
Instructs backend LLMs using the `response_format` API parameter (Strict JSON schema) to guarantee parsable objects containing a `title` and string arrays of `paragraphs`.
*   **Supported Models**: Llama-3-8B-Romanian (via vLLM/Ollama), GPT-4o, etc. Platform Format, Length) are combined. Output is forced via `response_format` JSON.

1.  **`complete` (Pure Synthetic)**
    *   **Prompt**: "Act as a [Persona] based in [Target Country]. Write a [Length] [Platform Format] about [Topic] maintaining a strictly [Viewpoint] perspective. Return the response strictly as JSON with a `title` (or subject) and `paragraphs`."
2.  **`based_on_real_article` (Thematic Inspiration)**
    *   **Prompt**: "Using the following Wikipedia extract [Seed] as a factual seed, write a brand new, [Length] [Platform Format] in the style of a [Persona] from [Target Country]. The overall narrative must be strongly [Viewpoint]. Output purely as structured JSON."
3.  **`based_on_topic` (Thematic Fiction)**
    *   **Prompt**: "Write a [Length] [Platform Format]
    *   **Prompt**: "Act as a [Persona] journalist based in [Target Country]. Write a [Length] original article about [Topic] maintaining a strictly [Viewpoint] perspective. Return the response strictly as JSON with a `title` and `paragraphs`."
2.  **`based_on_real_article` (Thematic Inspiration)**
    *   **Prompt**: "Using the following Wikipedia extract [Seed] as a factual seed, write a brand new, [Length] news article in the style of a [Persona] from [Target Country]. The overall narrative must be strongly [Viewpoint]. Output purely as structured JSON."
3.  **`based_on_topic` (Thematic Fiction)**
    * Diacritic / Encoding Module**: Applies the transformations defined in Section 2.7.
2.  **Counters & Checks**:
## 4. Post-Processing and Annotation Pipeline

Once the LLM returns the structured JSON, it passes through the pipeline:
1.  **Metadata Extraction**: Extracts `word_count`, `char_count`, `sentences`, `paragraph_count`, `file_size`.
2.  **Language Metadata**: Automatic detection for `lang_ro`, `lang_ru`.
3.  **Summarization/NER**: Fast LLM-based extraction for `summary` and `is_public_figure`.

## 5. Schema Compliance & Validation

All generated outputs are converted into JSON matching the Pydantic `TextMetadata` schema. 
*   **`real_fake`**: Always `"fake"`.
*   **Validation**: Every dictionary runs through `TextMetadata.model_validate(data)` before saving.

## 6. Configuration Example (config.yaml)

```yaml
llms:
  - name: "gpt-4o"
    endpoint: "api"
    temperature_range: [0.2, 0.7, 1.0]

target_countries:
  - "Romania"
  - "Republic of Moldova"

viewpoints:
  - "positive"
  - "neutral"
  - "negative"
  
lengths:
  - "short"
  - "medium"
  - "long"
  
platforms:
  - "știre_portal_standard"
  - "postare_telegram"
  - "fir_reddit"
  - "text_tiktok"
  - "postare_facebook"
    
personas:
  - "jurnalist_quality"
  - "redactor_tabloid"
  - "bot_traducere_automata"
  - "blogger_opinie"
  - "influencer_social_media"
  - "admin_telegram"

topics:
  - domain: "Politică"
    subdomains: ["Alegeri", "Relații Externe", "Corupție"]
  - domain: "Economie"
    subdomains: ["Inflație", "Imobiliare", "Criptomonede"]
  - domain: "Sănătate"
    subdomains: ["Pandemie", "Sistemul Medical", "Nutriție"]
  - domain: "Tehnologie"
    subdomains: ["Inteligență Artificială", "Cybersecurity", "Gadgeturi"]
  - domain: "Divertisment"
    subdomains: ["Vedete", "Filme", "Scandaluri"]
  - domain: "Educație"
    subdomains: ["Reforme", "Examene Naționale", "Abandon Școlar"]
  - domain: "Sport"
    subdomains: ["Fotbal", "Jocurile Olimpice", "Campionate"]
  - domain: "Mediu"
    subdomains: ["Schimbări Climatice", "Poluare", "Energie Verde"]
  - domain: "Societate"
    subdomains: ["Proteste", "Drepturile Omului", "Pensii"]
  - domain: "Justiție"
    subdomains: ["Legi Noi", "Dosare Penale", "Magistrați"]
  - domain: "Agricultură"
    subdomains: ["Secetă și Recolta", "Exporturi de Cereale", "Fermieri vs. Supermarketuri"]
  - domain: "Infrastructură"
    subdomains: ["Autostrăzi", "Transport Feroviar", "Fonduri PNRR", "Lucrări Publice Întârziate"]
  - domain: "Apărare și Securitate"
    subdomains: ["Baza NATO", "Războiul din Ucraina", "Achiziții Militare", "Securitate Cibernetică Națională"]
  - domain: "Diaspora și Migrație"
    subdomains: ["Condiții de Muncă în Străinătate", "Remitențe", "Întoarcerea Acasă", "Muncitori Asiatici în RO/MD"]
  - domain: "Turism"
    subdomains: ["Prețuri pe Litoral", "Turism Montan", "Criza HoReCa", "Promovare Turistică"]
  - domain: "Religie și Tradiții"
    subdomains: ["Biserica Ortodoxă", "Pelerinaje", "Sărbători Legale", "Dispute Clericale"]
  - domain: "Cultură"
    subdomains: ["Festivaluri de Muzică (Untold/Electric)", "Patrimoniu UNESCO", "Expoziții de Artă", "Finanțarea Culturii"]
  - domain: "Internațional"
    subdomains: ["Directive UE", "Alegeri SUA", "Crize Geopolitice", "Schengen"]
  - domain: "Piața Muncii"
    subdomains: ["Șomaj", "Work from Home", "Sindicate și Greve", "Generația Z la Job"]

output_directory: "../generated_dataset/"
```

## 7. Implementation Plan & Testable Features

*   **[x] Feature 1: Config & Wikipedia Scraper**: Parse YAML, fetch random Wikipedia text.
*   **[x] Feature 2: Prompt Engine**: Construct prompts integrating Geography, Personas, Viewpoint, and Length without errors.
*   **[x] Feature 3: LLM Structured Interface**: Guarantee JSON deserialization via `response_format` tools.
*   **[x] Feature 4: Schema Assembly & Validation**: Map the final pipeline output through Pydantic.