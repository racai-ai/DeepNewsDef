class PromptEngine:
    """
    Constructs prompts integrating Geography, Personas, Viewpoint, Topic, Length, Platform Formats, and Grammar.
    """

    @staticmethod
    def build_complete_prompt(persona: str, country: str, length: str, platform_format: str, topic: str, viewpoint: str, grammar: str, manipulation: str, propaganda: str, audience: str, urgency: str, formatting: str) -> str:
        """Generates a prompt for pure synthetic generation."""
        return (
            f"Vei adopta rolul de {persona} din {country}. "
            f"Creează un conținut text pentru {platform_format}, care să abordeze tema '{topic}'. "
            f"Dimensiunea textului trebuie să fie {length}, iar abordarea sau tonul general trebuie să fie de natură {viewpoint}. "
            f"Publicul tău țintă este: {audience}. "
            f"Stilul de formatație cerut: {formatting}. "
            f"Nivelul de urgență și viteza narativă a textului: {urgency}. "
            f"Mai mult, nivelul de exprimare și corectitudine gramaticală a textului trebuie să fie: {grammar}. "
            f"Nivelul de intenție manipulatoare al textului (subiectivism, propagandă, denaturare) va fi: {manipulation}. "
            f"Folosește următoarea tehnică sau eroare logică: {propaganda}. "
            f"Returnează rezultatul exclusiv sub formă de obiect JSON valid, care să conțină o cheie `title` pentru titlul textului și o cheie `paragraphs` cu o listă a paragrafelor."
        )

    @staticmethod
    def build_based_on_real_article_prompt(persona: str, country: str, length: str, platform_format: str, viewpoint: str, seed: str, grammar: str, manipulation: str, propaganda: str, audience: str, urgency: str, formatting: str) -> str:
        """Generates a prompt for thematic inspiration based on a Wikipedia seed."""
        return (
            f"Având la dispoziție următorul extras factual drept sursă de inspirație:\n"
            f"```\n{seed}\n```\n"
            f"Vei adopta rolul de {persona} din {country} și vei scrie un text complet nou. "
            f"Produsul final trebuie gândit pentru {platform_format}, să aibă o întindere {length} și să reflecte o opinie sau un ton evident {viewpoint}. "
            f"Textul se adresează către următorul segment demografic: {audience}. "
            f"Nivelul tău de alertă ori profunzime va fi: {urgency}. "
            f"Ca particularitate vizuală / stilistică, vei folosi: {formatting}. "
            f"Asigură-te că nivelul abilităților gramaticale se pliază pe următoarea descriere: {grammar}. "
            f"Nivelul tău de paritate față de adevăr și manipulare trebuie să fie: {manipulation}. "
            f"Aplică obligatoriu mecanismul discursiv sau sofismul: {propaganda}. "
            f"Returnează exclusiv un obiect JSON valid, compus dintr-o cheie `title` și o cheie `paragraphs` (conținând lista de paragrafe)."
        )

    @staticmethod
    def build_based_on_topic_prompt(persona: str, country: str, length: str, topic: str, viewpoint: str, grammar: str, manipulation: str, propaganda: str, audience: str, urgency: str, formatting: str) -> str:
        """Generates a prompt for thematic fiction."""
        return (
            f"Intră în pielea unui {persona} din teritoriul {country}. "
            f"Misiunea ta este să concepi un material original despre '{topic}'. Lungimea textului trebuie să fie {length}, incluzând o perspectivă profund {viewpoint}. "
            f"Publicul tău este format din: {audience}. Ritmul și senzația de urgență transmise de text: {urgency}. "
            f"Redactează conform acestui stil de redactare/estetică generală: {formatting}. "
            f"De asemenea, redactarea va respecta un nivel de corectitudine gramaticală descris astfel: {grammar}. "
            f"În ceea ce privește latura manipulatoare sau puritatea faptică, materialul tău va acționa extrem de precis după modelul: {manipulation}. "
            f"Folosește neapărat din plin următoarea tehnică retorică specifică polarizării depline: {propaganda}. "
            f"Oferă ca răspuns doar cod JSON valid. JSON-ul trebuie să aibă două elemente: `title` (titlul propus) și `paragraphs` (lista de paragrafe generate)."
        )

    @staticmethod
    def get_system_prompt() -> str:
        """Generates the system prompt to enforce JSON output."""
        return (
             "Ești un asistent AI creat pentru a genera structuri scurte de text sintetic, folosind exclusiv limba română corectă și naturală. "
             "Răspunsul tău trebuie să fie absolut întotdeauna un singur obiect JSON valid. "
             "Nu folosi marcaje markdown (precum ```json) și nu adăuga niciun fel de text explicativ în afara corpului JSON."
        )
