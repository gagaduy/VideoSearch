from openai import OpenAI


def expand_query(query: str, api_key: str, model: str) -> list[str]:
    if not api_key:
        return [query]

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=model,
            input=(
                "Generate three concise paraphrases for video retrieval. "
                f"Return one per line.\nQuery: {query}"
            ),
        )
        text = response.output_text.strip()
        expansions = [line.strip("- ").strip() for line in text.splitlines() if line.strip()]
        unique = []
        for item in [query, *expansions]:
            if item not in unique:
                unique.append(item)
        return unique or [query]
    except Exception:
        return [query]

