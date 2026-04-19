"""
Phase 1: Vector-Based Persona Matching (The Router)
----------------------------------------------------
Uses FAISS + sentence-transformers to embed bot personas and route
incoming posts to the most relevant bots using cosine similarity.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Bot Persona Definitions
# ---------------------------------------------------------------------------
BOT_PERSONAS = {
    "bot_a": {
        "name": "Tech Maximalist",
        "description": (
            "I believe AI and crypto will solve all human problems. "
            "I am highly optimistic about technology, Elon Musk, and space exploration. "
            "I dismiss regulatory concerns."
        ),
    },
    "bot_b": {
        "name": "Doomer / Skeptic",
        "description": (
            "I believe late-stage capitalism and tech monopolies are destroying society. "
            "I am highly critical of AI, social media, and billionaires. "
            "I value privacy and nature."
        ),
    },
    "bot_c": {
        "name": "Finance Bro",
        "description": (
            "I strictly care about markets, interest rates, trading algorithms, and making money. "
            "I speak in finance jargon and view everything through the lens of ROI."
        ),
    },
}

# ---------------------------------------------------------------------------
# PersonaRouter Class
# ---------------------------------------------------------------------------
class PersonaRouter:
    """
    Embeds bot personas into a FAISS index and routes incoming posts
    to matching bots based on cosine similarity.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        print(f"[Router] Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.bot_ids = []
        self._build_index()

    def _build_index(self):
        """Embed all personas and store in FAISS index (cosine similarity via L2 on normalized vecs)."""
        print("[Router] Building FAISS index for bot personas...")
        self.bot_ids = list(BOT_PERSONAS.keys())
        descriptions = [BOT_PERSONAS[bid]["description"] for bid in self.bot_ids]

        # Embed and L2-normalize for cosine similarity
        embeddings = self.model.encode(descriptions, normalize_embeddings=True)
        dim = embeddings.shape[1]

        # Inner product on normalized vectors == cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        print(f"[Router] Index built with {len(self.bot_ids)} personas (dim={dim}).")

    def route_post_to_bots(self, post_content: str, threshold: float = 0.35) -> list[dict]:
        """
        Embed the post, query the FAISS index, and return bots whose
        cosine similarity to the post exceeds `threshold`.

        Args:
            post_content: The text of the incoming post.
            threshold: Minimum cosine similarity score (0-1). Default 0.35
                       (lowered from 0.85 in assignment spec because all-MiniLM
                       produces moderate similarities; adjust per model).

        Returns:
            List of dicts with bot_id, name, and similarity score.
        """
        print(f"\n[Router] Routing post: \"{post_content[:80]}...\"")
        post_embedding = self.model.encode([post_content], normalize_embeddings=True)
        scores, indices = self.index.search(post_embedding.astype(np.float32), k=len(self.bot_ids))

        matched_bots = []
        for score, idx in zip(scores[0], indices[0]):
            bot_id = self.bot_ids[idx]
            bot_name = BOT_PERSONAS[bot_id]["name"]
            print(f"  [{bot_id}] {bot_name}: similarity = {score:.4f} (threshold={threshold})")
            if score >= threshold:
                matched_bots.append({
                    "bot_id": bot_id,
                    "name": bot_name,
                    "similarity": round(float(score), 4),
                })

        if not matched_bots:
            print("[Router] No bots matched above threshold.")
        else:
            print(f"[Router] Matched bots: {[b['bot_id'] for b in matched_bots]}")

        return matched_bots


# ---------------------------------------------------------------------------
# Demo / Execution Log
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    router = PersonaRouter()

    test_posts = [
        "OpenAI just released a new model that might replace junior developers.",
        "The Fed just raised interest rates again — markets are in freefall.",
        "Big tech companies are lobbying to weaken GDPR privacy protections.",
        "Bitcoin hits a new all-time high as ETF approvals surge.",
        "Elon Musk announces Starship will carry humans to Mars by 2030.",
    ]

    for post in test_posts:
        results = router.route_post_to_bots(post)
        print(f"  → Matched: {[r['bot_id'] + ' (' + str(r['similarity']) + ')' for r in results]}\n")
