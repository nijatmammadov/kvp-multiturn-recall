"""Turn retrieval for --offload_retrieve (batch v7): pick the earlier user turn that best matches the new user turn.

Lexical: content words (lowercase alphanumerics, length >= 2, minus a small stopword list); a past turn qualifies if it
shares >= MIN_SHARED content words with the new turn; score = sum over shared words of IDF computed over the past user
turns. Highest score wins; ties go to the more recent turn.
"""
import math
import re

MIN_SHARED = 2
STOP = set("""a an and are as at be been being but by can could did do does doing for from had has have having how i if
in into is it its itself me my myself of on or our ours please so than that the their theirs them then there these they
this those to was we were what when where which who whom why will with would you your yours about after again all also
any because before both each few further here him his her hers more most no nor not now once only other out over own
same she should some such too under until up very just""".split())


def content_words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 2 and w not in STOP}


def choose_turn(past_texts, query):
    """Index into past_texts of the best-matching turn and its score, or (None, 0.0) if none shares MIN_SHARED words."""
    q = content_words(query)
    docs = [content_words(t) for t in past_texts]
    n = len(docs)
    best, best_score = None, 0.0
    for j, d in enumerate(docs):
        shared = q & d
        if len(shared) < MIN_SHARED:
            continue
        score = sum(math.log((n + 1) / (1 + sum(w in dd for dd in docs))) + 1 for w in shared)
        if best is None or score >= best_score:
            best, best_score = j, score
    return best, best_score
