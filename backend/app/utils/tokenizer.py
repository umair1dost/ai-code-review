"""
Source Code Tokenizer
Converts raw source code to a fixed-length integer token sequence
for LSTM inference. Mirrors the preprocessing used during training.
"""

from __future__ import annotations
import re
from typing import List

VOCAB_SIZE = 10_000
MAX_SEQ_LEN = 512
UNK_TOKEN = 1
PAD_TOKEN = 0

# Minimal built-in vocabulary — the real vocabulary is serialised alongside
# the LSTM model as tokenizer.json and loaded from MODEL_DIR at runtime.
_BUILTIN_KEYWORDS = [
    "def", "class", "return", "if", "else", "elif", "for", "while",
    "import", "from", "try", "except", "with", "as", "pass", "raise",
    "True", "False", "None", "self", "super", "lambda", "yield",
    "public", "private", "protected", "static", "void", "int", "String",
    "new", "this", "extends", "implements", "interface", "abstract",
]
_KW_TO_ID = {kw: idx + 2 for idx, kw in enumerate(_BUILTIN_KEYWORDS)}


def tokenize_code(source_code: str, language: str = "python") -> List[int]:
    """
    Tokenise source code into a padded integer sequence of length MAX_SEQ_LEN.

    In production, the vocabulary is loaded from the serialised tokenizer
    saved alongside the LSTM model. This function uses a minimal fallback
    vocabulary so the API stays functional without model artefacts.
    """
    try:
        return _tokenize_with_pygments(source_code, language)
    except Exception:
        return _tokenize_simple(source_code)


def _tokenize_with_pygments(source_code: str, language: str) -> List[int]:
    from pygments.lexers import PythonLexer, JavaLexer
    from pygments.token import Token

    lexer = PythonLexer() if language == "python" else JavaLexer()
    tokens = [value for ttype, value in lexer.get_tokens(source_code)
              if ttype not in Token.Comment and value.strip()]

    ids = [_KW_TO_ID.get(tok, UNK_TOKEN) for tok in tokens]
    return _pad_or_truncate(ids)


def _tokenize_simple(source_code: str) -> List[int]:
    tokens = re.findall(r"[a-zA-Z_]\w*|[0-9]+|[^\s\w]", source_code)
    ids = [_KW_TO_ID.get(tok, UNK_TOKEN) for tok in tokens]
    return _pad_or_truncate(ids)


def _pad_or_truncate(ids: List[int]) -> List[int]:
    if len(ids) >= MAX_SEQ_LEN:
        return ids[:MAX_SEQ_LEN]
    return ids + [PAD_TOKEN] * (MAX_SEQ_LEN - len(ids))
