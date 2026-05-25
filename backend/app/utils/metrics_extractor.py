"""
Metrics Extractor
Extracts the 21 features used by the API from raw Python/Java source code.
For production use, CKJMExtended handles Java; Radon handles Python.
"""

from __future__ import annotations
import re
from typing import Dict


def extract_metrics_from_code(source_code: str, language: str = "python") -> Dict:
    """
    Extract software metrics from raw source code.

    For Python, uses Radon when available; falls back to regex-based
    approximations so the service stays functional without Radon installed.
    For Java, regex-based approximations are used (CKJMExtended is a
    standalone JAR invoked from the training pipeline, not at inference time).
    """
    if language == "python":
        return _extract_python(source_code)
    elif language == "java":
        return _extract_java_approx(source_code)
    else:
        raise ValueError(f"Unsupported language: {language}")


def _extract_python(code: str) -> Dict:
    try:
        return _extract_python_radon(code)
    except ImportError:
        return _extract_python_regex(code)


def _extract_python_radon(code: str) -> Dict:
    from radon.metrics import h_visit, mi_visit
    from radon.complexity import cc_visit
    from radon.raw import analyze

    raw = analyze(code)
    cc_blocks = cc_visit(code)
    avg_cc = sum(b.complexity for b in cc_blocks) / max(len(cc_blocks), 1)
    h = h_visit(code)

    methods = [b for b in cc_blocks if b.type in ("method", "function")]
    num_methods = len(methods)

    # Attribute approximation: count class-level assignments
    num_attributes = len(re.findall(r"^\s{4}self\.\w+\s*=", code, re.MULTILINE))

    return {
        "loc": raw.loc,
        "lloc": raw.lloc,
        "num_methods": num_methods,
        "num_attributes": num_attributes,
        "cyclomatic_complexity": round(avg_cc, 2),
        "halstead_volume": round(h.total.volume, 2) if h else 0,
        "halstead_effort": round(h.total.effort, 2) if h else 0,
        "halstead_difficulty": round(h.total.difficulty, 2) if h else 0,
        # OO metrics — approximated from code structure
        "cbo": _approx_cbo(code),
        "wmc": round(sum(b.complexity for b in cc_blocks), 2),
        "rfc": _approx_rfc(code),
        "lcom": _approx_lcom(code),
        "dit": _approx_dit(code),
        "noc": 0,  # Requires project-wide analysis
        # Process metrics — not available from static code alone; default to 0
        "num_authors": 0,
        "code_churn": 0,
        "prior_defects": 0,
        "module_age_days": 0,
        # Dependency metrics
        "fan_in": 0,
        "fan_out": _approx_fan_out(code),
        "instability": 0.5,
    }


def _extract_python_regex(code: str) -> Dict:
    lines = code.splitlines()
    loc = len(lines)
    lloc = len([l for l in lines if l.strip() and not l.strip().startswith("#")])
    num_methods = len(re.findall(r"^\s*def\s+\w+", code, re.MULTILINE))
    num_attributes = len(re.findall(r"^\s{4}self\.\w+\s*=", code, re.MULTILINE))
    num_classes = len(re.findall(r"^\s*class\s+\w+", code, re.MULTILINE))

    return {
        "loc": loc, "lloc": lloc,
        "num_methods": num_methods, "num_attributes": num_attributes,
        "cyclomatic_complexity": _approx_cc(code),
        "halstead_volume": loc * 10.0,
        "halstead_effort": loc * 200.0,
        "halstead_difficulty": min(loc / 10, 40),
        "cbo": _approx_cbo(code), "wmc": num_methods * 2,
        "rfc": _approx_rfc(code), "lcom": _approx_lcom(code),
        "dit": _approx_dit(code), "noc": 0,
        "num_authors": 0, "code_churn": 0, "prior_defects": 0, "module_age_days": 0,
        "fan_in": 0, "fan_out": _approx_fan_out(code), "instability": 0.5,
    }


def _extract_java_approx(code: str) -> Dict:
    lines = code.splitlines()
    loc = len(lines)
    lloc = len([l for l in lines if l.strip() and not l.strip().startswith("//")])
    num_methods = len(re.findall(r"(public|private|protected)\s+\w[\w<>\[\]]*\s+\w+\s*\(", code))
    num_attributes = len(re.findall(r"(private|protected|public)\s+\w[\w<>\[\]]*\s+\w+\s*;", code))

    return {
        "loc": loc, "lloc": lloc,
        "num_methods": num_methods, "num_attributes": num_attributes,
        "cyclomatic_complexity": _approx_cc(code),
        "halstead_volume": loc * 10.0, "halstead_effort": loc * 200.0, "halstead_difficulty": min(loc / 10, 40),
        "cbo": _approx_cbo(code), "wmc": num_methods * 2,
        "rfc": _approx_rfc(code), "lcom": _approx_lcom(code),
        "dit": _approx_dit(code), "noc": 0,
        "num_authors": 0, "code_churn": 0, "prior_defects": 0, "module_age_days": 0,
        "fan_in": 0, "fan_out": _approx_fan_out(code), "instability": 0.5,
    }


# ---------------------------------------------------------------------------
# Approximation helpers
# ---------------------------------------------------------------------------

def _approx_cc(code: str) -> float:
    """Approximate cyclomatic complexity by counting decision points."""
    keywords = ["if ", "elif ", "else:", "for ", "while ", "except", "and ", "or "]
    count = sum(len(re.findall(rf"\b{re.escape(kw)}", code)) for kw in keywords)
    return max(1.0, float(count))


def _approx_cbo(code: str) -> int:
    imports = re.findall(r"^(?:import|from)\s+(\w+)", code, re.MULTILINE)
    return len(set(imports))


def _approx_rfc(code: str) -> int:
    calls = re.findall(r"\w+\.\w+\s*\(", code)
    return len(set(calls))


def _approx_lcom(code: str) -> float:
    methods = re.findall(r"def\s+\w+[^:]*:(.*?)(?=\ndef|\Z)", code, re.DOTALL)
    if len(methods) < 2:
        return 0.0
    attrs_per_method = []
    all_attrs = set(re.findall(r"self\.(\w+)", code))
    for m in methods:
        used = set(re.findall(r"self\.(\w+)", m))
        attrs_per_method.append(used)
    pairs = [(i, j) for i in range(len(methods)) for j in range(i + 1, len(methods))]
    if not pairs:
        return 0.0
    shared = sum(1 for i, j in pairs if attrs_per_method[i] & attrs_per_method[j])
    return round(1 - (shared / len(pairs)), 3)


def _approx_dit(code: str) -> int:
    inheritances = re.findall(r"class\s+\w+\s*\(([^)]+)\)", code)
    return len(inheritances)


def _approx_fan_out(code: str) -> int:
    return _approx_cbo(code)
