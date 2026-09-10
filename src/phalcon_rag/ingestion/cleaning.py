import re

REFERENCE_DEFINITION_PATTERN = re.compile(r"^\s*\[[^\]]+\]:\s+\S+.*$", re.MULTILINE)
API_USES_PATTERN = re.compile(r"<ApiUses>.*?</ApiUses>", re.DOTALL)
API_TREE_PATTERN = re.compile(r"<ApiTree>.*?</ApiTree>", re.DOTALL)
API_USED_BY_PATTERN = re.compile(r"<ApiUsedBy>.*?</ApiUsedBy>", re.DOTALL)
MAX_LINES = 20

def clean_content(content: str) -> str:
    content = REFERENCE_DEFINITION_PATTERN.sub("", content)
    content = API_USES_PATTERN.sub("", content)
    content = API_USED_BY_PATTERN.sub("", content)
    content = API_TREE_PATTERN.sub(replace, content)

    return re.sub(r"\n{3,}", "\n\n", content)

def replace(match: re.Match) -> str:
    tree = match.group(0)
    return "" if len(tree.splitlines()) > MAX_LINES else tree