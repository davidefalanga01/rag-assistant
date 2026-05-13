from dataclasses import dataclass
import re
from typing import Any, Iterable, Optional

from llama_index.core.vector_stores import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)


COMPANY_ALIASES = {
    "AAPL": ("aapl", "apple", "apple inc", "apple inc."),
}

SECTION_RULES = (
    (("risk factor", "risk factors", "material adverse"), ("ITEM 1A",), ("risk_factors",)),
    (("cybersecurity", "cyber security", "information security"), ("ITEM 1C",), ("cybersecurity",)),
    (("business overview", "human capital", "seasonality", "product introduction"), ("ITEM 1",), ("business",)),
    (("property", "properties", "facilities"), ("ITEM 2",), ("properties",)),
    (("legal proceeding", "legal proceedings", "litigation"), ("ITEM 3",), ("legal_proceedings",)),
    (("mine safety",), ("ITEM 4",), ("mine_safety",)),
    (("market for registrant", "common equity", "share repurchase", "dividend"), ("ITEM 5",), ("market_equity",)),
    (("management discussion", "md&a", "mda", "results of operations", "liquidity", "capital resources"), ("ITEM 7",), ("mda",)),
    (("market risk", "foreign exchange risk", "interest rate risk"), ("ITEM 7A",), ("market_risk",)),
    (("financial statement", "financial statements", "supplementary data"), ("ITEM 8",), ("financial_statements",)),
    (("accountant disagreement", "accounting disagreement"), ("ITEM 9",), ("accounting_disagreements",)),
    (("controls and procedures", "internal control", "disclosure control"), ("ITEM 9A",), ("controls_procedures",)),
    (("other information", "trading plan", "rule 10b5-1"), ("ITEM 9B",), ("other_information",)),
    (("foreign jurisdiction",), ("ITEM 9C",), ("foreign_jurisdictions",)),
    (("director", "executive officer", "governance"), ("ITEM 10",), ("governance",)),
    (("executive compensation", "compensation"), ("ITEM 11",), ("executive_compensation",)),
    (("security ownership", "beneficial ownership"), ("ITEM 12",), ("security_ownership",)),
    (("related transaction", "related party"), ("ITEM 13",), ("related_transactions",)),
    (("accountant fees", "audit fees", "principal accountant"), ("ITEM 14",), ("accountant_fees",)),
    (("exhibit", "exhibits", "financial statement schedule"), ("ITEM 15",), ("exhibits",)),
    (("form 10-k summary", "10-k summary"), ("ITEM 16",), ("form_10k_summary",)),
)

STATEMENT_TYPE_RULES = (
    (("statement of operations", "statements of operations", "income statement"), "income_statement"),
    (("balance sheet", "balance sheets"), "balance_sheet"),
    (("cash flow", "cash flows", "operating activities", "investing activities", "financing activities"), "cash_flow_statement"),
    (("shareholders' equity", "shareholders equity", "stockholders' equity", "stockholders equity"), "shareholders_equity"),
    (("comprehensive income",), "comprehensive_income"),
    (("notes to consolidated financial statements", "notes to financial statements"), "notes"),
    (("auditor report", "independent registered public accounting firm"), "auditor_report"),
)

EXACT_ITEM_PATTERN = re.compile(r"\bitem\s+(\d+[a-z]?)\b", re.IGNORECASE)
FISCAL_YEAR_PATTERNS = (
    re.compile(r"\bfiscal\s+(?:year\s+)?(20\d{2})\b", re.IGNORECASE),
    re.compile(r"\bfy\s*(20\d{2})\b", re.IGNORECASE),
    re.compile(r"\bform\s+10-?k\s+(?:for\s+)?(?:fiscal\s+year\s+)?(20\d{2})\b", re.IGNORECASE),
    re.compile(r"\b(20\d{2})\s+form\s+10-?k\b", re.IGNORECASE),
)


@dataclass(frozen=True)
class RetrievalFilter:
    ticker: Optional[str] = None
    fiscal_year: Optional[int] = None
    form_type: Optional[str] = None
    sec_items: tuple[str, ...] = ()
    section_groups: tuple[str, ...] = ()
    chunk_type: Optional[str] = None
    statement_type: Optional[str] = None

    def is_empty(self) -> bool:
        return not any(
            (
                self.ticker,
                self.fiscal_year,
                self.form_type,
                self.sec_items,
                self.section_groups,
                self.chunk_type,
                self.statement_type,
            )
        )

    def to_llama_filters(self) -> Optional[MetadataFilters]:
        filters = []
        self._add_scalar_filter(filters, "ticker", self.ticker)
        self._add_scalar_filter(filters, "fiscal_year", self.fiscal_year)
        self._add_scalar_filter(filters, "form_type", self.form_type)
        self._add_tuple_filter(filters, "sec_item", self.sec_items)
        self._add_tuple_filter(filters, "section_group", self.section_groups)
        self._add_scalar_filter(filters, "chunk_type", self.chunk_type)
        self._add_scalar_filter(filters, "statement_type", self.statement_type)

        if not filters:
            return None

        return MetadataFilters(filters=filters, condition=FilterCondition.AND)

    def matches_node(self, node: Any) -> bool:
        metadata = node.metadata or {}
        return all(
            (
                self._matches_scalar(metadata, "ticker", self.ticker),
                self._matches_scalar(metadata, "fiscal_year", self.fiscal_year),
                self._matches_scalar(metadata, "form_type", self.form_type),
                self._matches_tuple(metadata, "sec_item", self.sec_items),
                self._matches_tuple(metadata, "section_group", self.section_groups),
                self._matches_scalar(metadata, "chunk_type", self.chunk_type),
                self._matches_scalar(metadata, "statement_type", self.statement_type),
            )
        )

    def merge(self, other: "RetrievalFilter") -> "RetrievalFilter":
        return RetrievalFilter(
            ticker=other.ticker or self.ticker,
            fiscal_year=other.fiscal_year or self.fiscal_year,
            form_type=other.form_type or self.form_type,
            sec_items=_merge_tuple_values(self.sec_items, other.sec_items),
            section_groups=_merge_tuple_values(self.section_groups, other.section_groups),
            chunk_type=other.chunk_type or self.chunk_type,
            statement_type=other.statement_type or self.statement_type,
        )

    @staticmethod
    def _add_scalar_filter(filters: list[MetadataFilter], key: str, value: Any) -> None:
        if value is None:
            return
        filters.append(MetadataFilter(key=key, operator=FilterOperator.EQ, value=value))

    @staticmethod
    def _add_tuple_filter(filters: list[MetadataFilter], key: str, values: tuple[str, ...]) -> None:
        if not values:
            return
        if len(values) == 1:
            filters.append(MetadataFilter(key=key, operator=FilterOperator.EQ, value=values[0]))
        else:
            filters.append(MetadataFilter(key=key, operator=FilterOperator.IN, value=list(values)))

    @staticmethod
    def _matches_scalar(metadata: dict, key: str, value: Any) -> bool:
        return value is None or metadata.get(key) == value

    @staticmethod
    def _matches_tuple(metadata: dict, key: str, values: tuple[str, ...]) -> bool:
        return not values or metadata.get(key) in values


def infer_retrieval_filter(query: str) -> RetrievalFilter:
    """Infer conservative metadata filters from a natural-language query."""
    query_lower = query.lower()

    ticker = _infer_ticker(query_lower)
    fiscal_year = _infer_fiscal_year(query)
    form_type = "10-K" if re.search(r"\b10-?k\b", query_lower) else None
    sec_items = _unique(_extract_explicit_sec_items(query))
    section_groups: tuple[str, ...] = ()
    statement_type = _infer_statement_type(query_lower)
    chunk_type = _infer_chunk_type(query_lower)

    if not sec_items:
        sec_items, section_groups = _infer_sections(query_lower)

    if statement_type:
        sec_items = _unique((*sec_items, "ITEM 8"))
        section_groups = _unique((*section_groups, "financial_statements"))

    return RetrievalFilter(
        ticker=ticker,
        fiscal_year=fiscal_year,
        form_type=form_type,
        sec_items=sec_items,
        section_groups=section_groups,
        chunk_type=chunk_type,
        statement_type=statement_type,
    )


def _infer_ticker(query_lower: str) -> Optional[str]:
    for ticker, aliases in COMPANY_ALIASES.items():
        if any(_contains_phrase(query_lower, alias) for alias in aliases):
            return ticker
    return None


def _infer_fiscal_year(query: str) -> Optional[int]:
    for pattern in FISCAL_YEAR_PATTERNS:
        match = pattern.search(query)
        if match:
            return int(match.group(1))
    return None


def _extract_explicit_sec_items(query: str) -> tuple[str, ...]:
    return tuple(f"ITEM {match.group(1).upper()}" for match in EXACT_ITEM_PATTERN.finditer(query))


def _infer_sections(query_lower: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    sec_items = []
    section_groups = []

    for phrases, item_values, group_values in SECTION_RULES:
        if any(_contains_phrase(query_lower, phrase) for phrase in phrases):
            sec_items.extend(item_values)
            section_groups.extend(group_values)

    return _unique(sec_items), _unique(section_groups)


def _infer_statement_type(query_lower: str) -> Optional[str]:
    for phrases, statement_type in STATEMENT_TYPE_RULES:
        if any(_contains_phrase(query_lower, phrase) for phrase in phrases):
            return statement_type
    return None


def _infer_chunk_type(query_lower: str) -> Optional[str]:
    if any(_contains_phrase(query_lower, phrase) for phrase in ("table", "tabular", "schedule")):
        return "table"
    return None


def _contains_phrase(text: str, phrase: str) -> bool:
    escaped = re.escape(phrase.lower())
    return re.search(rf"(?<!\w){escaped}(?!\w)", text) is not None


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _merge_tuple_values(base_values: tuple[str, ...], other_values: tuple[str, ...]) -> tuple[str, ...]:
    if not base_values:
        return other_values
    if not other_values:
        return base_values

    overlap = tuple(value for value in other_values if value in base_values)
    if overlap:
        return overlap

    return other_values
