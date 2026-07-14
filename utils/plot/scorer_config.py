SCORERS = {
    "finetuned_qualt5": {
        "enabled": True,
        "group": "qualt5",
        "cache_patterns": [
            "finetuned_qualt5_{dataset_name}.cache",
            "finetunedqualt5_{dataset_name}.cache",
        ],
        "compare_aliases": [
            "finetuned_qualt5",
            "finetunedqualt5",
        ],
        "label": "QualT5-Finetuned",
        "color": "#1f77b4",
        "linestyle": "-",
        "marker": "o",
        "higher_is_better": True,
    },

    "metadata_qualt5_concat_v2": {
        "enabled": True,
        "group": "metadata_qualt5",
        "cache_patterns": [
            "metadata_qualt5_{dataset_name}_CONCAT-V2-FA-ck1.cache",
        ],
        "compare_aliases": [
            "metadata_qualt5_CONCAT-V2-FA-ck1",

        ],
        "label": "Metadata-QualT5-CONCAT-V2",
        "color": "#01FF16",
        "linestyle": "--",
        "marker": "s",
        "higher_is_better": True,
    },

    "metadata_qualt5_allmetapj": {
        "enabled": True,
        "group": "metadata_qualt5",
        "cache_patterns": [
            "metadata_qualt5_{dataset_name}_ALLMETAPJ_ck1.cache",
        ],
        "compare_aliases": [
            "metadata_qualt5_ALLMETAPJ_ck1",
        ],
        "label": "Metadata-QualT5-ALLMETAPJ",
        "color": "#FF0105",
        "linestyle": "--",
        "marker": "x",
        "higher_is_better": True,
    },

    "metadata_qualt5_attfus": {
        "enabled": True,
        "group": "metadata_qualt5",
        "cache_patterns": [
            "metadata_qualt5_{dataset_name}_ATTFUS_ck5.cache",
        ],
        "compare_aliases": [

            "metadata_qualt5_ATTFUS_ck5",

        ],
        "label": "Metadata-QualT5-ATTFUS",
        "color": "#FF01AA",
        "linestyle": "--",
        "marker": "v",
        "higher_is_better": True,
    },

    "metadata_qualt5_mp": {
        "enabled": False,
        "group": "metadata_qualt5",
        "cache_patterns": [
            "metadata_qualt5_{dataset_name}_MP.cache",
        ],
        "compare_aliases": [
            "metadata_qualt5_mp",
            "metadata_qualt5_MP",
        ],
        "label": "Metadata-QualT5 MP",
        "color": "#ff7f0e",
        "linestyle": "-.",
        "marker": "^",
        "higher_is_better": True,
    },

    "metadata_qualt5_pooled": {
        "enabled": False,
        "group": "metadata_qualt5",
        "cache_patterns": [
            "metadata_qualt5_{dataset_name}_POOLEDCONCAT.cache",
        ],
        "compare_aliases": [
            "metadata_qualt5_pooled",
            "metadata_qualt5_pooledconcat",
            "metadata_qualt5_POOLEDCONCAT",
            "metadata_qualt5_POOLED_CONCAT",
        ],
        "label": "Metadata-QualT5 pooled",
        "color": "#9467bd",
        "linestyle": ":",
        "marker": "D",
        "higher_is_better": True,
    },

    "metadata_qualt5_concat_v1": {
        "enabled": False,
        "group": "metadata_qualt5",
        "cache_patterns": [
            "metadata_qualt5_{dataset_name}_CONCAT.cache",
        ],
        "compare_aliases": [
            "metadata_qualt5_CONCAT",
        ],
        "label": "Metadata-QualT5 CONCAT V1",
        "color": "#8c564b",
        "linestyle": "--",
        "marker": "x",
        "higher_is_better": True,
    },

    "tasb": {
        "enabled": False,
        "group": "baseline",
        "cache_patterns": [
            "tasb_{dataset_name}.cache",
        ],
        "compare_aliases": ["tasb", "TASB", "TAS-B"],
        "label": "TASB-Mag",
        "color": "#2ca02c",
        "linestyle": "-",
        "marker": "^",
        "higher_is_better": True,
    },

    "perplexity": {
        "enabled": False,
        "group": "baseline",
        "cache_patterns": [
            "perplexity_{dataset_name}.cache",
        ],
        "compare_aliases": ["perplexity", "ppl", "T5-Ppl"],
        "label": "T5-Ppl",
        "color": "#e377c2",
        "linestyle": "--",
        "marker": "v",
        "higher_is_better": False,
    },

    "itn": {
        "enabled": False,
        "group": "baseline",
        "cache_patterns": [
            "itn_{dataset_name}.cache",
        ],
        "compare_aliases": ["itn", "ITN"],
        "label": "ITN",
        "color": "#d62728",
        "linestyle": "-",
        "marker": "h",
        "higher_is_better": True,
    },

    "cdd": {
        "enabled": False,
        "group": "baseline",
        "cache_patterns": [
            "cdd_{dataset_name}.cache",
        ],
        "compare_aliases": ["cdd", "CDD"],
        "label": "CDD",
        "color": "#8c564b",
        "linestyle": "--",
        "marker": "*",
        "higher_is_better": True,
    },
}


BASE_MODEL = "finetuned_qualt5"


def parse_csv_arg(value):
    """
    Converte:
        "a,b,c" -> ["a", "b", "c"]

    Se value è None o stringa vuota, ritorna None.
    """
    if value is None:
        return None

    value = str(value).strip()

    if value == "":
        return None

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def normalize_name(value):
    if value is None:
        return None

    value = str(value).strip()

    while "__" in value:
        value = value.replace("__", "_")

    return value


def get_scorer_name_from_cache(filename, dataset_name):
    """
    Risolve il nome dello scorer usando solo scorer_config.py.
    """
    for scorer_name, cfg in SCORERS.items():
        for pattern in cfg["cache_patterns"]:
            expected_filename = pattern.format(dataset_name=dataset_name)

            if filename == expected_filename:
                return scorer_name

    return None


def resolve_scorer_name(raw_name):
    """
    Converte un nome trovato in CSV/TOST/cache nel nome canonico del config.
    Ritorna None se il nome non è configurato.
    """
    raw_name = normalize_name(raw_name)

    if raw_name is None:
        return None

    if raw_name in SCORERS:
        return raw_name

    raw_low = raw_name.lower()

    for scorer_name, cfg in SCORERS.items():
        aliases = cfg.get("compare_aliases", [])

        for alias in aliases:
            alias_norm = normalize_name(alias)
            if alias_norm is None:
                continue

            if raw_name == alias_norm:
                return scorer_name

            if raw_low == alias_norm.lower():
                return scorer_name

    return None


def is_scorer_enabled(scorer_name):
    scorer_name = resolve_scorer_name(scorer_name) or scorer_name
    return SCORERS.get(scorer_name, {}).get("enabled", False)


def get_enabled_scorers():
    return [
        scorer_name
        for scorer_name, cfg in SCORERS.items()
        if cfg.get("enabled", False)
    ]


def get_all_scorers():
    return list(SCORERS.keys())


def get_label(scorer_name):
    scorer_name = resolve_scorer_name(scorer_name) or scorer_name
    return SCORERS.get(scorer_name, {}).get("label", scorer_name)


def get_color(scorer_name):
    scorer_name = resolve_scorer_name(scorer_name) or scorer_name
    return SCORERS.get(scorer_name, {}).get("color", "black")


def get_linestyle(scorer_name):
    scorer_name = resolve_scorer_name(scorer_name) or scorer_name
    return SCORERS.get(scorer_name, {}).get("linestyle", "-")


def get_marker(scorer_name):
    scorer_name = resolve_scorer_name(scorer_name) or scorer_name
    return SCORERS.get(scorer_name, {}).get("marker", "o")


def get_scorer_style(scorer_name):
    return {
        "color": get_color(scorer_name),
        "linestyle": get_linestyle(scorer_name),
        "marker": get_marker(scorer_name),
        "dashes": None,
    }


def matches_any_filter(scorer_name, filters):
    """
    Match flessibile su:
    - nome canonico
    - label
    - alias

    Esempio:
        filters=["attfus"] matcha metadata_qualt5_attfus.
    """
    if not filters:
        return False

    canonical = resolve_scorer_name(scorer_name) or scorer_name
    cfg = SCORERS.get(canonical, {})

    candidates = [
        str(canonical),
        str(cfg.get("label", canonical)),
    ]
    candidates.extend([str(a) for a in cfg.get("compare_aliases", [])])

    candidates = [c.lower() for c in candidates]

    for f in filters:
        f = str(f).lower()
        for candidate in candidates:
            if f == candidate or f in candidate:
                return True

    return False


def get_active_scorers(include=None, exclude=None, group=None, only_enabled=True):
    """
    Ritorna gli scorer da usare.

    Regole:
    - senza include: prende gli scorer enabled=True;
    - con include: prende gli scorer richiesti anche se enabled=False;
    - exclude rimuove sempre;
    - group filtra per cfg["group"].
    """
    selected = []

    for scorer_name, cfg in SCORERS.items():
        if group is not None and cfg.get("group") != group:
            continue

        if include is None:
            if only_enabled and not cfg.get("enabled", False):
                continue
        else:
            if not matches_any_filter(scorer_name, include):
                continue

        if exclude is not None and matches_any_filter(scorer_name, exclude):
            continue

        selected.append(scorer_name)

    return selected


def prepare_scores_for_roc(scorer_name, scores):
    """
    Di default assume che score più alto = documento più buono/rilevante.

    Se uno scorer ha valori più bassi migliori, imposta:
        "higher_is_better": False
    """
    scorer_name = resolve_scorer_name(scorer_name) or scorer_name
    higher_is_better = SCORERS.get(scorer_name, {}).get("higher_is_better", True)

    if higher_is_better:
        return scores

    return -scores
