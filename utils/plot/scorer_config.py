SCORERS = {
    "finetuned_qualt5": {
        "enabled": True,
        "group": "qualt5",
        "cache_patterns": [
            "finetuned_qualt5_{dataset_name}.cache",
            "finetunedqualt5_{dataset_name}.cache",
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
            "metadata_qualt5_{dataset_name}_CONCAT-V2-FA.cache",
        ],
        "label": "Metadata-QualT5-CONCAT-V2",
        "color": "#01FF16",
        "linestyle": "--",
        "marker": "s",
        "higher_is_better": True,
    },
    
    "metadata_qualt5_allmetapj": {
        "enabled": False,
        "group": "metadata_qualt5",
        "cache_patterns": [
            "metadata_qualt5_{dataset_name}_ALLMETAPJ.cache",
        ],
        "label": "Metadata-QualT5-ALLMETAPJ",
        "color": "#FF0105",
        "linestyle": "--",
        "marker": "x",
        "higher_is_better": True,
    },
    
    "metadata_qualt5_attfus": {
        "enabled": False,
        "group": "metadata_qualt5",
        "cache_patterns": [
            "metadata_qualt5_{dataset_name}_ATTFUS.cache",
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
        "label": "Metadata-QualT5 CONCAT old",
        "color": "#8c564b",
        "linestyle": "--",
        "marker": "x",
        "higher_is_better": True,
    },

    # Esempi di scorer non metadata.
    # Attivali solo quando ti servono.
    "tasb": {
        "enabled": False,
        "group": "baseline",
        "cache_patterns": [
            "tasb_{dataset_name}.cache",
        ],
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
        "label": "T5-Ppl",
        "color": "#e377c2",
        "linestyle": "--",
        "marker": "v",
        # Se per perplexity valori più bassi sono migliori,
        # lasciando False viene invertito automaticamente nei plot ROC.
        "higher_is_better": False,
    },

    "itn": {
        "enabled": False,
        "group": "baseline",
        "cache_patterns": [
            "itn_{dataset_name}.cache",
        ],
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

    value = value.strip()

    if value == "":
        return None

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def get_scorer_name_from_cache(filename, dataset_name):
    """
    Risolve il nome dello scorer usando solo scorer_config.py.

    Esempio:
    metadata_qualt5_msmarco_passage_CONCAT-V2-FA.cache
        -> metadata_qualt5_concat_v2_fa
    """

    for scorer_name, cfg in SCORERS.items():
        for pattern in cfg["cache_patterns"]:
            expected_filename = pattern.format(dataset_name=dataset_name)

            if filename == expected_filename:
                return scorer_name

    return None


def is_scorer_enabled(scorer_name):
    return SCORERS.get(scorer_name, {}).get("enabled", False)


def get_enabled_scorers():
    return [
        scorer_name
        for scorer_name, cfg in SCORERS.items()
        if cfg.get("enabled", False)
    ]


def get_label(scorer_name):
    return SCORERS.get(scorer_name, {}).get("label", scorer_name)


def get_color(scorer_name):
    return SCORERS.get(scorer_name, {}).get("color", "black")


def get_linestyle(scorer_name):
    return SCORERS.get(scorer_name, {}).get("linestyle", "-")


def get_marker(scorer_name):
    return SCORERS.get(scorer_name, {}).get("marker", None)

def prepare_scores_for_roc(scorer_name, scores):
    """
    Di default assume che score più alto = documento più buono/rilevante.

    Se in futuro hai uno scorer tipo perplexity dove score più basso = meglio,
    nel config puoi aggiungere:
        "higher_is_better": False
    e qui verrà invertito automaticamente.
    """

    higher_is_better = SCORERS.get(scorer_name, {}).get("higher_is_better", True)

    if higher_is_better:
        return scores

    return -scores