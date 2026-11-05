import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PODCAST_CONFIG_FILE = PROJECT_ROOT / "config" / "podcasts.json"


def load_podcasts():
    """
    Load podcast configuration.
    """

    try:

        with open(
            PODCAST_CONFIG_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            podcasts = json.load(
                file
            )


        return podcasts


    except FileNotFoundError:

        raise FileNotFoundError(
            f"Missing configuration file: {PODCAST_CONFIG_FILE}"
        )


    except json.JSONDecodeError:

        raise ValueError(
            f"Invalid JSON in {PODCAST_CONFIG_FILE}"
        )



def get_enabled_podcasts():
    """
    Return only enabled podcasts.
    """

    podcasts = load_podcasts()


    return [
        podcast
        for podcast in podcasts
        if podcast.get(
            "enabled",
            False
        )
    ]



PODCASTS = load_podcasts()


ENABLED_PODCASTS = [
    podcast
    for podcast in PODCASTS
    if podcast.get(
        "enabled",
        False
    )
]
