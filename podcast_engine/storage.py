import json
import os

from google.cloud import storage
from google.api_core.exceptions import PreconditionFailed

from .episode_contract import (
    ensure_v3_record,
    new_episode_record,
    now_iso,
)
from .episode_identity import episode_key_for


BUCKET_NAME = os.environ.get(
    "PODCAST_GCS_BUCKET", "YOUR_GCS_BUCKET"
).removeprefix("gs://").strip("/")
STORAGE_FILE = "episodes.json"


client = None


def get_client():
    """Create the GCS client lazily so pure contract tests need no credentials."""

    global client
    if client is None:
        client = storage.Client()
    return client


def get_bucket():
    """
    Return Google Cloud Storage bucket.
    """

    return get_client().bucket(
        BUCKET_NAME
    )


def load_episodes():
    """
    Load episodes from Google Cloud Storage.
    """

    bucket = get_bucket()

    blob = bucket.blob(
        STORAGE_FILE
    )

    if not blob.exists():
        return []

    content = blob.download_as_text(
        encoding="utf-8"
    )

    return [
        ensure_v3_record(episode)
        for episode in json.loads(content)
    ]


def load_episodes_with_generation():
    """Load the global index together with its GCS generation for safe writes."""

    blob = get_bucket().blob(STORAGE_FILE)
    if not blob.exists():
        return [], 0
    content = blob.download_as_text(encoding="utf-8")
    blob.reload()
    return (
        [ensure_v3_record(episode) for episode in json.loads(content)],
        int(blob.generation),
    )


def save_episodes(episodes, if_generation_match=None):
    """
    Save episodes to Google Cloud Storage.
    """

    bucket = get_bucket()

    blob = bucket.blob(
        STORAGE_FILE
    )

    blob.upload_from_string(
        json.dumps(
            episodes,
            indent=2,
            ensure_ascii=False
        ),
        content_type="application/json",
        if_generation_match=if_generation_match,
    )


def download_gcs_bytes(gcs_path):
    """Return the exact bytes of a canonical GCS object."""

    return get_bucket().blob(gcs_path).download_as_bytes()


def upload_path_to_gcs(local_file, gcs_path):
    """Upload a file to an explicit canonical object path."""

    blob = get_bucket().blob(gcs_path)
    blob.upload_from_filename(local_file)
    print(f"Uploaded to GCS: {gcs_path}")
    return gcs_path


def download_file_from_gcs(
    gcs_path,
    local_file
):
    """
    Download file from Google Cloud Storage.
    """

    bucket = get_bucket()

    blob = bucket.blob(
        gcs_path
    )

    os.makedirs(
        os.path.dirname(local_file),
        exist_ok=True
    )

    blob.download_to_filename(
        local_file
    )

    print(
        f"Downloaded from GCS: {gcs_path}"
    )

    return local_file


def file_exists_in_gcs(
    gcs_path
):
    """
    Check if file exists in GCS.
    """

    bucket = get_bucket()

    blob = bucket.blob(
        gcs_path
    )

    return blob.exists()


def get_episode_by_key(episode_key):
    """Return an episode by its canonical feed-scoped identity."""

    for episode in load_episodes():
        if episode.get("episode_key") == episode_key:
            return ensure_v3_record(episode)
    return None


def add_episode(
    episode
):
    """
    Add new episode.
    """

    feed_url = episode.get("feed_url")
    if not feed_url:
        raise ValueError("RSS episode is missing feed_url; cannot create stable identity")
    new_episode = new_episode_record(
        episode_key=episode_key_for(feed_url, episode["guid"]),
        podcast=episode["podcast"],
        podcast_id=episode.get("podcast_id"),
        feed_url=feed_url,
        podcast_url=episode.get("podcast_url"),
        rss_guid=episode["guid"],
        title=episode["title"],
        published=episode.get("published"),
        link=episode.get("link"),
        audio_url=episode.get("audio_url"),
        category=episode.get("category"),
        prompt=episode.get("prompt"),
    )


    for _ in range(3):
        episodes, generation = load_episodes_with_generation()
        existing = next(
            (
                item
                for item in episodes
                if ensure_v3_record(item).get("episode_key")
                == new_episode["episode_key"]
            ),
            None,
        )
        if existing:
            return ensure_v3_record(existing)
        episodes.append(new_episode)
        try:
            save_episodes(episodes, if_generation_match=generation)
            return new_episode
        except PreconditionFailed:
            continue
    raise RuntimeError("episodes.json changed repeatedly while adding an episode")


def merge_episode_metadata(episode_key, source_episode):
    """Merge fresh RSS details into a record that Apple ingest created first."""

    fields = (
        "podcast",
        "podcast_id",
        "podcast_url",
        "feed_url",
        "category",
        "prompt",
        "published",
        "link",
        "audio_url",
    )
    for _ in range(3):
        episodes, generation = load_episodes_with_generation()
        updated = None
        for index, episode in enumerate(episodes):
            episode = ensure_v3_record(episode)
            episodes[index] = episode
            if episode.get("episode_key") != episode_key:
                continue
            for field in fields:
                value = source_episode.get(field)
                if value is not None:
                    episode[field] = value
            episode["updated_at"] = now_iso()
            updated = episode
            break
        if updated is None:
            return None
        try:
            save_episodes(episodes, if_generation_match=generation)
            return updated
        except PreconditionFailed:
            continue
    raise RuntimeError("episodes.json changed repeatedly while merging RSS metadata")



RETRANSCRIBABLE_COMPILER_STATES = frozenset({"pending", "blocked", "ready", "review_required"})


def request_whisper_retranscription(episode_key):
    """Mark one unfinished episode's Whisper source pending again (TASK-125).

    The next Worker run then re-transcribes with the current producer
    (artifacts from another producer are never reused) and recompiles, which
    replaces the pending Human Review queue. Only the episode's
    ``status.whisper`` changes; completed episodes are refused because the
    Worker does not resume them. The write is generation-conditional.
    """

    for _ in range(3):
        episodes, generation = load_episodes_with_generation()
        matches = [
            index for index, episode in enumerate(episodes)
            if episode.get("episode_key") == episode_key
        ]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one episode {episode_key}, found {len(matches)}")
        episode = episodes[matches[0]]
        compiler_state = episode["status"]["compiler"].get("state")
        if compiler_state not in RETRANSCRIBABLE_COMPILER_STATES:
            raise ValueError(
                f"Episode {episode_key} has compiler state {compiler_state!r}; "
                "only unfinished episodes can be re-transcribed"
            )
        episode["status"]["whisper"] = {"state": "pending", "updated_at": now_iso()}
        episode["updated_at"] = now_iso()
        try:
            save_episodes(episodes, if_generation_match=generation)
            return episode
        except PreconditionFailed:
            continue
    raise RuntimeError("episodes.json changed repeatedly while requesting re-transcription")


def update_episode(
    episode_key,
    download_completed="KEEP",
    transcript_file="KEEP",
    whisper_metadata_file="KEEP",
    compiled_transcript_file="KEEP",
    compiler_report_file="KEEP",
    compiler_review_required="KEEP",
    summary_body_file="KEEP",
    summary_metadata_file="KEEP",
    markdown_file="KEEP",
    apple_transcript_late_notified_at="KEEP",
):
    """
    Update episode state by its canonical episode key.
    """

    for _ in range(3):
        episodes, generation = load_episodes_with_generation()
        updated = None
        for index, episode in enumerate(episodes):

            episode = ensure_v3_record(episode)
            episodes[index] = episode
            if episode.get("episode_key") == episode_key:


                if download_completed != "KEEP":
                    episode["status"]["download"] = {
                        "state": "completed" if download_completed else "pending",
                        "updated_at": now_iso(),
                    }



                if transcript_file != "KEEP":

                    if transcript_file:
                        episode["status"]["whisper"] = {"state": "ready", "updated_at": now_iso()}
                        episode["files"]["sources"]["whisper"]["text"] = transcript_file
                        episode["status"]["compiler"] = {
                            "state": "ready"
                            if episode["files"]["sources"]["apple"].get("text")
                            else "blocked",
                            "updated_at": now_iso(),
                        }

                if whisper_metadata_file != "KEEP":

                    episode["files"]["sources"]["whisper"]["metadata"] = whisper_metadata_file

                if compiled_transcript_file != "KEEP":

                    episode["files"]["compiled"]["transcript"] = compiled_transcript_file

                if compiler_report_file != "KEEP":

                    episode["files"]["compiled"]["report"] = compiler_report_file

                if compiler_review_required != "KEEP":

                    episode["status"]["compiler"] = {
                        "state": "review_required" if compiler_review_required else "completed",
                        "updated_at": now_iso(),
                        "review_required": compiler_review_required,
                    }



                if summary_body_file != "KEEP":
                    episode["files"]["summary"]["body"] = summary_body_file

                if summary_metadata_file != "KEEP":
                    episode["files"]["summary"]["metadata"] = summary_metadata_file

                if markdown_file != "KEEP":

                    if markdown_file:
                        episode["status"]["summary"] = {"state": "ready", "updated_at": now_iso()}
                        episode["files"]["summary"]["markdown"] = markdown_file

                if apple_transcript_late_notified_at != "KEEP":

                    episode[
                        "apple_transcript_late_notified_at"
                    ] = apple_transcript_late_notified_at



                updated = episode
                episode["updated_at"] = now_iso()
                break
        if updated is None:
            return None
        try:
            save_episodes(episodes, if_generation_match=generation)
            return updated
        except PreconditionFailed:
            continue
    raise RuntimeError("episodes.json changed repeatedly while updating an episode")
