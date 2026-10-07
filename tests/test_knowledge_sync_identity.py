import unittest
from unittest.mock import Mock, patch

from macos_agent.knowledge_sync import (
    BUCKET_NAME,
    CLOUD_PLATFORM_SCOPE,
    IMPERSONATE_ENV,
    knowledge_bucket_from_env,
)


class KnowledgeSyncIdentityTests(unittest.TestCase):
    def test_no_impersonation_configuration_keeps_existing_adc_path(self):
        with patch("macos_agent.knowledge_sync.google.auth.default") as default:
            bucket = knowledge_bucket_from_env({})

        self.assertIsNone(bucket)
        default.assert_not_called()

    def test_dedicated_identity_uses_user_adc_only_as_impersonation_source(self):
        source_credentials = object()
        impersonated = object()
        bucket = object()
        client = Mock()
        client.bucket.return_value = bucket

        env = {
            IMPERSONATE_ENV:
                "podcast-knowledge-sync-runtime@YOUR_GCP_PROJECT_ID.iam.gserviceaccount.com",
            "GOOGLE_CLOUD_PROJECT": "YOUR_GCP_PROJECT_ID",
        }

        with (
            patch(
                "macos_agent.knowledge_sync.google.auth.default",
                return_value=(source_credentials, "detected-project"),
            ) as default,
            patch(
                "macos_agent.knowledge_sync.impersonated_credentials.Credentials",
                return_value=impersonated,
            ) as credentials,
            patch(
                "macos_agent.knowledge_sync.storage.Client",
                return_value=client,
            ) as storage_client,
        ):
            result = knowledge_bucket_from_env(env)

        self.assertIs(result, bucket)

        default.assert_called_once_with(
            scopes=[CLOUD_PLATFORM_SCOPE],
        )

        credentials.assert_called_once_with(
            source_credentials=source_credentials,
            target_principal=env[IMPERSONATE_ENV],
            target_scopes=[CLOUD_PLATFORM_SCOPE],
        )

        storage_client.assert_called_once_with(
            project="YOUR_GCP_PROJECT_ID",
            credentials=impersonated,
        )

        client.bucket.assert_called_once_with(BUCKET_NAME)

    def test_impersonation_failure_does_not_fall_back_to_user_identity(self):
        env = {
            IMPERSONATE_ENV:
                "podcast-knowledge-sync-runtime@YOUR_GCP_PROJECT_ID.iam.gserviceaccount.com",
        }

        with patch(
            "macos_agent.knowledge_sync.google.auth.default",
            side_effect=RuntimeError("impersonation source unavailable"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "impersonation source unavailable",
            ):
                knowledge_bucket_from_env(env)


if __name__ == "__main__":
    unittest.main()
