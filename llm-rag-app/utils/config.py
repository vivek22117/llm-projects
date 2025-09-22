import json
from typing import Any

import boto3
from botocore.exceptions import ClientError
from pydantic import (
    AnyHttpUrl,
    PostgresDsn,
    field_validator,
    model_validator,
)
from pydantic_core.core_schema import ValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file="../.env", env_file_encoding="utf-8")

    # --- Basic Project Config
    PROJECT_NAME: str = "LLM RAG API"
    BACKEND_CORS_ORIGINS: list[AnyHttpUrl] = []
    ENV_NAME: str = "local"

    # --- AWS Secrets Manager Config (loaded from environment)
    AWS_SECRET_NAME: str | None = None
    AWS_REGION: str = "us-east-1"  # Or your default region

    # --- Database Config (will be populated from AWS Secrets Manager)
    POSTGRES_SERVER: str | None = None
    POSTGRES_USER: str | None = None
    POSTGRES_PASSWORD: str | None = None
    POSTGRES_DB: str | None = None
    SQLALCHEMY_DATABASE_URI: PostgresDsn | None = None

    @model_validator(mode='after')
    def load_secrets_from_aws(self) -> 'Settings':
        """
        If AWS_SECRET_NAME is set, fetch secrets from AWS Secrets Manager
        and populate the corresponding fields.
        """
        if self.AWS_SECRET_NAME:
            session = boto3.session.Session()
            client = session.client(
                service_name='secretsmanager',
                region_name=self.AWS_REGION
            )
            try:
                get_secret_value_response = client.get_secret_value(
                    SecretId=self.AWS_SECRET_NAME
                )
            except ClientError as e:
                # You can add more specific error handling here
                raise ValueError(f"Couldn't retrieve secret from AWS Secrets Manager: {e}")
            else:
                # Secrets Manager stores secrets as a JSON string
                secret = json.loads(get_secret_value_response['SecretString'])

                # Populate the model fields from the secret's keys
                self.POSTGRES_SERVER = secret.get('host')
                self.POSTGRES_USER = secret.get('username')
                self.POSTGRES_PASSWORD = secret.get('password')
                self.POSTGRES_DB = secret.get('dbname')

        return self

    @field_validator("SQLALCHEMY_DATABASE_URI")
    @classmethod
    def assemble_db_connection(cls, v: str | None, info: ValidationInfo) -> Any:
        """
        Assembles the database connection string after other fields are populated.
        """
        if isinstance(v, str):
            return v

        # Check if all required DB components are present
        if not all([
            info.data.get("POSTGRES_SERVER"),
            info.data.get("POSTGRES_USER"),
            info.data.get("POSTGRES_PASSWORD"),
            info.data.get("POSTGRES_DB")
        ]):
            # If not all components are there (e.g., secrets fetch failed),
            # this will prevent a cryptic error later on.
            return None

        postgres_dsn = PostgresDsn.build(
            scheme="postgresql+asyncpg",
            username=info.data.get("POSTGRES_USER"),
            password=info.data.get("POSTGRES_PASSWORD"),
            host=info.data.get("POSTGRES_SERVER"),
            path=f"{info.data.get('POSTGRES_DB') or ''}",
        )
        return str(postgres_dsn)


settings = Settings()