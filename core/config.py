from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):

    # Load contents of .env file
    model_config = SettingsConfigDict(
        env_file=".env",
        env_ignore_empty=True
    )

settings = Settings()