"""
Configuration Management for Indian AI Trader
Centralized settings with environment variable support
"""
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class GrowwAPISettings(BaseSettings):
    """Groww API Configuration"""
    api_key: str = Field(default="", alias="GROWW_API_KEY")
    api_secret: str = Field(default="", alias="GROWW_API_SECRET")
    access_token: str = Field(default="", alias="GROWW_ACCESS_TOKEN")
    base_url: str = Field(default="https://api.groww.in", alias="GROWW_API_BASE_URL")
    api_version: str = Field(default="v1", alias="GROWW_API_VERSION")
    
    # Endpoints
    instruments_csv_url: str = "https://growwapi-assets.groww.in/instruments/instrument.csv"
    
    @property
    def quote_url(self) -> str:
        return f"{self.base_url}/{self.api_version}/live-data/quote"
    
    @property
    def ltp_url(self) -> str:
        return f"{self.base_url}/{self.api_version}/live-data/ltp"
    
    @property
    def historical_candles_url(self) -> str:
        return f"{self.base_url}/{self.api_version}/historical/candles"


class DatabaseSettings(BaseSettings):
    """Database Configuration"""
    # PostgreSQL
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_db: str = Field(default="indian_ai_trader", alias="POSTGRES_DB")
    postgres_user: str = Field(default="trader", alias="POSTGRES_USER")
    postgres_password: str = Field(default="", alias="POSTGRES_PASSWORD")
    
    @property
    def postgres_url(self) -> str:
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    @property
    def async_postgres_url(self) -> str:
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
    
    # ClickHouse
    clickhouse_host: str = Field(default="localhost", alias="CLICKHOUSE_HOST")
    clickhouse_port: int = Field(default=9000, alias="CLICKHOUSE_PORT")
    clickhouse_http_port: int = Field(default=8123, alias="CLICKHOUSE_HTTP_PORT")
    clickhouse_db: str = Field(default="market_data", alias="CLICKHOUSE_DB")
    clickhouse_user: str = Field(default="default", alias="CLICKHOUSE_USER")
    clickhouse_password: str = Field(default="", alias="CLICKHOUSE_PASSWORD")
    
    # Redis
    redis_host: str = Field(default="localhost", alias="REDIS_HOST")
    redis_port: int = Field(default=6379, alias="REDIS_PORT")
    redis_password: str = Field(default="", alias="REDIS_PASSWORD")
    redis_db: int = Field(default=0, alias="REDIS_DB")
    
    @property
    def redis_url(self) -> str:
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"


class KafkaSettings(BaseSettings):
    """Kafka/Redpanda Configuration"""
    bootstrap_servers: str = Field(default="localhost:9092", alias="KAFKA_BOOTSTRAP_SERVERS")
    security_protocol: str = Field(default="PLAINTEXT", alias="KAFKA_SECURITY_PROTOCOL")
    consumer_group: str = Field(default="indian-ai-trader", alias="KAFKA_CONSUMER_GROUP")
    
    # Topic names
    topic_ticks_raw: str = "market.ticks.raw"
    topic_ticks_normalized: str = "market.ticks.normalized"
    topic_candles_1m: str = "market.candles.1m"
    topic_candles_5m: str = "market.candles.5m"
    topic_candles_15m: str = "market.candles.15m"
    topic_signals: str = "signals.generated"
    topic_orders: str = "orders.submitted"
    topic_audits: str = "audits.packets"


class SecuritySettings(BaseSettings):
    """Security Configuration"""
    jwt_secret_key: str = Field(default="your-secret-key-change-in-production", alias="JWT_SECRET_KEY")
    jwt_algorithm: str = Field(default="HS256", alias="JWT_ALGORITHM")
    jwt_expire_minutes: int = Field(default=30, alias="JWT_ACCESS_TOKEN_EXPIRE_MINUTES")
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60, alias="RATE_LIMIT_REQUESTS_PER_MINUTE")
    rate_limit_burst: int = Field(default=10, alias="RATE_LIMIT_BURST")


class TradingSettings(BaseSettings):
    """Trading Configuration"""
    enable_paper_trading: bool = Field(default=True, alias="ENABLE_PAPER_TRADING")
    enable_live_trading: bool = Field(default=False, alias="ENABLE_LIVE_TRADING")
    enable_signal_service: bool = Field(default=True, alias="ENABLE_SIGNAL_SERVICE")
    
    # Risk Parameters
    default_watchlist: str = Field(default="NIFTY50", alias="DEFAULT_WATCHLIST")
    max_positions: int = Field(default=10, alias="MAX_POSITIONS")
    max_position_size_percent: float = Field(default=5.0, alias="MAX_POSITION_SIZE_PERCENT")
    default_stop_loss_percent: float = Field(default=2.0, alias="DEFAULT_STOP_LOSS_PERCENT")
    max_daily_trades: int = Field(default=20, alias="MAX_DAILY_TRADES")


class ServicePorts(BaseSettings):
    """Service Port Configuration"""
    api_gateway: int = Field(default=8000, alias="API_GATEWAY_PORT")
    instrument_service: int = Field(default=8001, alias="INSTRUMENT_SERVICE_PORT")
    market_ingestor: int = Field(default=8002, alias="MARKET_INGESTOR_PORT")
    bar_aggregator: int = Field(default=8003, alias="BAR_AGGREGATOR_PORT")
    feature_service: int = Field(default=8004, alias="FEATURE_SERVICE_PORT")
    signal_service: int = Field(default=8005, alias="SIGNAL_SERVICE_PORT")
    backtest_service: int = Field(default=8006, alias="BACKTEST_SERVICE_PORT")
    portfolio_service: int = Field(default=8007, alias="PORTFOLIO_SERVICE_PORT")
    order_service: int = Field(default=8008, alias="ORDER_SERVICE_PORT")
    audit_service: int = Field(default=8009, alias="AUDIT_SERVICE_PORT")
    notification_service: int = Field(default=8010, alias="NOTIFICATION_SERVICE_PORT")


class Settings(BaseSettings):
    """Main Settings aggregating all configurations"""
    # Environment
    debug: bool = Field(default=False, alias="DEBUG")
    reload: bool = Field(default=True, alias="RELOAD")
    environment: str = Field(default="development", alias="ENVIRONMENT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    
    # Sub-configurations
    groww: GrowwAPISettings = GrowwAPISettings()
    database: DatabaseSettings = DatabaseSettings()
    kafka: KafkaSettings = KafkaSettings()
    security: SecuritySettings = SecuritySettings()
    trading: TradingSettings = TradingSettings()
    ports: ServicePorts = ServicePorts()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
