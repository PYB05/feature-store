"""
============================================================
ML Feature Store — Kafka Producer
============================================================
Reads the creditcard.csv dataset row by row and streams each
transaction as a JSON message to a Kafka topic called 'transactions'.

What this does:
    1. Connects to Kafka broker (host from KAFKA_BROKER env var)
    2. Reads creditcard.csv row by row
    3. Attaches a unique UUID as transaction_id to each row
    4. Publishes each transaction as JSON to topic 'transactions'
    5. Streams at 10 transactions/second (configurable via STREAM_RATE)
    6. Prints a confirmation every 100 messages sent
    7. Retries up to 3 times on connection failure (5s wait between)

Environment Variables:
    KAFKA_BROKER  — Kafka broker address (default: localhost:9092)
    STREAM_RATE   — Transactions per second (default: 10)
    DATA_PATH     — Path to creditcard.csv (default: data/creditcard.csv)

Usage:
    python src/producer.py
============================================================
"""

import os
import sys
import time
import uuid
import json
import logging

import pandas as pd
from kafka import KafkaProducer
from kafka.errors import KafkaError, NoBrokersAvailable

# ============================================================
# Configure Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PRODUCER] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================
# Constants (overridable via environment variables)
# ============================================================
KAFKA_BROKER  = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC_NAME    = os.getenv("KAFKA_TOPIC",  "transactions")
DATA_PATH     = os.getenv("DATA_PATH",    "data/creditcard.csv")
STREAM_RATE   = max(1, int(os.getenv("STREAM_RATE", "10")))  # transactions per second (min 1 to avoid ZeroDivisionError)
SLEEP_INTERVAL = 1.0 / STREAM_RATE                          # seconds between messages


# ============================================================
# Connect to Kafka (with retry logic)
# ============================================================
def create_producer(max_retries: int = 3, retry_delay: int = 5) -> KafkaProducer:
    """
    Create and return a KafkaProducer instance.

    Retries up to max_retries times because Kafka might not be
    fully ready when this script first runs (common in Docker).

    Args:
        max_retries:  Number of connection attempts before giving up
        retry_delay:  Seconds to wait between attempts

    Returns:
        A connected KafkaProducer object
    """
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                f"Connecting to Kafka at {KAFKA_BROKER} "
                f"(attempt {attempt}/{max_retries})..."
            )

            producer = KafkaProducer(
                bootstrap_servers=KAFKA_BROKER,

                # Serialize Python dicts → JSON bytes automatically
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),

                # Retry failed sends up to 3 times internally
                retries=3,

                # Wait up to 30s for broker acknowledgement
                request_timeout_ms=30000,

                # Require at least the leader broker to acknowledge writes
                acks="all",

                # Batch messages to improve throughput
                linger_ms=5,
                batch_size=16384,
            )

            # BUG FIX: KafkaProducer connects lazily — the constructor succeeds
            # even if Kafka is down. We must probe the broker explicitly to
            # confirm the connection is real before returning.
            producer.partitions_for(TOPIC_NAME)  # Triggers actual broker contact

            logger.info(f"Connected to Kafka broker at {KAFKA_BROKER}")
            return producer

        except NoBrokersAvailable:
            # Raised by partitions_for() when broker is truly unreachable
            if attempt < max_retries:
                logger.warning(
                    f"Kafka broker not available yet. "
                    f"Retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
            else:
                logger.error(
                    f"Failed to connect to Kafka after {max_retries} attempts. "
                    f"Is Kafka running at {KAFKA_BROKER}?"
                )
                sys.exit(1)

        except KafkaError as e:
            # Other Kafka-level errors (auth, protocol mismatch, etc.)
            if attempt < max_retries:
                logger.warning(f"Kafka error (attempt {attempt}/{max_retries}): {e}. Retrying...")
                time.sleep(retry_delay)
            else:
                logger.error(f"Kafka error after {max_retries} attempts: {e}")
                sys.exit(1)

        except Exception as e:
            # Catch-all for unexpected errors during connection
            logger.error(f"Unexpected error connecting to Kafka: {e}")
            sys.exit(1)


# ============================================================
# Load Dataset
# ============================================================
def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the creditcard.csv dataset.

    Args:
        filepath: Path to the CSV file

    Returns:
        DataFrame with all 284,807 transactions
    """
    if not os.path.exists(filepath):
        logger.error(f"Dataset not found at: {filepath}")
        logger.error("   Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        sys.exit(1)

    logger.info(f"Loading dataset from {filepath}...")
    df = pd.read_csv(filepath)

    logger.info(f"   Loaded {len(df):,} transactions")
    logger.info(f"   Fraud:     {df['Class'].sum():,} ({df['Class'].mean() * 100:.3f}%)")
    logger.info(f"   Legitimate:{(df['Class'] == 0).sum():,}")
    return df


# ============================================================
# Delivery Callback
# ============================================================
def on_send_success(record_metadata):
    """Called when a message is successfully delivered to Kafka."""
    # Uncomment for verbose logging (slows things down a bit):
    # logger.debug(
    #     f"Delivered → topic={record_metadata.topic} "
    #     f"partition={record_metadata.partition} "
    #     f"offset={record_metadata.offset}"
    # )
    pass


def on_send_error(exc):
    """Called when a message fails to deliver."""
    logger.error(f"Message delivery failed: {exc}")


# ============================================================
# Main Streaming Loop
# ============================================================
def stream_transactions(producer: KafkaProducer, df: pd.DataFrame):
    """
    Stream all rows of the DataFrame to Kafka, one by one.

    Each row is enriched with a unique UUID transaction_id before
    publishing. The loop sleeps SLEEP_INTERVAL seconds between
    messages to simulate a real-time feed at STREAM_RATE TPS.

    Args:
        producer:  Connected KafkaProducer
        df:        Full dataset as a DataFrame
    """
    total = len(df)
    sent  = 0
    start_time = time.time()

    logger.info(f"\n{'='*55}")
    logger.info(f"Starting stream: {total:,} transactions at {STREAM_RATE} TPS")
    logger.info(f"   Topic: {TOPIC_NAME}")
    logger.info(f"   Estimated time: {total / STREAM_RATE / 60:.1f} minutes")
    logger.info(f"{'='*55}\n")

    for idx, row in df.iterrows():
        # ── Build the message payload ──────────────────────────
        # Convert the row to a plain Python dict.
        # Add a UUID transaction_id so the API can use it as a Redis key.

        # BUG FIX: Skip rows with NaN values in critical columns.
        # NaN values cause json.dumps() to produce invalid JSON (NaN is not
        # valid JSON — it silently produces 'NaN' which breaks the consumer).
        critical_cols = ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]
        if row[critical_cols].isnull().any():
            logger.warning(f"Skipping row {idx} -- contains NaN values")
            continue

        message = {
            "transaction_id": str(uuid.uuid4()),   # unique per message
            "time":   float(row["Time"]),
            "amount": float(row["Amount"]),

            # V1 through V28 — PCA-transformed features
            **{
                f"v{i}": float(row[f"V{i}"])
                for i in range(1, 29)
            },

            # Ground-truth label — useful for evaluation but NOT sent to model
            "actual_class": int(row["Class"]),
        }

        # ── Publish to Kafka ───────────────────────────────────
        try:
            producer.send(TOPIC_NAME, value=message) \
                    .add_callback(on_send_success) \
                    .add_errback(on_send_error)
        except KafkaError as e:
            logger.error(f"Send error at row {idx}: {e}")
            continue

        sent += 1

        # ── Print progress every 100 messages ─────────────────
        if sent % 100 == 0:
            elapsed   = time.time() - start_time
            actual_tps = sent / elapsed if elapsed > 0 else 0
            pct        = (sent / total) * 100

            logger.info(
                f"Sent {sent:>7,}/{total:,}  ({pct:5.1f}%)  "
                f"| Actual TPS: {actual_tps:5.1f}  "
                f"| Elapsed: {elapsed:6.1f}s"
            )

        # ── Rate limiting — sleep to hit target TPS ────────────
        time.sleep(SLEEP_INTERVAL)

    # ── Final flush ────────────────────────────────────────────
    # Flush ensures all buffered messages are sent before we exit
    logger.info("\nFlushing remaining messages...")
    producer.flush()

    total_time = time.time() - start_time
    actual_tps = sent / total_time if total_time > 0 else 0

    logger.info(f"\n{'='*55}")
    logger.info(f"Streaming complete!")
    logger.info(f"   Sent:       {sent:,} messages")
    logger.info(f"   Duration:   {total_time:.1f} seconds ({total_time/60:.1f} min)")
    logger.info(f"   Actual TPS: {actual_tps:.2f}")
    logger.info(f"{'='*55}")


# ============================================================
# Entry Point
# ============================================================
def main():
    logger.info("\n" + "="*55)
    logger.info("ML Feature Store -- Kafka Transaction Producer")
    logger.info("="*55)

    # Step 1: Load the dataset
    df = load_dataset(DATA_PATH)

    # Step 2: Connect to Kafka (with retry)
    producer = create_producer(max_retries=3, retry_delay=5)

    # Step 3: Stream all transactions
    try:
        stream_transactions(producer, df)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user. Flushing and shutting down...")
        producer.flush()
    finally:
        producer.close()
        logger.info("Producer shut down cleanly.")


if __name__ == "__main__":
    main()
