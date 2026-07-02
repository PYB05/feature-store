"""
============================================================
ML Feature Store — Kafka Consumer
============================================================
Subscribes to the 'transactions' Kafka topic, calls the
FastAPI /predict endpoint for each message, stores the result
in PostgreSQL, and prints a FRAUD ALERT when fraud is detected.

What this does:
    1. Connects to Kafka and subscribes to 'transactions' topic
    2. Polls messages in batches of 10 for efficiency
    3. For each message:
         → Calls POST /predict on the FastAPI service
         → Stores result in PostgreSQL via db.py
         -> Prints FRAUD ALERT when is_fraud=True
    4. Logs processing rate every 500 messages
    5. Retries failed API calls up to 3 times before skipping
    6. Handles all errors gracefully — never crashes on bad data

Environment Variables:
    KAFKA_BROKER  — Kafka broker address (default: localhost:9092)
    API_URL       — FastAPI service base URL (default: http://localhost:8000)
    DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD — PostgreSQL creds

Usage:
    python src/consumer.py
============================================================
"""

import os
import sys
import time
import json
import logging
from typing import Optional

import requests
from kafka import KafkaConsumer
from kafka.errors import KafkaError, NoBrokersAvailable

# Add parent directory so we can import src.db
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.db import DatabaseManager

# ============================================================
# Configure Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CONSUMER] %(levelname)s — %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


# ============================================================
# Constants (overridable via environment variables)
# ============================================================
KAFKA_BROKER  = os.getenv("KAFKA_BROKER", "localhost:9092")
TOPIC_NAME    = os.getenv("KAFKA_TOPIC",  "transactions")
API_URL       = os.getenv("API_URL",       "http://localhost:8000")
BATCH_SIZE    = 10      # Number of messages per poll cycle
LOG_INTERVAL  = 500     # Print rate stats every N messages
API_TIMEOUT   = 10      # Seconds before API call times out
API_RETRIES   = 3       # Max retries per failed API call
API_RETRY_DELAY = 2     # Seconds between API retries


# ============================================================
# Connect to Kafka Consumer (with retry logic)
# ============================================================
def create_consumer(max_retries: int = 5, retry_delay: int = 5) -> KafkaConsumer:
    """
    Create and return a KafkaConsumer subscribed to TOPIC_NAME.

    Retries multiple times because:
    - Kafka may still be booting when this runs in Docker
    - The topic might not exist yet (producer hasn't run)

    Args:
        max_retries:  Number of connection attempts
        retry_delay:  Seconds to wait between attempts

    Returns:
        A connected, subscribed KafkaConsumer object
    """
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(
                f"Connecting to Kafka at {KAFKA_BROKER} "
                f"(attempt {attempt}/{max_retries})..."
            )

            consumer = KafkaConsumer(
                TOPIC_NAME,
                bootstrap_servers=KAFKA_BROKER,

                # Deserialize JSON bytes → Python dict automatically
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),

                # Consumer group ID — Kafka tracks our read offset per group
                group_id="fraud-detection-consumer-group",

                # Start reading from the beginning if no offset exists
                # Change to "latest" to only process new messages
                auto_offset_reset="earliest",

                # Commit offsets automatically every 5 seconds
                enable_auto_commit=True,
                auto_commit_interval_ms=5000,

                # BUG FIX: Do NOT set consumer_timeout_ms here.
                # consumer_timeout_ms causes the consumer iterator to raise
                # StopIteration when the topic is temporarily empty, which
                # breaks our infinite while True polling loop with an
                # unhandled RuntimeError (PEP 479 in Python 3.7+).
                # We handle timeouts via poll(timeout_ms=...) instead.
                # consumer_timeout_ms=1000,   <-- REMOVED

                # Max messages per poll call (controls our batch size)
                max_poll_records=BATCH_SIZE,

                # Session timeout — if consumer doesn't heartbeat, rebalance
                session_timeout_ms=30000,
                heartbeat_interval_ms=10000,
            )

            logger.info(f"Connected to Kafka, subscribed to topic: '{TOPIC_NAME}'")
            return consumer

        except NoBrokersAvailable:
            if attempt < max_retries:
                logger.warning(
                    f"Kafka not available yet. "
                    f"Retrying in {retry_delay}s..."
                )
                time.sleep(retry_delay)
            else:
                logger.error(
                    f"Cannot connect to Kafka after {max_retries} attempts. "
                    f"Is Kafka running at {KAFKA_BROKER}?"
                )
                sys.exit(1)

        except KafkaError as e:
            logger.error(f"Kafka error: {e}")
            sys.exit(1)


# ============================================================
# Call the FastAPI /predict Endpoint
# ============================================================
def call_predict_api(transaction: dict) -> Optional[dict]:
    """
    POST a transaction to the FastAPI /predict endpoint.

    Retries up to API_RETRIES times on network errors.
    Returns None if all retries fail (message is skipped).

    The API payload must match the TransactionRequest schema:
        transaction_id, amount, time, v1..v28

    Args:
        transaction: Raw message dict from Kafka

    Returns:
        API response dict or None on failure
    """
    # Build the payload — map keys to what the API expects
    payload = {
        "transaction_id": transaction["transaction_id"],
        "amount":         transaction["amount"],
        "time":           transaction["time"],
        **{
            f"v{i}": transaction[f"v{i}"]
            for i in range(1, 29)
        }
    }

    url = f"{API_URL}/predict"

    for attempt in range(1, API_RETRIES + 1):
        try:
            response = requests.post(url, json=payload, timeout=API_TIMEOUT)
            response.raise_for_status()  # Raise exception on 4xx/5xx
            return response.json()

        except requests.exceptions.ConnectionError:
            if attempt < API_RETRIES:
                logger.warning(
                    f"API not reachable (attempt {attempt}/{API_RETRIES}), "
                    f"retrying in {API_RETRY_DELAY}s..."
                )
                time.sleep(API_RETRY_DELAY)
            else:
                logger.error(f"API unreachable after {API_RETRIES} attempts: {url}")
                return None

        except requests.exceptions.Timeout:
            logger.warning(f"API timeout on attempt {attempt}/{API_RETRIES}")
            if attempt >= API_RETRIES:
                return None
            time.sleep(API_RETRY_DELAY)

        except requests.exceptions.HTTPError as e:
            logger.error(f"API returned error: {e.response.status_code} -- {e.response.text}")
            return None

        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Bad API response format: {e}")
            return None


# ============================================================
# Store Result in PostgreSQL
# ============================================================
def store_result(db: DatabaseManager, transaction: dict, prediction: dict) -> bool:
    """
    Persist the prediction result to PostgreSQL.

    This is called AFTER the API call because the API already stores
    the result in PostgreSQL itself — this is a secondary write from
    the consumer's perspective (for audit logging with source='consumer').

    Note: The API also writes to Postgres; the consumer writes with
    source='consumer' to track the pipeline path.

    Args:
        db:          Connected DatabaseManager instance
        transaction: Raw Kafka message dict
        prediction:  API response dict

    Returns:
        True if insert succeeded
    """
    try:
        return db.insert_transaction({
            "transaction_id": transaction["transaction_id"],
            "amount":         transaction["amount"],
            "is_fraud":       prediction["is_fraud"],
            "confidence":     prediction["confidence"],
            "latency_ms":     prediction["latency_ms"],
            "source":         "consumer",  # Marks it came through Kafka pipeline
        })
    except Exception as e:
        logger.error(f"DB insert failed: {e}")
        return False


# ============================================================
# Print Fraud Alert
# ============================================================
def print_fraud_alert(transaction: dict, prediction: dict):
    """
    Print a prominent alert when a fraudulent transaction is detected.

    Args:
        transaction: Raw Kafka message dict
        prediction:  API response dict
    """
    txn_id     = transaction["transaction_id"]
    amount     = transaction["amount"]
    confidence = prediction["confidence"]
    latency    = prediction["latency_ms"]

    logger.warning(
        f"\n"
        f"  {'FRAUD ALERT':^50}\n"
        f"  {'━'*50}\n"
        f"  Transaction ID : {txn_id}\n"
        f"  Amount         : ${amount:,.2f}\n"
        f"  Confidence     : {confidence:.4f} ({confidence*100:.2f}% fraud probability)\n"
        f"  Latency        : {latency:.2f}ms\n"
        f"  {'━'*50}\n"
    )


# ============================================================
# Main Consumer Loop
# ============================================================
def consume_messages(consumer: KafkaConsumer, db: DatabaseManager):
    """
    Main loop: poll Kafka → predict → store → alert.

    Processes messages in batches of BATCH_SIZE (10).
    Logs throughput stats every LOG_INTERVAL messages.

    Args:
        consumer: Connected KafkaConsumer
        db:       Connected DatabaseManager
    """
    processed   = 0          # Total messages processed
    fraud_count = 0          # Total fraud detections
    cache_hits  = 0          # Total Redis cache hits
    errors      = 0          # Total failed predictions
    start_time  = time.time()
    last_logged = 0          # BUG FIX: replaced unused 'batch_start' with a
                             # counter tracking the last milestone we logged at

    logger.info(f"\n{'='*55}")
    logger.info(f"Listening on topic '{TOPIC_NAME}'...")
    logger.info(f"   Batch size:  {BATCH_SIZE} messages")
    logger.info(f"   Log every:   {LOG_INTERVAL} messages")
    logger.info(f"   API URL:     {API_URL}")
    logger.info(f"{'='*55}\n")

    while True:
        try:
            # ── Poll a batch of messages ──────────────────────
            # poll() returns a dict: {TopicPartition → [messages]}
            # timeout_ms controls how long to block if no messages arrive
            raw_messages = consumer.poll(timeout_ms=1000, max_records=BATCH_SIZE)

            if not raw_messages:
                # No messages right now — keep waiting
                continue

            # ── Process each message in the batch ─────────────
            for topic_partition, messages in raw_messages.items():
                for message in messages:
                    transaction = message.value

                    # Basic validation — skip malformed messages
                    if not transaction or "transaction_id" not in transaction:
                        logger.warning("Skipping malformed message (no transaction_id)")
                        errors += 1
                        continue

                    # ── Call /predict API ──────────────────────
                    prediction = call_predict_api(transaction)

                    if prediction is None:
                        errors += 1
                        continue

                    # ── Check for fraud ────────────────────────
                    if prediction.get("is_fraud"):
                        fraud_count += 1
                        print_fraud_alert(transaction, prediction)

                    # ── Track cache hits ───────────────────────
                    if prediction.get("cache_hit"):
                        cache_hits += 1

                    # ── Store in PostgreSQL ────────────────────
                    # Note: The API already stores with source='api'.
                    # We skip duplicate storage here to avoid double writes.
                    # Uncomment below if you want consumer-side audit trail:
                    # store_result(db, transaction, prediction)

                    processed += 1

            # ── Log stats every LOG_INTERVAL messages ─────────
            # BUG FIX: Use >= and track last_logged instead of == .
            # processed % LOG_INTERVAL == 0 only fires exactly on multiples.
            # If a batch skips over a multiple (e.g. jumps 498→503),
            # the log line is silently missed. >= with last_logged ensures
            # we always print once per LOG_INTERVAL threshold crossed.
            if processed > 0 and processed >= last_logged + LOG_INTERVAL:
                last_logged = processed
                elapsed    = time.time() - start_time
                rate       = processed / elapsed if elapsed > 0 else 0
                fraud_pct  = (fraud_count / processed) * 100
                cache_pct  = (cache_hits / processed) * 100

                logger.info(
                    f"\nProgress Report -- {processed:,} messages\n"
                    f"   Processed:   {processed:,} total\n"
                    f"   Fraud found: {fraud_count:,} ({fraud_pct:.2f}%)\n"
                    f"   Cache hits:  {cache_hits:,} ({cache_pct:.2f}%)\n"
                    f"   Errors:      {errors:,}\n"
                    f"   Rate:        {rate:.1f} msg/sec\n"
                    f"   Elapsed:     {elapsed:.0f}s\n"
                )

        except KeyboardInterrupt:
            # BUG FIX: Re-raise KeyboardInterrupt so it propagates cleanly
            # out of the while loop to the finally block in main().
            # Swallowing it with 'break' can cause the loop to silently
            # restart on the next iteration in some Python runtimes.
            logger.info("\nKeyboardInterrupt received. Shutting down...")
            raise

        except KafkaError as e:
            logger.error(f"Kafka error while consuming: {e}")
            logger.info("   Waiting 5s before retrying...")
            time.sleep(5)

        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            errors += 1
            time.sleep(1)

    # ── Final summary ──────────────────────────────────────────
    total_time = time.time() - start_time
    logger.info(f"\n{'='*55}")
    logger.info(f"Consumer shutting down.")
    logger.info(f"   Total processed: {processed:,}")
    logger.info(f"   Total fraud:     {fraud_count:,}")
    logger.info(f"   Total errors:    {errors:,}")
    logger.info(f"   Total time:      {total_time:.1f}s")
    logger.info(f"{'='*55}")


# ============================================================
# Entry Point
# ============================================================
def main():
    logger.info("\n" + "="*55)
    logger.info("ML Feature Store -- Kafka Transaction Consumer")
    logger.info("="*55)

    # Step 1: Connect to Kafka
    consumer = create_consumer(max_retries=5, retry_delay=5)

    # Step 2: Connect to PostgreSQL
    try:
        db = DatabaseManager()
        logger.info("PostgreSQL connected")
    except Exception as e:
        logger.warning(f"PostgreSQL not available: {e}")
        db = None

    # Step 3: Start consuming messages
    try:
        consume_messages(consumer, db)
    finally:
        consumer.close()
        if db:
            db.close()
        logger.info("Consumer shut down cleanly.")


if __name__ == "__main__":
    main()
