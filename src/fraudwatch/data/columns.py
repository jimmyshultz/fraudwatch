"""Column layout of the raw IEEE-CIS files (train_transaction.csv, train_identity.csv).

Single source of truth for the raw schema, the synthetic fixture generator, and the
data pipeline. Order matches the Kaggle files.
"""

from __future__ import annotations

ID_COL = "TransactionID"
TARGET_COL = "isFraud"
TIME_COL = "TransactionDT"
AMOUNT_COL = "TransactionAmt"
EVENT_DAY_COL = "event_day"  # derived: days since dataset origin, float

CARD_COLS = [f"card{i}" for i in range(1, 7)]
C_COLS = [f"C{i}" for i in range(1, 15)]
D_COLS = [f"D{i}" for i in range(1, 16)]
M_COLS = [f"M{i}" for i in range(1, 10)]
V_COLS = [f"V{i}" for i in range(1, 340)]

TRANSACTION_COLS: list[str] = [
    ID_COL,
    TARGET_COL,
    TIME_COL,
    AMOUNT_COL,
    "ProductCD",
    *CARD_COLS,
    "addr1",
    "addr2",
    "dist1",
    "dist2",
    "P_emaildomain",
    "R_emaildomain",
    *C_COLS,
    *D_COLS,
    *M_COLS,
    *V_COLS,
]

ID_FEATURE_COLS = [f"id_{i:02d}" for i in range(1, 39)]
IDENTITY_COLS: list[str] = [ID_COL, *ID_FEATURE_COLS, "DeviceType", "DeviceInfo"]

# id_* columns that are numeric in the Kaggle data; the rest are categorical strings.
NUMERIC_ID_COLS = [
    f"id_{i:02d}"
    for i in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 17, 18, 19, 20, 21, 22, 24, 25, 26, 32)
]
CATEGORICAL_ID_COLS = [c for c in ID_FEATURE_COLS if c not in NUMERIC_ID_COLS]

PRODUCT_CODES = ["W", "H", "C", "S", "R"]
CARD4_VALUES = ["visa", "mastercard", "american express", "discover"]
CARD6_VALUES = ["debit", "credit", "charge card", "debit or credit"]
M_FLAG_VALUES = ["T", "F"]
M4_VALUES = ["M0", "M1", "M2"]
DEVICE_TYPES = ["mobile", "desktop"]

TRANSACTION_FILE = "train_transaction.csv"
IDENTITY_FILE = "train_identity.csv"
