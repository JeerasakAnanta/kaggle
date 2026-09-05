"""Central config for S6E9 - Predicting Electric Vehicle Purchases."""
from pathlib import Path

COMPETITION = "playground-series-s6e9"
TARGET = "Will_Buy_EV"
POS_LABEL = "Yes"
ID_COL = "id"

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "outputs"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"
MODEL_DIR = OUTPUT_DIR / "models"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
SAMPLE_SUB_PATH = DATA_DIR / "sample_submission.csv"

NUM_FEATURES = [
    "Age",
    "Annual_Income_USD",
    "Daily_Commute_km",
    "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home",
    "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]
CAT_FEATURES = [
    "Gender",
    "City_Type",
    "Current_Car_Type",
    "Home_Charging_Possible",
    "Subsidy_Available",
    "Range_Anxiety_Level",
]

RANDOM_STATE = 42
N_SPLITS = 5
