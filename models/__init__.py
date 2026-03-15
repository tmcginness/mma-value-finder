from models.logistic_model import LogisticFightModel
from models.elo_model import EloFightModel

MODEL_REGISTRY = {
    "logistic": LogisticFightModel,
    "elo": EloFightModel,
}

try:
    from models.xgboost_model import XGBoostFightModel
    MODEL_REGISTRY["xgboost"] = XGBoostFightModel
except Exception:
    pass
