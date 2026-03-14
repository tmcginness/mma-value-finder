from models.logistic_model import LogisticFightModel
from models.xgboost_model import XGBoostFightModel
from models.elo_model import EloFightModel

MODEL_REGISTRY = {
    "logistic": LogisticFightModel,
    "xgboost": XGBoostFightModel,
    "elo": EloFightModel,
}
