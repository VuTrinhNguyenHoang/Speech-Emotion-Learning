from .crema import EMO_MAP

EMO2ID = {e: i for i, e in enumerate(EMO_MAP.values())}
ID2EMO = {i: e for e, i in EMO2ID.items()}