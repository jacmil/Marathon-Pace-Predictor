from src.baselines import riegel_predict, vdot_from_race, vdot_predict

def test_riegel_time_prediction_average_case():
    predicted = riegel_predict(race_time=18.00, race_distance=3, target_distance=13.1)
    assert abs(predicted - 85.93) < 2

def test_vdot_time_prediction_average_case():
    vdot = vdot_from_race(race_time=18, race_distance=3)
    assert abs(vdot - 53.9) < 1
    pass

def test_vdot_score_superlow():
    predicted = vdot_predict(vdot=10, race_distance=26.2)
    assert abs(predicted - 633.52) < 3

def test_vdot_score_amatuer():
    predicted = vdot_predict(vdot=40, race_distance=26.2)
    assert abs(predicted - 229.62) < 3

def test_vdot_score_elite():
    predicted = vdot_predict(vdot=65, race_distance=26.2)
    assert abs(predicted - 152.58) < 3

