from src.data_generator import generate_wind_dataset

def test_dataset_not_empty():
    df = generate_wind_dataset()
    assert df is not None
    assert len(df) > 0