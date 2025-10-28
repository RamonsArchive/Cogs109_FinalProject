import os
from load import load_data
from clean import clean_data
from model import train_model_1


def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(
        curr_dir,
        "../datasets/df_master_schedule_injury_surface_2019_23_weather_distance_days_since_last_game.csv",
    )

    df = load_data(data_dir)
    df = clean_data(df)
    results = train_model_1(df)

    print("--------------------------------")
    print("Model 1 Results:")
    print("--------------------------------")
    print(results["10foldCV"])


if __name__ == "__main__":
    main()
