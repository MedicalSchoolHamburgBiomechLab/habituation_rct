from spiro import get_demographics, load_df_spiro

if __name__ == '__main__':
    df_spiro = load_df_spiro()
    df_demographics = get_demographics()
    df_merged = df_spiro.merge(df_demographics, on="participant_id", how="left")
    print(df_merged.head())