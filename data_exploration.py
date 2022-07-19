import seaborn as sns
from pandas_profiling import ProfileReport

def profile_data(raw_df, processed_df):
    profile_raw = ProfileReport(raw_df, title="Waze raw Profiling Report")
    profile_raw.to_file("waze_raw_data_profile.html")

    profile_raw = ProfileReport(processed_df, title="Waze processed Profiling Report")
    profile_raw.to_file("waze_processed_data_profile.html")


def get_pair_plot(df):
    plt = sns.pairplot(df)
    plt.savefig("pair_plot.png")
