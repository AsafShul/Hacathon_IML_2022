import argparse
import plotly.graph_objects as go
import pandas as pd


def configure_run_params():
    # run example:
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', default=False,
                        help='boolian flag to profile the data', type=bool)
    parser.add_argument('--save_processed', default=False,
                        help='boolian flag to profile the data', type=bool)

    parser.add_argument('--dir_path', help='path to work dir', type=str)
    parser.add_argument('--train_name', help='name of the train file', type=str)
    parser.add_argument('--test_name', help='name of the test file', type=str)

    parser.add_argument('--train_path_q1', default='data/waze_data_train_tlv.csv',
                        help='name of the test file', type=str)
    parser.add_argument('--dev_path_q1', default='data/waze_data_dev_tlv.csv',
                        help='name of the test file', type=str)
    parser.add_argument('--test_path_q1', default='data/waze_data_test_tlv.csv',
                        help='name of the test file', type=str)
    parser.add_argument('--final_test_path_q1', default='data/waze_take_features_test.csv',
                        help='name of the test file', type=str)

    return parser.parse_args()


def create_visual_map():
    """
    Creates a map of the samples
    """
    df = pd.read_csv('waze_data.csv')
    is_highway = df.linqmap_city.isnull()
    colors = ['lightsalmon', 'lavender']

    fig = go.Figure(
        ([go.Scatter(x=df.x, y=df.y, mode="markers", marker=dict(size=0.57),
                     marker_color=[colors[int(val)] for val in is_highway.values])]),
        layout=go.Layout(xaxis_title=dict({'text': "x"}),
                         yaxis_title=dict({'text': "y"}),
                         plot_bgcolor='rgba(5,5,5,5)', height=960, width=620))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.show()
