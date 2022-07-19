import os
import numpy as np

from data_process_helper_functions import *
from task_1 import task1_split_xy, task1_split_xy_test

# def load_data(args, filename):
#     path = os.path.join(args.dir_path, filename)
#     df = pd.read_csv(path, parse_dates=['pubDate'])
#     return df


save_flag = False

def load_data(path):
    df = pd.read_csv(path, parse_dates=['pubDate'])
    return df

def preprocess_data(raw_df):
    # process:
    df = raw_df.copy()

    df['update_date'] = df.update_date.apply(convert_posix)

    # pubDate:
    df['is_weekend'] = df.pubDate.apply(lambda date: date.weekday() in WEEKEND)
    df['weekday'] = df.pubDate.apply(lambda date: date.strftime(DAY_NAME_FORMAT))
    df['seasons'] = df.pubDate.apply(lambda date: get_season(date))
    # df['holiday'] = raw_df.pubDate.apply(lambda date: get_holiday(date))
    df['window_1'] = df.pubDate.apply(lambda ts: time_in_range(TIME_WINDOW_1, ts))
    df['window_2'] = df.pubDate.apply(lambda ts: time_in_range(TIME_WINDOW_2, ts))
    df['window_3'] = df.pubDate.apply(lambda ts: time_in_range(TIME_WINDOW_3, ts))
    df['relative_time'] = df.pubDate.apply(lambda ts: (
                                ts - ts.replace(hour=0, minute=0, second=0,
                                microsecond=0)).total_seconds())

    # reportings:
    df.linqmap_reportRating = np.where(df.linqmap_reportRating >= 3, 1, 0)
    df.rename(columns={'linqmap_reportRating': 'is_highly_rated'}, inplace=True)


    # drop columns:
    df.drop(DROP_COLS, axis=1, inplace=True)

    df.drop(['linqmap_reportDescription', 'linqmap_street', 'linqmap_magvar'],
            axis=1, inplace=True)

    df = pd.get_dummies(df, columns=['linqmap_type','linqmap_subtype',
                                     'weekday', 'seasons'])

    tlv_df = df[df.linqmap_city == 'תל אביב - יפו'].drop(
        ['linqmap_city'], axis=1)

    tlv_df.reset_index(drop=True, inplace=True)
    tlv_df = pd.get_dummies(tlv_df, columns=['linqmap_roadType'])
    tlv_df.linqmap_roadType_1 = tlv_df.linqmap_roadType_17.astype(int) + tlv_df.linqmap_roadType_1.astype(int)
    drop_type_cols = [col for col in tlv_df.columns if col.startswith('linqmap_roadType_') and not (col.endswith('_1') or col.endswith('_2'))]
    tlv_df.drop(drop_type_cols, axis=1, inplace=True)

    total_cols = []
    for i in range(4):
        for c in COL_NAMES:
            total_cols.append(f'{c}_{i + 1}')

    group_df, label_df = task1_split_xy(tlv_df.to_numpy())
    tlv_df = pd.DataFrame(group_df, columns=total_cols)
    label_df = pd.DataFrame(label_df, columns=COL_NAMES)


    label_drop_cols = [col for col in label_df.columns if not (('type' in col) or (col in ['x', 'y']))]
    label_df.drop(label_drop_cols, axis=1, inplace=True)

    tlv_df['d1'] = tlv_df[['x_1', 'y_1', 'x_2', 'y_2']].apply(distance_coord, axis=1)
    tlv_df['d2'] = tlv_df[['x_2', 'y_2', 'x_3', 'y_3']].apply(distance_coord, axis=1)
    tlv_df['d3'] = tlv_df[['x_3', 'y_3', 'x_4', 'y_4']].apply(distance_coord, axis=1)
    tlv_df['x_centroid'] = tlv_df[['x_1', 'x_2', 'x_3', 'x_4']].mean(axis=1)
    tlv_df['y_centroid'] = tlv_df[['y_1', 'y_2', 'y_3', 'y_4']].mean(axis=1)

    drop_cols_dates = []
    [drop_cols_dates.extend([f'update_date_{i + 1}', f'pubDate_{i + 1}']) for i in range(4)]
    tlv_df.drop(drop_cols_dates, axis=1, inplace=True)

    tlv_df = pd.concat([tlv_df, label_df], axis=1)

    tlv_train, tlv_dev, tlv_test = split_train_dev_test(tlv_df)

    prob_cols = [c for c in COL_NAMES if 'weekday' in c] + \
                [c for c in COL_NAMES if 'window' in c] + \
                ['update_date'] + [c for c in COL_NAMES if 'linqmap_type' in c]

    prob_df = df[prob_cols]
    if save_flag:
        tel_aviv_res_path = 'waze_data_tlv.csv'
        tlv_train_res_path = 'waze_data_train_tlv.csv'
        tlv_dev_res_path = 'waze_data_dev_tlv.csv'
        tlv_test_res_path = 'waze_data_test_tlv.csv'
        prob_res_path = 'waze_data_prob.csv'

        tlv_df.to_csv(tel_aviv_res_path)
        tlv_train.to_csv(tlv_train_res_path)
        tlv_dev.to_csv(tlv_dev_res_path)
        tlv_test.to_csv(tlv_test_res_path)
        prob_df.to_csv(prob_res_path)

    return tlv_df, df


def preprocess_test(raw_test, raw_train):
    # process:
    raw_test.drop(['test_set'], axis=1, inplace=True)
    df = raw_test.append(raw_train)
    seperator = raw_test.shape[0]

    df['update_date'] = df.update_date.apply(convert_posix)

    # pubDate:
    df['is_weekend'] = df.pubDate.apply(lambda date: date.weekday() in WEEKEND)
    df['weekday'] = df.pubDate.apply(lambda date: date.strftime(DAY_NAME_FORMAT))
    df['seasons'] = df.pubDate.apply(lambda date: get_season(date))
    df['window_1'] = df.pubDate.apply(lambda ts: time_in_range(TIME_WINDOW_1, ts))
    df['window_2'] = df.pubDate.apply(lambda ts: time_in_range(TIME_WINDOW_2, ts))
    df['window_3'] = df.pubDate.apply(lambda ts: time_in_range(TIME_WINDOW_3, ts))
    df['relative_time'] = df.pubDate.apply(lambda ts: (
                                ts - ts.replace(hour=0, minute=0, second=0,
                                microsecond=0)).total_seconds())

    # reportings:
    df.linqmap_reportRating = np.where(df.linqmap_reportRating >= 3, 1, 0)
    df.rename(columns={'linqmap_reportRating': 'is_highly_rated'}, inplace=True)


    # drop columns:
    df.drop(DROP_COLS, axis=1, inplace=True)

    df.drop(['linqmap_reportDescription', 'linqmap_street', 'linqmap_magvar'],
            axis=1, inplace=True)

    df = pd.get_dummies(df, columns=['linqmap_type','linqmap_subtype',
                                     'weekday', 'seasons'])

    df.drop(['linqmap_city'] , axis=1, inplace=True)

    df = pd.get_dummies(df, columns=['linqmap_roadType'])
    df.linqmap_roadType_1 = df.linqmap_roadType_17.astype(int) + df.linqmap_roadType_1.astype(int)
    drop_type_cols = [col for col in df.columns if col.startswith('linqmap_roadType_') and not (col.endswith('_1') or col.endswith('_2'))]
    df.drop(drop_type_cols, axis=1, inplace=True)

    total_cols = []
    for i in range(4):
        for c in COL_NAMES:
            total_cols.append(f'{c}_{i + 1}')


    df = df[:seperator]
    group_df = task1_split_xy_test(df)
    df = pd.DataFrame(group_df, columns=total_cols)


    df['d1'] = df[['x_1', 'y_1', 'x_2', 'y_2']].apply(distance_coord, axis=1)
    df['d2'] = df[['x_2', 'y_2', 'x_3', 'y_3']].apply(distance_coord, axis=1)
    df['d3'] = df[['x_3', 'y_3', 'x_4', 'y_4']].apply(distance_coord, axis=1)
    df['x_centroid'] = df[['x_1', 'x_2', 'x_3', 'x_4']].mean(axis=1)
    df['y_centroid'] = df[['y_1', 'y_2', 'y_3', 'y_4']].mean(axis=1)

    drop_cols_dates = []
    [drop_cols_dates.extend([f'update_date_{i + 1}', f'pubDate_{i + 1}']) for i in range(4)]
    df.drop(drop_cols_dates, axis=1, inplace=True)

    if save_flag:
        test_res_path = 'waze_take_features_test.csv'

        df.to_csv(test_res_path)

    return df
