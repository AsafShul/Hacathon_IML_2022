import pandas as pd
import datetime as dt


def generate_baseline_task2(df):
    """
    Generate baseline for task 2, using average
    """
    events = ['ACCIDENT', 'JAM', 'ROAD_CLOSED', 'WEATHERHAZARD']
    time_slots = ['window_1', 'window_2', 'window_3']

    avg_data = pd.DataFrame(columns = events, index = time_slots)

    for ts in time_slots:
        evgs_for_event = []
        for event in events:
            sum_df = df.groupby([(pd.to_datetime(df.pubDate).dt.date), (df.linqmap_type == event)])[ts].sum()
            temp = sum_df.index.to_frame(index=False).linqmap_type
            evgs_for_event.append(float(pd.DataFrame({'sum': sum_df.values})[temp].sum() / 5))

        avg_data.loc[ts] = evgs_for_event

    avg_data = avg_data.rename(index={'window_1': '8:00-10:00', 'window_2': '12:00-14:00', 'window_3': '18:00-20:00'})
    return avg_data


def generate_labels_task2(df, dates_list):
    for date in dates_list:
        generate_labels_task2_helper(df, pd.to_datetime(date))


def generate_labels_task2_helper(df, date):
    """
    Using statistic learning (average with adjustments) to create predictions
    """
    events = ['linqmap_type_ACCIDENT', 'linqmap_type_JAM',
              'linqmap_type_ROAD_CLOSED', 'linqmap_type_WEATHERHAZARD']
    time_slots = ['window_1', 'window_2', 'window_3']

    avg_data = pd.DataFrame(columns=events, index=time_slots)
    day = date.weekday()

    for ts in time_slots:
        evgs_for_event = []
        for event in events:

            slice_df = df[df[ts] & df[event]]
            event_sum = slice_df.groupby([(pd.to_datetime(slice_df.update_date).dt.date)]).sum().sum(axis=0)
            event_sum = event_sum[['weekday_Sunday', 'weekday_Monday', 'weekday_Tuesday', 'weekday_Wednesday']]

            val = (event_sum).sum() / 5

            # thursday - 5% less in 08:00-10:00
            if day == 3 and ts == 'window_1':
                val = val * 0.95

            # shavuot holiday, like saturday - 64% of regular traffic
            if day == 6:
                val = val * 0.64

            evgs_for_event.append(round(val, 3))

        avg_data.loc[ts] = evgs_for_event

    avg_data = avg_data.rename(columns={'linqmap_type_ACCIDENT': 'ACCIDENT',
                                        'linqmap_type_JAM': 'JAM',
                                        'linqmap_type_ROAD_CLOSED': 'ROAD_CLOSED',
                                        'linqmap_type_WEATHERHAZARD': 'WEATHERHAZARD'},

                               index={'window_1': '8:00-10:00',
                                      'window_2': '12:00-14:00',
                                      'window_3': '18:00-20:00'})

    avg_data.to_csv(f'second_task_{date.day}.{date.month}.{date.year}.csv',
                    index=False, header=False)
    return avg_data
