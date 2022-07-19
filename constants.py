import datetime as dt
import pandas as pd

# constants:
POSIX_FACTOR = 1000.0
DAY_NAME_FORMAT = '%A'

Y = 2000  # dummy leap year to allow input X-02-29 (leap day)
WEEKEND = [4, 5]
SEASONS = [('winter', (dt.date(Y,  1,  1),  dt.date(Y,  3,  1))),
           ('spring', (dt.date(Y,  3,  2),  dt.date(Y,  5, 31))),
           ('summer', (dt.date(Y,  6,  1),  dt.date(Y,  9, 20))),
           ('autumn', (dt.date(Y,  9, 21),  dt.date(Y, 11, 30))),
           ('winter', (dt.date(Y, 12,  1),  dt.date(Y, 12, 31)))]

HOLIDAY_EVE = [dt.date(2021, 3, 27), dt.date(2021, 4, 14),
               dt.date(2021, 5, 16), dt.date(2021, 9, 6),
               dt.date(2021, 9, 15), dt.date(2021, 9, 20),
               dt.date(2022, 4, 15)]

HOL_HMOED = [[dt.date(2021, 2, 25), dt.date(2021, 2, 27)],
             [dt.date(2021, 3, 29), dt.date(2021, 4, 3)],
             [dt.date(2021, 4, 15), dt.date(2021, 4, 15)],
             [dt.date(2021, 4, 30), dt.date(2021, 4, 30)],
             [dt.date(2021, 9, 22), dt.date(2021, 9, 29)],
             [dt.date(2021, 11, 30), dt.date(2021, 12, 6)],
             [dt.date(2022, 3, 16), dt.date(2022, 3, 18)],
             [dt.date(2022, 4, 17), dt.date(2022, 4, 23)],
             [dt.date(2022, 5, 5), dt.date(2022, 5, 5)],
             [dt.date(2022, 5, 19), dt.date(2022, 5, 19)]]

HOLIDAY = [dt.date(2021, 3, 28), dt.date(2021, 5, 17), dt.date(2021, 5, 18),
           dt.date(2021, 9, 7), dt.date(2021, 9, 8), dt.date(2021, 9, 21),
           dt.date(2022, 4, 16), dt.date(2022, 6, 5)]


TIME_WINDOW_1 = [pd.to_datetime(dt.datetime(Y, 1, 1,  8, 0)), pd.to_datetime(dt.datetime(Y, 1, 1, 10, 0))]
TIME_WINDOW_2 = [pd.to_datetime(dt.datetime(Y, 1, 1, 12, 0)), pd.to_datetime(dt.datetime(Y, 1, 1, 14, 0))]
TIME_WINDOW_3 = [pd.to_datetime(dt.datetime(Y, 1, 1, 18, 0)), pd.to_datetime(dt.datetime(Y, 1, 1, 20, 0))]

DROP_COLS = ['linqmap_nearby', 'linqmap_expectedBeginDate', 'nComments',
             'linqmap_expectedEndDate', 'linqmap_reportMood', 'OBJECTID']

TEL_AVIV_CENTER = (179142.84, 663973.52)

COL_NAMES = [
       'pubDate', 'is_highly_rated', 'linqmap_reliability', 'update_date', 'x',
       'y', 'is_weekend', 'window_1', 'window_2', 'window_3', 'relative_time',
       'linqmap_type_ACCIDENT', 'linqmap_type_JAM', 'linqmap_type_ROAD_CLOSED',
       'linqmap_type_WEATHERHAZARD', 'linqmap_subtype_ACCIDENT_MAJOR',
       'linqmap_subtype_ACCIDENT_MINOR', 'linqmap_subtype_HAZARD_ON_ROAD',
       'linqmap_subtype_HAZARD_ON_ROAD_CAR_STOPPED',
       'linqmap_subtype_HAZARD_ON_ROAD_CONSTRUCTION',
       'linqmap_subtype_HAZARD_ON_ROAD_ICE',
       'linqmap_subtype_HAZARD_ON_ROAD_OBJECT',
       'linqmap_subtype_HAZARD_ON_ROAD_POT_HOLE',
       'linqmap_subtype_HAZARD_ON_ROAD_ROAD_KILL',
       'linqmap_subtype_HAZARD_ON_ROAD_TRAFFIC_LIGHT_FAULT',
       'linqmap_subtype_HAZARD_ON_SHOULDER',
       'linqmap_subtype_HAZARD_ON_SHOULDER_ANIMALS',
       'linqmap_subtype_HAZARD_ON_SHOULDER_CAR_STOPPED',
       'linqmap_subtype_HAZARD_ON_SHOULDER_MISSING_SIGN',
       'linqmap_subtype_HAZARD_WEATHER',
       'linqmap_subtype_HAZARD_WEATHER_FLOOD',
       'linqmap_subtype_HAZARD_WEATHER_FOG',
       'linqmap_subtype_HAZARD_WEATHER_HAIL',
       'linqmap_subtype_HAZARD_WEATHER_HEAVY_SNOW',
       'linqmap_subtype_JAM_HEAVY_TRAFFIC',
       'linqmap_subtype_JAM_MODERATE_TRAFFIC',
       'linqmap_subtype_JAM_STAND_STILL_TRAFFIC',
       'linqmap_subtype_ROAD_CLOSED_CONSTRUCTION',
       'linqmap_subtype_ROAD_CLOSED_EVENT', 'weekday_Friday', 'weekday_Monday',
       'weekday_Saturday', 'weekday_Sunday', 'weekday_Thursday',
       'weekday_Tuesday', 'weekday_Wednesday', 'seasons_autumn',
       'seasons_spring', 'seasons_summer', 'seasons_winter',
       'linqmap_roadType_1', 'linqmap_roadType_2']


TYPES=['ACCIDENT', 'JAM', 'ROAD_CLOSED', 'WEATHERHAZARD']
SUB_TYPES=['ACCIDENT_MAJOR',
       'ACCIDENT_MINOR', 'HAZARD_ON_ROAD',
       'HAZARD_ON_ROAD_CAR_STOPPED',
       'HAZARD_ON_ROAD_CONSTRUCTION',
       'HAZARD_ON_ROAD_ICE',
       'HAZARD_ON_ROAD_OBJECT',
       'HAZARD_ON_ROAD_POT_HOLE',
       'HAZARD_ON_ROAD_ROAD_KILL',
       'HAZARD_ON_ROAD_TRAFFIC_LIGHT_FAULT',
       'HAZARD_ON_SHOULDER',
       'HAZARD_ON_SHOULDER_ANIMALS',
       'HAZARD_ON_SHOULDER_CAR_STOPPED',
       'HAZARD_ON_SHOULDER_MISSING_SIGN',
       'HAZARD_WEATHER',
       'HAZARD_WEATHER_FLOOD',
       'HAZARD_WEATHER_FOG',
       'HAZARD_WEATHER_HAIL',
       'HAZARD_WEATHER_HEAVY_SNOW',
       'JAM_HEAVY_TRAFFIC',
       'JAM_MODERATE_TRAFFIC',
       'JAM_STAND_STILL_TRAFFIC',
       'ROAD_CLOSED_CONSTRUCTION',
       'ROAD_CLOSED_EVENT']
