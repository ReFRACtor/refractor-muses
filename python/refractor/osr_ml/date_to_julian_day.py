"""
Title	: date_to_julian_day.py
To Run	: from date_to_julian_day import date_to_julian_day
Author	: Frank!
Date	: 20240503
Modf	: 20240503

"""


def date_to_julian_day(date_year, date_month, date_day):
    ##############
    # For a year, month and day find the Julian day
    #
    # Parameters
    # ---------
    # date_year : integer, float or string; year of interest
    # date_month : integer, float or string; month of interest
    # date_day : integer, float or string; day of interest
    #
    # Returns
    # -------
    # julian : integer, float or string; Julian day
    ##############
    # dummy:
    julian = -999

    # leap date_year -> s=1:
    s = 0
    if date_year % 4 == 0:
        s = 1
    if date_month == 1:
        if date_day <= 31:
            julian = date_day
    if date_month == 2:
        if date_day <= 28 and s == 0:
            julian = date_day + 31
        if date_day <= 29 and s == 1:
            julian = date_day + 31
    if date_month == 3:
        if date_day <= 31:
            julian = date_day + 31 + 28 + s
    if date_month == 4:
        if date_day <= 30:
            julian = date_day + 31 + 28 + s + 31
    if date_month == 5:
        if date_day <= 31:
            julian = date_day + 31 + 28 + s + 31 + 30
    if date_month == 6:
        if date_day <= 30:
            julian = date_day + 31 + 28 + s + 31 + 30 + 31
    if date_month == 7:
        if date_day <= 31:
            julian = date_day + 31 + 28 + s + 31 + 30 + 31 + 30
    if date_month == 8:
        if date_day <= 31:
            julian = date_day + 31 + 28 + s + 31 + 30 + 31 + 30 + 31
    if date_month == 9:
        if date_day <= 30:
            julian = date_day + 31 + 28 + s + 31 + 30 + 31 + 30 + 31 + 31
    if date_month == 10:
        if date_day <= 31:
            julian = date_day + 31 + 28 + s + 31 + 30 + 31 + 30 + 31 + 31 + 30
    if date_month == 11:
        if date_day <= 30:
            julian = date_day + 31 + 28 + s + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31
    if date_month == 12:
        if date_day <= 31:
            julian = date_day + 31 + 28 + s + 31 + 30 + 31 + 30 + 31 + 31 + 30 + 31 + 30
    #                     J  (F+s)M  A  M  J  J  A  S  O  N

    return julian
