# Author: J. Pollard


"""
I used the second year of Bay Area Bike Share Data that is openly
available on their website. The data came in csv format so unfortunately
there was no fun to be had with json. However, there was one complication
in that some of the entries in the precipitation column in the weather data
files had an alpha-character 'T' for when there were trace amounts of rain.
They defined this as when the precipitation was less than 0.01 inches on
that date. This caused issues with loading the data since the column entries
must be a consistent data type. Since there weren't that many T's in the total
set, I just went through in Excel and filtered for when the entry was T and
changed this to 0. Also, the data for the first year was split into two sets and
there was a third for the second year. Hence, all the lists and looping below.
"""

import psycopg2, psycopg2.extras

# DSN location of the AWS - RDS instance
DB_DSN = "host=****.us-west-1.rds.amazonaws.com dbname=**** " \
         "user=**** password=****"

# location of the input data file
WEATHER = "~/project/babs_open_data_year_2/201508_weather_data.csv"
TRIP = "~/project/babs_open_data_year_2/201508_trip_data.csv"
STATUS = "~/project/babs_open_data_year_2/201508_status_data.csv"
STATION = "~/project/babs_open_data_year_2/201508_station_data.csv"

TABLENAMES = ["weather", "trip", "status", "station"]


def drop_table(tablename):
    """
    drops the table from the database if it exists
    :param tablename:
    :return:
    """
    try:
        sql = "DROP TABLE IF EXISTS %s" % tablename
        conn = psycopg2.connect(dsn=DB_DSN)
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()

    except psycopg2.Error as e:
        print e.message

    else:
        cur.close()
        conn.close()


def create_weather_table():
    """
    creates a postgres table with columns date, max_temp, mean_temp, min_temp,
    max_dewpt, mean_dewpt, min_dewpt, max_humid, mean_humid, min_humid,
    max_sealev, mean_sealev, min_sealev, max_vis, mean_vis, min_vis,
    max_wind, mean_wind, max_gust, precip, cloud_cov, events, wind_dir,
    zipcode
    :return:
    """
    try:
        sql = "CREATE TABLE weather (dt DATE," \
              "max_temp INTEGER, mean_temp INTEGER, min_temp INTEGER," \
              "max_dewpt INTEGER, mean_dewpt INTEGER, min_dewpt INTEGER," \
              "max_humid INTEGER, mean_humid INTEGER, min_humid INTEGER," \
              "max_sealev NUMERIC, mean_sealev NUMERIC, min_sealev NUMERIC," \
              "max_vis INTEGER, mean_vis INTEGER, min_vis INTEGER," \
              "max_wind INTEGER, mean_wind INTEGER, max_gust INTEGER, " \
              "precip NUMERIC, cloud_cov INTEGER, events TEXT," \
              "wind_dir INTEGER, zipcode TEXT);"
        conn = psycopg2.connect(dsn=DB_DSN)
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()

    except psycopg2.Error as e:
        print e.message

    else:
        cur.close()
        conn.close()


def create_trip_table():
    """
    creates a postgres table with columns trip_id, duration,
    start_dt, start_station, start_terminal, end_dt, end_station, end_terminal,
    bike_num, subscription type, zipcode
    :return:
    """
    try:
        sql = "CREATE TABLE trip (trip_id TEXT, duration INTEGER," \
              "start_dt TIMESTAMP, start_station TEXT, start_terminal TEXT," \
              "end_dt TIMESTAMP, end_station TEXT, end_terminal TEXT," \
              "bike_num TEXT, sub_type TEXT, zipcode TEXT);"
        conn = psycopg2.connect(dsn=DB_DSN)
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()

    except psycopg2.Error as e:
        print e.message

    else:
        cur.close()
        conn.close()


def create_status_table():
    """
    creates a postgres table with columns station_id, bikes, docks, the_time
    :return:
    """
    try:
        sql = "CREATE TABLE status (station_id TEXT," \
              "bikes INTEGER, docks INTEGER, the_time TIMESTAMP);"
        conn = psycopg2.connect(dsn=DB_DSN)
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()

    except psycopg2.Error as e:
        print e.message

    else:
        cur.close()
        conn.close()


def create_station_table():
    """
    creates a postgres table with columns station_id, station_name, latitude,
    longitude, number of docks, landmark, installation
    :return:
    """
    try:
        sql = "CREATE TABLE station (station_id TEXT," \
              "station_name TEXT, lat NUMERIC, lon NUMERIC," \
              "dock_ct INTEGER, landmark TEXT, installation DATE);"
        conn = psycopg2.connect(dsn=DB_DSN)
        cur = conn.cursor()
        cur.execute(sql)
        conn.commit()

    except psycopg2.Error as e:
        print e.message

    else:
        cur.close()
        conn.close()


def insert_data(tablename, filepath):
    """
    inserts the data using copy_expert function and an open file
    :param: a table name, and a filepath
    :return:
    """
    try:
        f = open(filepath, 'r')
        sql_command = "COPY %s FROM STDIN WITH CSV HEADER DELIMITER ','"
        conn = psycopg2.connect(dsn=DB_DSN)
        cur = conn.cursor()
        cur.copy_expert(sql=sql_command % tablename, file = f)
        conn.commit()

    except psycopg2.Error as e:
        print e.message

    else:
        cur.close()
        conn.close()
        f.close()



if __name__ == '__main__':

    # drop the tables if they exist
    for tb_nm in TABLENAMES:

        print "dropping table %s" % tb_nm
        drop_table(tb_nm)

    # create the db
    print "creating weather table"
    create_weather_table()

    print "creating trip table"
    create_trip_table()

    print "creating status table"
    create_status_table()

    print "creating station table"
    create_station_table()

    # insert the weather data
    print "inserting weather file"
    insert_data(TABLENAMES[0], WEATHER)

    # inserting trip data
    print "inserting trip file"
    insert_data(TABLENAMES[1], TRIP)

    # inserting status data
    print "inserting status file"
    insert_data(TABLENAMES[2], STATUS)

    # inserting station data
    print "inserting station file"
    insert_data(TABLENAMES[3], STATION)


