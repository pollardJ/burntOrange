# Author: J. Pollard


from flask import Flask, request, jsonify
import psycopg2, psycopg2.extras

# DSN location of the AWS - RDS instance
DB_DSN = "host=****.us-west-1.rds.amazonaws.com dbname=****" \
         "user=**** password=****"

app = Flask(__name__)

@app.route('/')
def default():

    output = dict()

    # a nice little welcome message, you know, happy to have you visit!
    output['message'] = 'Welcome to the Bay Area Bike Share App!'

    return jsonify(output)


@app.route('/station')
def get_station_count():
    """
    returns the number of stations in the bay area bike share program
    :return:
    """
    output = dict()

    try:
        sql = "SELECT count(station_name) as stations FROM station;"
        conn = psycopg2.connect(dsn=DB_DSN)
        cur = conn.cursor()
        cur.execute(sql)
        rs = cur.fetchall()

        for el in rs:
            output['station_count'] = el[0]

    except psycopg2.Error as e:
        print e.message

    else:
        cur.close()
        conn.close()

    return jsonify(output)


@app.route('/bike')
def get_bike_count():
    """
    returns the number of stations in the bay area bike share program
    :return:
    """
    output = dict()

    try:
        sql = "SELECT count(bike_num) FROM (SELECT DISTINCT bike_num FROM trip) as inner_q;"
        conn = psycopg2.connect(dsn=DB_DSN)
        cur = conn.cursor()
        cur.execute(sql)
        rs = cur.fetchall()

        for el in rs:
            output['bike_count'] = el[0]

    except psycopg2.Error as e:
        print e.message

    else:
        cur.close()
        conn.close()

    return jsonify(output)


@app.route('/subscribers')
def get_num_subscribers():
    """
    calculates the number of trips taken by subscribers and the total number
    of trips for reference
    :return:
    """
    output = dict()

    try:
        sql = "SELECT sub_type, count(sub_type) as ct FROM trip GROUP BY sub_type;"
        conn = psycopg2.connect(dsn=DB_DSN)
        cur = conn.cursor()
        cur.execute(sql)
        rs = cur.fetchall()
        print rs
        for item in rs:
            output[item[0]] = item[1]

    except psycopg2.Error as e:
        print e.message

    else:
        cur.close()
        conn.close()

    return jsonify(output)


@app.route('/station/landmark')
def get_station_landmarks():
    """
    returns the counts of unique station ids per landmark
    """
    output = dict()

    try:
        sql = "SELECT landmark, count(landmark) as ct FROM station GROUP BY landmark;"
        conn = psycopg2.connect(dsn=DB_DSN)
        cur = conn.cursor()
        cur.execute(sql)
        rs = cur.fetchall()

        for el in rs:
            output[el[0]] = el[1]

    except psycopg2.Error as e:
        print e.message

    else:
        cur.close()
        conn.close()

    return jsonify(output)


@app.route('/trip/start')
def get_trip_start_station():
    """
    finds the unique start stations in the trip data and returns the
    number of times they occur
    :return: a dict of all kv pairs, key = borough and value = count
    """
    out = dict()

    sql = "SELECT start_station, count(start_station) as ct FROM trip " \
          "GROUP BY start_station;"
    try:
        conn = psycopg2.connect(dsn=DB_DSN)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql)
        rs = cur.fetchall()

        for item in rs:
            out[item['start_station']] = item['ct']

    except psycopg2.Error as e:
        print e.message

    else:
        cur.close()
        conn.close()

    return jsonify(out)


@app.route('/trip/start/avg_duration')
def get_trip_station_duration():
    """
    computes the average trip duration in minutes by start station
    :return: a dict of all kv pairs, key = borough and value = count
    """
    out = dict()

    sql = "SELECT start_station, AVG(duration::FLOAT/60) OVER(PARTITION BY start_station) " \
          "FROM trip;"

    try:
        conn = psycopg2.connect(dsn=DB_DSN)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql)
        rs = cur.fetchall()

        for item in rs:
            out[item['start_station']] = item['avg']

    except psycopg2.Error as e:
        print e.message

    else:
        cur.close()
        conn.close()

    return jsonify(out)


@app.route('/trip/bike/avg_duration')
def get_trip_bike_duration():
    """
    computes average trip duration in minutes by bike id
    :return: a dict of all kv pairs, key = borough and value = count
    """
    out = dict()

    sql = "SELECT bike_num, AVG(duration::FLOAT/60) OVER(PARTITION BY bike_num) " \
          "FROM trip;"

    try:
        conn = psycopg2.connect(dsn=DB_DSN)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql)
        rs = cur.fetchall()

        for item in rs:
            out[item['bike_num']] = item['avg']

    except psycopg2.Error as e:
        print e.message

    else:
        cur.close()
        conn.close()

    return jsonify(out)


@app.route('/weather/avgprecip')
def get_avg_precip_zipcode():
    """
    computes the average precipitation in inches by zipcode
    """
    output = dict()

    try:
        sql = "SELECT zipcode, avg(precip::FLOAT) as avg_precip FROM weather " \
              "GROUP BY 1;"
        conn = psycopg2.connect(dsn=DB_DSN)
        cur = conn.cursor()
        cur.execute(sql)
        rs = cur.fetchall()
        print rs
        for item in rs:
            output[item[0]] = item[1]

    except psycopg2.Error as e:
        print e.message

    else:
        cur.close()
        conn.close()

    return jsonify(output)


@app.route('/trip/stendsame')
def get_start_same_end():
    """
    computes the number of trips that start and end at the same station
    """
    output = dict()

    try:
        sql = "SELECT start_station, count(start_station) as ct FROM " \
              "(SELECT start_station, end_station FROM trip WHERE start_station = end_station) as inner_q " \
              "GROUP BY start_station;"
        conn = psycopg2.connect(dsn=DB_DSN)
        cur = conn.cursor()
        cur.execute(sql)
        rs = cur.fetchall()

        for item in rs:
            output[item[0]] = item[1]

    except psycopg2.Error as e:
        print e.message

    else:
        cur.close()
        conn.close()

    return jsonify(output)


@app.route('/trip/stenddiff')
def get_start_diff_end():
    """
    computes the number of trips that start and end at different stations
    """
    output = dict()

    try:
        sql = "SELECT start_station, count(start_station) as ct FROM " \
              "(SELECT start_station, end_station FROM trip WHERE start_station != end_station) as inner_q " \
              "GROUP BY start_station;"
        conn = psycopg2.connect(dsn=DB_DSN)
        cur = conn.cursor()
        cur.execute(sql)
        rs = cur.fetchall()

        for item in rs:
            output[item[0]] = item[1]

    except psycopg2.Error as e:
        print e.message

    else:
        cur.close()
        conn.close()

    return jsonify(output)


@app.route('/trip/difference')
def get_start_end_difference():
    """
    computes the difference in the counts of trips that start and end at the same station and
    those that start and end at different stations
    a negative value means that more trips that begin at that station do not return to
    that station
    """
    output = dict()

    try:
        sql = "SELECT lhs.start_station, lhs.ct - rhs.ct as diff FROM " \
              "(SELECT start_station, count(start_station) as ct FROM " \
              "(SELECT start_station, end_station FROM trip WHERE start_station = end_station) as inner_q " \
              "GROUP BY start_station) as lhs " \
              "INNER JOIN " \
              "(SELECT start_station, count(start_station) as ct FROM " \
              "(SELECT start_station, end_station FROM trip WHERE start_station != end_station) as inner_q " \
              "GROUP BY start_station) as rhs " \
              "ON lhs.start_station = rhs.start_station;"
        conn = psycopg2.connect(dsn=DB_DSN)
        cur = conn.cursor()
        cur.execute(sql)
        rs = cur.fetchall()

        for item in rs:
            output[item[0]] = item[1]

    except psycopg2.Error as e:
        print e.message

    else:
        cur.close()
        conn.close()

    return jsonify(output)


if __name__ == "__main__":

    # app.debug = True # only have this on for debugging!
    app.run(host='****') # need this to access from the outside world!

