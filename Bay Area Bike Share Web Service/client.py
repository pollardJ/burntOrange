# Author: J. Pollard


import httplib

SERVER = '****.us-west-1.compute.amazonaws.com:****'


def default():
    out = dict()
    h = httplib.HTTPConnection(SERVER)
    # want the url to look something like this
    # 'http://localhost:5000/restaurants/borough/counts'
    h.request('GET', 'http://'+SERVER+'/')
    resp = h.getresponse()
    out = resp.read()
    return out


def get_station_count():
    out = dict()
    h = httplib.HTTPConnection(SERVER)
    # want the url to look something like this
    # 'http://localhost:5000/restaurants/borough/counts'
    h.request('GET', 'http://'+SERVER+'/station')
    resp = h.getresponse()
    out = resp.read()
    return out


def get_bike_count():
    out = dict()
    h = httplib.HTTPConnection(SERVER)
    # want the url to look something like this
    # 'http://localhost:5000/restaurants/borough/counts'
    h.request('GET', 'http://'+SERVER+'/bike')
    resp = h.getresponse()
    out = resp.read()
    return out


def get_num_subscribers():
    out = dict()
    h = httplib.HTTPConnection(SERVER)
    # want the url to look something like this
    # 'http://localhost:5000/restaurants/borough/counts'
    h.request('GET', 'http://'+SERVER+'/subscribers')
    resp = h.getresponse()
    out = resp.read()
    return out


def get_station_landmarks():
    out = dict()
    h = httplib.HTTPConnection(SERVER)
    # want the url to look something like this
    # 'http://localhost:5000/restaurants/borough/counts'
    h.request('GET', 'http://'+SERVER+'/station/landmark')
    resp = h.getresponse()
    out = resp.read()
    return out


def get_trip_start_station():
    out = dict()
    h = httplib.HTTPConnection(SERVER)
    # want the url to look something like this
    # 'http://localhost:5000/restaurants/borough/counts'
    h.request('GET', 'http://'+SERVER+'/trip/start')
    resp = h.getresponse()
    out = resp.read()
    return out


def get_trip_station_duration():
    out = dict()
    h = httplib.HTTPConnection(SERVER)
    # want the url to look something like this
    # 'http://localhost:5000/restaurants/borough/counts'
    h.request('GET', 'http://'+SERVER+'/trip/start/avg_duration')
    resp = h.getresponse()
    out = resp.read()
    return out


def get_trip_bike_duration():
    out = dict()
    h = httplib.HTTPConnection(SERVER)
    # want the url to look something like this
    # 'http://localhost:5000/restaurants/borough/counts'
    h.request('GET', 'http://'+SERVER+'/trip/bike/avg_duration')
    resp = h.getresponse()
    out = resp.read()
    return out


def get_avg_precip_zipcode():
    out = dict()
    h = httplib.HTTPConnection(SERVER)
    # want the url to look something like this
    # 'http://localhost:5000/restaurants/borough/counts'
    h.request('GET', 'http://'+SERVER+'/weather/avgprecip')
    resp = h.getresponse()
    out = resp.read()
    return out


def get_start_same_end():
    out = dict()
    h = httplib.HTTPConnection(SERVER)
    # want the url to look something like this
    # 'http://localhost:5000/restaurants/borough/counts'
    h.request('GET', 'http://'+SERVER+'/trip/stendsame')
    resp = h.getresponse()
    out = resp.read()
    return out


def get_start_diff_end():
    out = dict()
    h = httplib.HTTPConnection(SERVER)
    # want the url to look something like this
    # 'http://localhost:5000/restaurants/borough/counts'
    h.request('GET', 'http://'+SERVER+'/trip/stenddiff')
    resp = h.getresponse()
    out = resp.read()
    return out


def get_start_end_difference():
    out = dict()
    h = httplib.HTTPConnection(SERVER)
    # want the url to look something like this
    # 'http://localhost:5000/restaurants/borough/counts'
    h.request('GET', 'http://'+SERVER+'/trip/difference')
    resp = h.getresponse()
    out = resp.read()
    return out


if __name__ == '__main__':
    print "======================================================="
    print "Test of my Bay Area Bike Share app running at ", SERVER
    print "created by Jacob Pollard using year two data from the "
    print "Bay Area Bike Share program"
    print "======================================================="
    print " "
    print "******** welcome **********"
    print default()
    print " "
    print "******** Number of Stations in the Bike Share **********"
    print get_station_count()
    print " "
    print "******** Number of Bikes in the Bike Share *************"
    print get_bike_count()
    print " "
    print "******** Number of Subscribers/Nonsubscribers **********"
    print get_num_subscribers()
    print " "
    print "******** The Station Names *****************************"
    print get_station_landmarks()
    print " "
    print "******** Counts of Trips That Start at a Station *******"
    print get_trip_start_station()
    print " "
    print "******** Avg Duration of Trip From Starting Station ****"
    print get_trip_station_duration()
    print " "
    print "******** Avg Duration of Trip by Bike ID ***************"
    print get_trip_bike_duration()
    print " "
    print "******** Avg Precipitation by Zipcode ******************"
    print get_avg_precip_zipcode()
    print " "
    print "******** Count of trips that end at same start *********"
    print get_start_same_end()
    print " "
    print "******** Count of trips that end at different station **"
    print get_start_diff_end()
    print " "
    print "******** Difference of trips that end in the same station"
    print "******** and those that end at a different station ******"
    print get_start_end_difference()
    print " "
    print "========================================================"
