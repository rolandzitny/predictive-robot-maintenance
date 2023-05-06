"""
This handler enables communication with the InfluxDB database.
It provides the possibility of obtaining sensor data from the database.
"""
import configparser
import pandas as pd
from influxdb_client import InfluxDBClient


class InfluxDBHandler:
    def __init__(self):
        """
        Initialize handler using configuration file.
        """
        config = configparser.ConfigParser()
        config.read('config/configurations.ini')

        # Data source connection
        self.src_host = config.get('influxdb-source', 'host')
        self.src_token = config.get('influxdb-source', 'token')
        self.src_bucket = config.get('influxdb-source', 'bucket')
        self.src_org = config.get('influxdb-source', 'org')
        self.src_client = InfluxDBClient(url=self.src_host,
                                         token=self.src_token,
                                         org=self.src_org,
                                         timeout=0)

        # Data destination connection
        self.dst_host = config.get('influxdb-destination', 'host')
        self.dst_token = config.get('influxdb-destination', 'token')
        self.dst_bucket = config.get('influxdb-destination', 'bucket')
        self.dst_org = config.get('influxdb-destination', 'org')
        self.dst_client = InfluxDBClient(url=self.dst_host,
                                         token=self.dst_token,
                                         org=self.dst_org,
                                         timeout=0)

    def query_energy_consumption_by_hours(self, hours, arm_tag):
        """
        Query X hours of sensory data from database.

        :param hours: Number of hours.
        :param arm_tag: Tag of diagnosed device in InfluxDB.
        :return: Dataframe with energy consumption data.
        """
        query = f'from(bucket:"{self.src_bucket}") \
                |> range(start:-{str(hours)}h) \
                |> filter(fn: (r) => r._measurement == "energy-consumption" and r.arm_tag == "{arm_tag}" and\
                 (r._field == "J1" or r._field == "J2" or r._field == "J3" or \
                 r._field == "J4" or r._field == "J5" or r._field == "J6"))\
                |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")'

        query_api = self.src_client.query_api()
        result = query_api.query_data_frame(query, org=self.src_org)
        df = pd.DataFrame(result)
        df = df[["J1", "J2", "J3", "J4", "J5", "J6"]]
        # Extract the signal values as a numpy array
        energy_consumption_values = df.values[:, 0:]

        return energy_consumption_values

    def query_energy_consumption_by_day(self, date, arm_tag):
        """
        Query one day of sensory data from database.

        :param date: Date of desired day.
        :param arm_tag: Tag of diagnosed device in InfluxDB.
        :return: Dataframe with energy consumption data.
        """
        query = f'from(bucket:"{self.src_bucket}") \
                |> range(start: {date}T06:00:00Z, stop: {date}T22:00:00Z)\
                |> filter(fn: (r) => r._measurement == "energy-consumption" and r.arm_tag == "{arm_tag}" and\
                 (r._field == "J1" or r._field == "J2" or r._field == "J3" or \
                 r._field == "J4" or r._field == "J5" or r._field == "J6"))\
                |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")'

        query_api = self.src_client.query_api()
        result = query_api.query_data_frame(query, org=self.src_org)
        df = pd.DataFrame(result)
        df = df[["J1", "J2", "J3", "J4", "J5", "J6"]]
        # Extract the signal values as a numpy array
        energy_consumption_values = df.values[:, 0:]

        return energy_consumption_values

    def write_maintenance_results(self):
        # TODO
        pass
