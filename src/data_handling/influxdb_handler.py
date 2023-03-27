import configparser
import pandas as pd
from influxdb_client import InfluxDBClient



class InfluxDBHandler:
    def __init__(self):
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

    def query_maintenance_data(self):
        end_time = pd.Timestamp.utcnow()
        start_time = end_time - pd.Timedelta(minutes=1)

        query = f"""
        from(bucket:"slmp_robot")
        |> range(start: 2023-01-11T12:00:00Z, stop: 2023-01-11T12:01:00Z)
        |> filter(fn: (r) => r._measurement == "energy-consumption")
        |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
        |> group(columns: ["arm_tag"])
        """
        query_api = self.src_client.query_api()
        results = query_api.query(query=query, org=self.src_org)

        dfs = {}
        for table in results:
            df = pd.DataFrame.from_records(table.records)
            dfs[df["arm_tag"].iloc[0]] = df.drop(columns=["arm_tag"])

        print(dfs)

    def write_maintenance_results(self):
        pass
