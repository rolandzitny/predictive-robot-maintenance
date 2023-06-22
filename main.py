"""
Main program. Schedule diagnosis every hour and history diagnosis every 24 hours.
"""

import time
import schedule
import processes
import configparser
import multiprocessing


config = configparser.ConfigParser()
config.read('config/configurations.ini')
robot_names = config['maintained-arms']['arms'].split(', ')


def run_diagnose(robot_name):
    processes.diagnose_robot_health(robot_name)


def run_diagnose_history(robot_name):
    processes.diagnose_robot_health_history(robot_name)


def schedule_diagnose(robot_name):
    # Schedule run_diagnose to run every hour between 6:00 and 23:00
    for hour in range(6, 24):
        schedule.every().day.at(f"{hour:02d}:00").do(run_diagnose, robot_name)

    # Schedule run_diagnose_history to run once per day at 23:00
    schedule.every().day.at("23:00").do(run_diagnose_history, robot_name)

    # Loop indefinitely to keep the schedule running
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == '__main__':
    # Create a process for each robot name
    diagnostic_processes = []

    # Create a process for each robot name
    for robot_name in robot_names:
        process = multiprocessing.Process(target=schedule_diagnose, args=([robot_name],))
        diagnostic_processes.append(process)

    # Start all processes
    for process in diagnostic_processes:
        process.start()

    # Wait for all processes to finish
    for process in diagnostic_processes:
        process.join()
