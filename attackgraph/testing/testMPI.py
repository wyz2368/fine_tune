import multiprocessing as mp
import time
import os
import subprocess

def call_and_wait_with_timeout(command_str):
    print("Will run:\n" + command_str)
    my_process = subprocess.Popen(command_str, shell=True)
    timeout_seconds = 3600
    try:
        my_process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        print("Process ran more seconds than: " + str(timeout_seconds))
    sleep_sec = 5
    time.sleep(sleep_sec)
    my_process.kill()


print(os.getcwd())