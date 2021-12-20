from time import sleep
import os
import psutil

class MonitorResources():
    def __init__(self):
        """
        Top-Level class to monitor cpu, ram and gpu usage

        Attributes:
            cpu: percentage of cpu used by the process
            ram: bytes used by the process
            process: python process
        """

        self.keep_monitoring = True
        self.cpu = []
        self.ram = []
        self.process = psutil.Process(os.getpid())

    def calculate_avg(self):
        self.cpu = round(sum(self.cpu) / len(self.cpu),2)
        self.ram = round(sum(self.ram) / len(self.ram), 3)

    def measure_usage(self):
        """
        High level function executed from each tread
        """

        while self.keep_monitoring:
            cpu = round(self.process.cpu_percent()/ psutil.cpu_count(), 3)
            ram = self.process.memory_info()[1] / (1024 ** 2) #in MB
            self.cpu.append(cpu)
            self.ram.append(ram)
            sleep(0.01)
