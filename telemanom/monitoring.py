from time import sleep
import os
import psutil
import subprocess

class MonitorResources():
    def __init__(self, hardware):
        """
        Top-Level class to monitor cpu, ram and gpu usage

        Attributes:
            cpu: percentage of cpu used by the process
            ram: bytes used by the process
            process: python process
        """

        self.keep_monitoring = True
        self.gpu = []
        self.cpu = []
        self.ram = []
        self.hardware = hardware
        self.sync = False

    def calculate_avg(self):

        while not self.sync:
            sleep(0.01)

        if len(self.cpu) > 0:
            self.cpu = round(sum(self.cpu) / len(self.cpu),2)
        if len(self.gpu) > 0:
            self.gpu = round(sum(self.gpu) / len(self.gpu), 2)
        self.ram = round(sum(self.ram) / len(self.ram), 3)

    def measure_usage(self):
        """
        High level function executed from each tread
        """
        if self.hardware == 'cpu':
            self.process = psutil.Process(os.getpid())
            while self.keep_monitoring:
                cpu = round(self.process.cpu_percent()/ psutil.cpu_count(), 3)
                ram = self.process.memory_info()[1] / (1024 ** 2) #in MB
                self.cpu.append(cpu)
                self.ram.append(ram)
                sleep(0.1)

        elif self.hardware == 'gpu':
            process = subprocess.Popen('/usr/bin/tegrastats', stdout=subprocess.PIPE, encoding='UTF8')
            while self.keep_monitoring:
                sleep(0.1)

            process.kill()
            lines = process.stdout.read().split('\n')

            for line in lines[:-1]:
                words = line.split(' ')
                ram = int(words[1].split('/')[0])
                gpu = int(words[13][:-1])
                self.ram.append(ram)
                if gpu > 0:
                    self.gpu.append(gpu)

        self.sync = True