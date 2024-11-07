import os
import logging

class Job:
    def __init__(self, id, type, argument):
        # Create a log file for the job
        self.success = False
        self.filename = os.path.join("logs", str(id))
        self.logger = logging.getLogger(f"job:{id}")
        self.logger.addHandler(logging.FileHandler(self.filename))
        self.logger.setLevel(logging.DEBUG)

        # Add default headers
        self.logger.info(f"Job ID: {id}")
        self.logger.info(f"Type: {type}")
        self.logger.info(f"Argument: {argument}")

    def log(self, text):
        self.logger.debug(text)