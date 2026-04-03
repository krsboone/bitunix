import sys
import os
import datetime

class MultiLogger:
    def __init__(self, process_name):
        log_dir = "log"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.terminal = sys.stdout
        self.log_path = os.path.join(log_dir, f"{process_name}.log")
        # Open in append mode
        self.log_file = open(self.log_path, "a", encoding="utf-8")

    def write(self, message):
        # Only want to add a timestamp to messages that actually contain text
        # to avoid double-timestamping blank lines/newlines
        if message.strip():
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted_msg = f"[{timestamp}] {message}"
        else:
            formatted_msg = message

        self.terminal.write(formatted_msg)
        self.log_file.write(formatted_msg)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

def start_logging(process_name):
    logger = MultiLogger(process_name)
    sys.stdout = logger
    sys.stderr = logger # Also catch errors in the same file
