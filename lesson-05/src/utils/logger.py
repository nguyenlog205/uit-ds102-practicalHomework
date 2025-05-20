import logging
import json
import csv
import re
from datetime import datetime
from colorama import Fore, init
import os

# Initialize colorama for colored terminal output
# This ensures colors reset automatically after each colored output.
init(autoreset=True)

class SimpleLogger:
    """
    A simple and easy-to-use logging utility with colored console output
    and optional file logging, plus built-in export functionality.
    """
    _instance = None # Class-level variable to hold the singleton instance
    _is_initialized = False # Flag to ensure initialization runs only once

    def __new__(cls, *args, **kwargs):
        """
        Implements a Singleton pattern to ensure only one instance of the logger exists.
        This prevents duplicate handlers if setup_logger is called multiple times.
        """
        if cls._instance is None:
            cls._instance = super(SimpleLogger, cls).__new__(cls)
        return cls._instance

    def __init__(self, log_file=None):
        """
        Initializes the SimpleLogger. This method will only execute its
        setup logic once, even if called multiple times due to the singleton pattern.

        Args:
            log_file (str, optional): Path to a file where logs should be written.
                                      If None, logs are only sent to the console.
        """
        if self._is_initialized:
            return # Already initialized, prevent re-setup of handlers etc.

        self.logger = logging.getLogger("simple_app_logger")
        self.logger.setLevel(logging.DEBUG) # Capture all messages

        self.log_file = log_file

        # Ensure log file directory exists if a log_file is specified
        if log_file:
            dir_name = os.path.dirname(log_file)
            if dir_name and not os.path.exists(dir_name):
                try:
                    os.makedirs(dir_name)
                except OSError as e:
                    print(f"{Fore.RED}ERROR: Could not create log directory '{dir_name}': {e}{Fore.RESET}")
                    self.log_file = None # Disable file logging if dir creation fails

        # Setup handlers only if they haven't been added yet
        if not self.logger.handlers:
            formatter = logging.Formatter('%(message)s')

            # Console Handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # File Handler (if log_file is provided and directory creation was successful)
            if self.log_file:
                try:
                    file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
                    file_handler.setLevel(logging.DEBUG)
                    file_handler.setFormatter(formatter)
                    self.logger.addHandler(file_handler)
                except Exception as e:
                    print(f"{Fore.RED}ERROR: Could not set up file handler for '{self.log_file}': {e}{Fore.RESET}")
                    self.log_file = None # Disable file logging if handler setup fails

        self._is_initialized = True # Mark as initialized

    def _format_message(self, level: str, msg: str, color: str = "") -> str:
        """Internal method to format log messages with timestamp, level, and color."""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"{color}[{timestamp}] [{level}] {msg}{Fore.RESET}"

    def info(self, msg: str):
        """Logs an informational message."""
        self.logger.info(self._format_message("INFO", msg, Fore.CYAN))

    def success(self, msg: str):
        """Logs a success message."""
        self.logger.info(self._format_message("SUCCESS", msg, Fore.GREEN))

    def warning(self, msg: str):
        """Logs a warning message."""
        self.logger.warning(self._format_message("WARNING", msg, Fore.YELLOW))

    def error(self, msg: str):
        """Logs an error message."""
        self.logger.error(self._format_message("ERROR", msg, Fore.RED))

    def critical(self, msg: str):
        """Logs a critical message."""
        self.logger.critical(self._format_message("CRITICAL", msg, Fore.MAGENTA))

    def read_log_file(self) -> list[str]:
        """
        Reads and returns all lines from the configured log file.
        Raises an error if no log file was specified or found.
        """
        if not self.log_file:
            raise ValueError("No log file was specified during logger initialization. Cannot read logs.")
        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                return f.readlines()
        except FileNotFoundError:
            raise FileNotFoundError(f"Log file not found: '{self.log_file}'. Make sure it exists and messages have been written to it.")
        except Exception as e:
            raise IOError(f"Error reading log file '{self.log_file}': {e}")

    def export_to_file(self, output_path: str, format: str = 'csv'):
        """
        Exports parsed log entries from the log file to a CSV, JSON, or TXT file.

        Args:
            output_path (str): The destination file path for the exported logs.
            format (str): The desired output format ('csv', 'json', or 'txt').
        """
        if not self.log_file:
            raise ValueError("No log file was specified during logger initialization. Cannot export logs.")

        log_line_pattern = re.compile(r'\[(.*?)\] \[(.*?)\] (.*)')
        entries = []

        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    clean_line = re.sub(r'\x1b\[\d+m', '', line).strip() # Remove ANSI codes if present
                    match = log_line_pattern.match(clean_line)
                    if match:
                        timestamp, level, message = match.groups()
                        entries.append({
                            "timestamp": timestamp,
                            "level": level,
                            "message": message
                        })
        except FileNotFoundError:
            raise FileNotFoundError(f"Source log file not found: '{self.log_file}'. No logs to export.")
        except Exception as e:
            raise IOError(f"Error reading source log file '{self.log_file}' for export: {e}")

        # Ensure the output directory for the exported file exists.
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except OSError as e:
                raise OSError(f"Failed to create output directory '{output_dir}': {e}")

        # Write the parsed entries to the specified output format.
        try:
            if format == 'csv':
                with open(output_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=["timestamp", "level", "message"])
                    writer.writeheader()
                    writer.writerows(entries)
            elif format == 'json':
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(entries, f, indent=4)
            elif format == 'txt':
                with open(output_path, 'w', encoding='utf-8') as f:
                    for entry in entries:
                        f.write(f"[{entry['timestamp']}] [{entry['level']}] {entry['message']}\n")
            else:
                raise ValueError("Unsupported export format. Choose 'csv', 'json', or 'txt'.")
        except Exception as e:
            raise IOError(f"Error exporting logs to '{output_path}' in '{format}' format: {e}")

# --- Global Logger Instance (Factory Function) ---

# This function provides a central point to get the logger instance.
# It uses the Singleton pattern of SimpleLogger to ensure consistency.
def get_logger(log_file: str = None) -> SimpleLogger:
    """
    Returns a singleton instance of the SimpleLogger.
    If called for the first time, it will initialize the logger.
    Subsequent calls will return the same instance.

    Args:
        log_file (str, optional): The path to the log file. This parameter
                                  is primarily used during the first call
                                  to configure file logging.
    Returns:
        SimpleLogger: The singleton logger instance.
    """
    return SimpleLogger(log_file)