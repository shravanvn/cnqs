__author__ = "Saibal De"

from datetime import datetime


class Logger(object):
    """Logging in plaintext."""

    def __init__(self, log_dir):
        """Create a log file inside log_dir."""
        self.file_name = log_dir + "/" + datetime.now().replace(microsecond=0).isoformat() + ".csv"
        self.var_names = None
        self.var_values = None

    def set_variables(self, var_names):
        self.var_names = var_names
        self.var_values = {var: 0.0 for var in var_names}

    def log_scalar(self, var, value):
        self.var_values[var] = value

    def write_header(self):
        header = "step,timestamp"
        for var in self.var_names:
            header += "," + var
        header += "\n"

        file_object = open(self.file_name, "w")
        file_object.write(header)
        file_object.close()

    def write_step(self, step):
        line = str(step) + "," + datetime.now().isoformat()
        for var in self.var_names:
            line += "," + "{:e}".format(self.var_values[var])
        line += "\n"

        file_object = open(self.file_name, "a")
        file_object.write(line)
        file_object.close()
