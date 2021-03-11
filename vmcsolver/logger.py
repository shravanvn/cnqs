__author__ = "Saibal De"

from datetime import datetime


class Logger(object):
    """Logging in plaintext."""

    def __init__(self, run_dir):
        """Create a log file inside run_dir."""
        self.file_name = run_dir + "/output.csv"
        self.var_names = None
        self.var_values = None

    def set_variables(self, var_names):
        self.var_names = var_names
        self.var_values = {var: 0.0 for var in var_names}

    def log_scalar(self, var, value):
        self.var_values[var] = value

    def write_header(self):
        header = "timestamp,step"
        for var in self.var_names:
            header += "," + var
        header += "\n"

        file_object = open(self.file_name, "w")
        file_object.write(header)
        file_object.close()

    def write_step(self, step):
        line = datetime.now().isoformat() + "," + str(step)
        for var in self.var_names:
            line += "," + "{:e}".format(self.var_values[var])
        line += "\n"

        file_object = open(self.file_name, "a")
        file_object.write(line)
        file_object.close()

    def write_stdout(self, var_names):
        line = ""
        for i in range(len(var_names)):
            if i == len(var_names) - 1:
                line += "{:s} = {:e}".format(var_names[i], self.var_values[var_names[i]])
            else:
                line += "{:s} = {:e}, ".format(var_names[i], self.var_values[var_names[i]])

        print(line)
