from itertools import product


class GridSearch:
    """Generate every possible combination of parameter values."""

    def __init__(self, parameter_values):
        self.parameter_values = parameter_values

    def generate_candidates(self):
        """Return all parameter combinations as dictionaries."""

        parameter_names = list(self.parameter_values.keys())
        value_lists = list(self.parameter_values.values())

        candidates = []

        for combination in product(*value_lists):
            candidate = {}

            for index in range(len(parameter_names)):
                candidate[parameter_names[index]] = combination[index]

            candidates.append(candidate)

        return candidates