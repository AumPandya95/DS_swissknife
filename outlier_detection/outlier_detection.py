r"""
Notes
-----
- As α nears 1, a significant change in X t will result in a significant increase in μ t , and an even greater
  impact on σ t . The resultant model will have a poorer fit to the data, lessening the detection capabilities.

- As span increases, sensitivity to outliers increases too. However, this can result into over-fitting
"""

__all__ = ["EWMA"]

import numpy as np


class EWMA:
    r"""
    Parameters
    ----------
    input_array : list
        Input data in a list

    span : float, optional
        Specify decay in terms of span, α=2/(span+1), for span≥1. Span of 1 would imply equal weightage to all
        historical values.

    alpha : float, optional
        Specify smoothing factor α directly, 0<α≤1. Higher α implies more weightage to older values while calculating
        the average.

    adjust : bool, default=False
        Divide by decaying adjustment factor in beginning periods to account for imbalance in relative weightings
        (viewing EWMA as a moving average).
        When adjust=True (default), the EW function is calculated using weights wi=(1−α)i. For example, the EW moving
        average of the series [x0,x1,...,xt] would be:
                    yt=xt+(1−α)xt−1+(1−α)2xt−2+...+(1−α)tx01+(1−α)+(1−α)2+...+(1−α)t
        When adjust=False, the exponentially weighted function is calculated recursively:
                    y0=x0
                    yt=(1−α)yt−1+αxt

    m : int, default=2
        Multiplier to the exponentially weighted standard deviations to set the threshold for identifying outliers
    """

    def __init__(self, **kwargs):
        self.input_array = kwargs.get("input_array")
        self.span = kwargs.get("span", None)
        self.alpha = kwargs.get("alpha", None)
        self.adjust = kwargs.get("adjust", True)
        self.multiplier = kwargs.get("m", None)

        if not self.span and not self.alpha:
            self.span = 15
            self.alpha = 2 / (self.span + 1)
        elif not self.alpha and self.span:
            self.alpha = 2 / (self.span + 1)

        if not self.multiplier:
            self.multiplier = 2

    def get_outliers(self) -> list:
        """Return the detected outliers index.

        Returns
        -------
        outlier_indexes: list
            The indexes at which the outliers are present in the input array
        """
        import pandas as pd

        input_array = pd.Series(self.input_array)
        exponentially_weighed_average = input_array.ewm(
            span=self.span, adjust=self.adjust
        ).mean()
        exponentially_weighed_standard_deviation = input_array.ewm(
            span=self.span, adjust=self.adjust
        ).std()

        # Comparing the difference between original and predicted values with the product of multiplier and
        # exponentially weighed standard deviations
        moving_sd = list(
            map(
                float,
                exponentially_weighed_standard_deviation.values[
                    0 : len(input_array) - 1
                ],
            )
        )
        moving_average = list(
            map(float, exponentially_weighed_average.values[0 : len(input_array) - 1])
        )
        outlier_series = np.divide(
            abs(np.array(self.input_array[1:]) - moving_average), moving_sd
        )
        # If element in the outlier_series more than the multiplier then it is an outlier
        outlier_indexes = [
            _index + 1
            for _index, value in enumerate(outlier_series)
            if value > self.multiplier
        ]

        return outlier_indexes


class PEWMA:
    r"""
    Parameters
    ----------
    input_array : list
        Input data in a list

    span : float, optional
        Specify decay in terms of span, α=2/(span+1), for span≥1. Span of 1 would imply equal weightage to all
        historical values.

    alpha : float, optional
        Specify smoothing factor α directly, 0<α≤1. Higher α implies more weightage to older values while calculating
        the average.

    beta : float, optional
        Specify roughing factor β directly, 0<β≤1. Parameter that controls how much you allow outliers to affect your
        MA, for standard EWMA set to 0.

    adjust : bool, default=False
        Divide by decaying adjustment factor in beginning periods to account for imbalance in relative weightings
        (viewing EWMA as a moving average).
        When adjust=True (default), the EW function is calculated using weights wi=(1−α)i. For example, the EW moving
        average of the series [x0,x1,...,xt] would be:
                    yt=xt+(1−α)xt−1+(1−α)2xt−2+...+(1−α)tx01+(1−α)+(1−α)2+...+(1−α)t
        When adjust=False, the exponentially weighted function is calculated recursively:
                    y0=x0
                    yt=(1−α)yt−1+αxt

    m : int, default=2
        Multiplier to the exponentially weighted standard deviations to set the threshold for identifying outliers
    """

    def __init__(self, **kwargs):
        self.input_array = kwargs.get("input_array")
        self.span = kwargs.get("span", None)
        self.beta = kwargs.get("beta", None)
        self.adjust = kwargs.get("adjust", None)
        self.multiplier = kwargs.get("m", None)

        if not self.span:
            self.span = 10

        self.alpha = 2 / (self.span + 1)

        if not self.beta:
            self.beta = 1

        if not self.multiplier:
            self.multiplier = 2

    def get_outliers(self) -> list:
        """Return the detected outliers index.

        Returns
        -------
        outlier_indexes: list
            The indexes at which the outliers are present in the input array
        """
        from scipy.stats import norm

        time_window = 0
        probabilistic_exponential_moving_average = []
        probabilistic_exponentially_weighed_standard_deviation = []
        _s2 = []

        for i in range(len(self.input_array)):
            if i == 0:
                s1 = self.input_array[0]
                s2 = self.input_array[0] ** 2
                standard_deviation = np.sqrt(s2 - s1**2)
                time_window += 1
            else:
                if time_window <= self.span:
                    alpha_t = 1 - (1 / time_window)
                    if i == 1:
                        s1 = self.input_array[i - 1]
                        s2 = self.input_array[i - 1] ** 2
                    else:
                        s1 = ((1 - alpha_t) * self.input_array[i - 1]) + (
                            alpha_t * probabilistic_exponential_moving_average[i - 1]
                        )
                        s2 = ((1 - alpha_t) * self.input_array[i - 1] ** 2) + (
                            alpha_t * _s2[i - 1]
                        )
                    standard_deviation = np.sqrt(s2 - s1**2)
                    time_window += 1
                else:
                    _std = probabilistic_exponentially_weighed_standard_deviation[i - 1]
                    z = (
                        self.input_array[i - 1]
                        - probabilistic_exponential_moving_average[i - 1]
                    )
                    probability = norm.pdf(z / _std) if _std not in [np.nan, 0] else 0
                    alpha_t = (1 - self.beta * probability) * (1 - self.alpha)
                    s1 = ((1 - alpha_t) * self.input_array[i - 1]) + (
                        alpha_t * probabilistic_exponential_moving_average[i - 1]
                    )
                    s2 = ((1 - alpha_t) * self.input_array[i - 1] ** 2) + (
                        alpha_t * _s2[i - 1]
                    )
                    standard_deviation = np.sqrt(s2 - s1**2)
                    time_window += 1
            probabilistic_exponential_moving_average.append(s1)
            probabilistic_exponentially_weighed_standard_deviation.append(
                standard_deviation
            )
            _s2.append(s2)

        # Comparing the difference between original and predicted values with the product of multiplier and
        # exponentially weighed standard deviations
        moving_sd = np.array(probabilistic_exponentially_weighed_standard_deviation)
        outlier_series = np.divide(
            abs(
                np.array(self.input_array)
                - np.array(probabilistic_exponential_moving_average)
            ),
            moving_sd,
        )

        # If element in the outlier_series more than the multiplier then it is an outlier
        outlier_indexes = [
            _index
            for _index, value in enumerate(outlier_series)
            if value > self.multiplier
        ]

        return outlier_indexes
