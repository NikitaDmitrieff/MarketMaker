import numpy as np
from abc import abstractmethod


class MarketMaker:
    def __init__(self, tick_size=None, std=2, mid_price=50):
        self.tick_size = tick_size

        # PnL
        self.pnl = 0
        self.pnl_history = [0]

        # Assets
        self.assets = 0
        self.assets_history = [0]
        self.maximum_asset = -float("inf")
        self.minimum_asset = float("inf")

        # Price parameters
        self.standard_deviation = std
        self.mid_price = mid_price

        # Stats to update
        self.buying_price_history = np.array([])
        self.selling_price_history = np.array([])
        self.price_history = np.array([])
        self.transactions = []

    def reset(self):
        """
        Resets the statistics of the Market Maker
        """
        # PnL
        self.pnl = 0
        self.pnl_history = [0]

        # Assets
        self.assets = 0
        self.assets_history = [0]
        self.maximum_asset = -float("inf")
        self.minimum_asset = float("inf")

        # Stats to update
        self.buying_price_history = np.array([])
        self.selling_price_history = np.array([])
        self.price_history = np.array([])
        self.transactions = []

    def update_stats(self, number_of_stocks, traders):
        """
        Actualize the stats of the Market Maker based on the number of
            stocks bought. A negative number corresponds to stocks being
            sold.

        Args:
            number_of_stocks: number of stocks exchanged or list of number of
                stocks exchanged

        Returns:
            None
        """

        if self.buying_price_history.shape[0] == 0:
            raise ValueError

        transaction = 0

        for index, stocks in enumerate(number_of_stocks):

            if stocks <= 0:
                # From the MM's point of vue, it sold stocks at the selling price
                self.pnl -= stocks * self.selling_price_history[-1]
                transaction -= stocks * self.selling_price_history[-1]
            else:
                # From the MM's point of vue, it bought stocks at the buying price
                self.pnl -= stocks * self.buying_price_history[-1]
                transaction -= stocks * self.buying_price_history[-1]

        self.pnl_history.append(self.pnl)
        self.transactions.append(transaction)

        self.assets = 0
        for trader in traders:
            self.assets -= trader.assets - trader.initial_balance
        self.assets_history.append(self.assets)

        if self.assets < self.minimum_asset:
            self.minimum_asset = self.assets
        elif self.assets > self.maximum_asset:
            self.maximum_asset = self.assets
        else:
            pass

        return

    @abstractmethod
    def set_prices(self):
        """
        This method is used to set buying and selling prices. It is
            called at every iteration.

        Returns:
            tuple[int, int]: (b, s) b is the buying price and s the
                selling price
        """
        return NotImplementedError

    @abstractmethod
    def set_mid_price(self):
        """
        This method sets the average price from which gaussian will be added
            to generate a buying and selling price. This method hence sets a
            general price trend for the simulation.

        Returns:
            float: mid price
        """

        return NotImplementedError


class GaussianMarketMaker(MarketMaker):
    def __init(self, tick_size=None, std=2, mid_price=50):
        super().__init__(tick_size=tick_size, std=std, mid_price=mid_price)

    def set_mid_price(self):
        """
        This method sets the average price from which gaussian will be added
            to generate a buying and selling price. This method hence sets a
            general price trend for the simulation.

        Returns:
            float: mid price
        """

        price = -1

        while price < 1:
            price = np.random.normal(self.mid_price, self.standard_deviation)

        return price

    def set_prices(self):
        """
        This method is used to set buying and selling prices. It is
            called at every iteration.

        Returns:
            tuple[int, int]: (b, s) b is the buying price and s the
                selling price
        """

        mid_price = self.set_mid_price()

        correlated_noise = False

        self.price_history = np.append(self.price_history, mid_price)

        buying_price = 1
        selling_price = 0
        count = 0

        while buying_price > selling_price or buying_price <= 0 or selling_price <= 0:

            count += 1

            if not correlated_noise:
                # Each variable is generated using 2 different gaussian noises
                buying_noise = abs(np.random.normal(0, self.standard_deviation))
                selling_noise = abs(np.random.normal(0, self.standard_deviation))
            else:
                noise = abs(np.random.normal(0, self.standard_deviation))
                buying_noise = noise
                selling_noise = noise

            if count > 10000:
                breakpoint()

            buying_price = mid_price - buying_noise
            selling_price = mid_price + selling_noise

        self.buying_price_history = np.append(self.buying_price_history, buying_price)
        self.selling_price_history = np.append(
            self.selling_price_history, selling_price
        )

        return buying_price, selling_price


class IntelligentMarketMaker(MarketMaker):
    def __init(self, tick_size=1, std=2, mid_price=50):
        super().__init__(tick_size=1, std=2, mid_price=50)

    def set_prices(self):
        return
