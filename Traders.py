from abc import abstractmethod
import numpy as np


class Trader:
    def __init__(self, initial_balance, name=None, trading_fee_percentage=0):

        self.trading_fee_percentage = trading_fee_percentage

        # PnL
        self.pnl = 0
        self.pnl_history = [0]
        self.running_pnl = [0]

        # Balance
        self.initial_balance = initial_balance
        self.balance = self.initial_balance

        # Stocks
        self.stocks = 0

        # Transactions
        self.transactions = []

        # Assets
        self.assets = self.initial_balance
        self.assets_history = [self.initial_balance]
        self.maximum_asset = -float("inf")
        self.minimum_asset = float("inf")

        # Volume traded
        self.volumes = 0

        # Name
        self.name = name

    def reset(self):
        """
        Resets the Trader's statistics
        """

        # PnL
        self.pnl = 0
        self.pnl_history = [0]
        self.running_pnl = [0]

        # Balance
        self.balance = self.initial_balance

        # Stocks
        self.stocks = 0

        # Transactions
        self.transactions = []

        # Assets
        self.assets = self.initial_balance
        self.assets_history = [self.initial_balance]
        self.maximum_asset = -float("inf")
        self.minimum_asset = float("inf")

        self.reset_special_attributes()

        return

    def buys(self, buying_price, number_of_stocks):
        """
        This method buys a certain number of stocks at the buying price set
            by the Market Maker's selling price. It hence updates the Trader's
            PnL, balance and stock number.

        Args:
            buying_price: Price at which the Trader buys stocks
            number_of_stocks: Number of stocks the Trader buys

        Returns:
            None
        """

        # Could add trading fee here
        trading_fee = self.trading_fee_percentage * buying_price * number_of_stocks

        self.pnl -= number_of_stocks * buying_price - trading_fee
        self.pnl_history.append(self.pnl)
        self.running_pnl.append(sum(self.pnl_history) / len(self.pnl_history))

        self.balance -= number_of_stocks * buying_price + trading_fee
        self.stocks += number_of_stocks

        self.assets_history.append(self.assets)

        self.transactions.append(-number_of_stocks * buying_price - trading_fee)

        assert self.balance >= 0, "Error: negative balance is not allowed"

        return

    def sells(self, selling_price, number_of_stocks):
        """
        This method sells a certain number of stocks at the selling price set
            by the Market Maker's selling price. It hence updates the Trader's
            PnL, balance and stock number.

        Args:
            selling_price: Price at which the Trader buys stocks
            number_of_stocks: Number of stocks the Trader buys

        Returns:
            None
        """

        trading_fee = self.trading_fee_percentage * selling_price * number_of_stocks

        self.pnl += number_of_stocks * selling_price - trading_fee
        self.pnl_history.append(self.pnl)
        self.assets = self.pnl + self.initial_balance
        self.assets_history.append(self.assets)
        self.running_pnl.append(sum(self.pnl_history) / len(self.pnl_history))
        self.balance += number_of_stocks * selling_price - trading_fee
        self.stocks -= number_of_stocks

        self.transactions.append(number_of_stocks * selling_price - trading_fee)

        if self.assets < self.minimum_asset:
            self.minimum_asset = self.assets
        elif self.assets > self.maximum_asset:
            self.maximum_asset = self.assets
        else:
            pass

        return

    def buying_process(self, buying_price):
        """
        Trader decides to buy stocks. This method initiates the buying process.

        Args:
            buying_price: Price at which the Trader can buy stocks

        Returns:
            int: number of stocks we are buying
        """

        number_of_stocks = self.should_buy(buying_price)
        self.buys(buying_price, number_of_stocks)
        self.volumes += number_of_stocks

        return number_of_stocks

    def selling_process(self, selling_price):
        """
        Trader decides to sell stocks. This method initiates the selling process.

        Args:
            selling_price: Price at which the Trader can sell stocks

        Returns:
            int: number of stocks we are buying (since we are selling, this should be negative)
        """

        number_of_stocks = self.should_sell(selling_price=selling_price)
        self.sells(selling_price, number_of_stocks)

        return -number_of_stocks

    def holding_process(self):
        """
        Trader decides to hold
        """
        self.pnl_history.append(self.pnl)
        self.running_pnl.append(sum(self.pnl_history) / len(self.pnl_history))
        self.assets_history.append(self.assets)
        self.transactions.append(0)

        return 0

    def can_buy(self, buying_price):
        """
        Gives the number of stocks the Trader can buy given his finances
            and the buying price.

        Args:
            buying_price: Price at which the Trader can buy stocks

        Returns:
            int: Maximum number of stocks the Trader can buy given his
                current finances
        """

        number_of_stocks = self.balance // buying_price

        count = 0
        while (
            self.balance
            - number_of_stocks * buying_price
            - self.trading_fee_percentage * number_of_stocks * buying_price
            < 0
        ):
            count += 1

            if count > 10000:
                breakpoint()

            number_of_stocks -= 1

        return number_of_stocks

    def should_sell(self, selling_price=None):
        """
        Gives the number of stocks the Trader should buy given his finances
            and the buying price.

        Args:
            selling_price: Price at which the Trader can sell stocks

        Returns:
            int: Maximum number of stocks the Trader can buy given his
                current finances
        """
        return self.stocks

    @abstractmethod
    def reset_special_attributes(self):
        """
        Resets attributes that are not common to all traders
        """
        return NotImplementedError

    @abstractmethod
    def should_buy(self, buying_price):
        """
        Gives the number of stocks the Trader should buy given his finances
            and the buying price.

        Args:
            buying_price: Price at which the Trader can buy stocks

        Returns:
            int: Maximum number of stocks the Trader can buy given his
                current finances
        """

        return NotImplementedError

    @abstractmethod
    def trade(self, buying_price, selling_price, iteration_number):
        """
        Defines the core of the Trader's strategy. This is an abstract method that
        will be different for each Trader.

        Returns:
            None
        """
        return NotImplementedError


class EvenTrader(Trader):
    def __init__(self, initial_balance=1000, name="Even Trader"):
        super().__init__(initial_balance=initial_balance, name=name)

    def should_buy(self, buying_price):
        return self.can_buy(buying_price=buying_price)

    def trade(self, buying_price, selling_price, iteration_number):
        """
        On even iterations, the Trader buys all that it can. On odd iterations,
            it sells all stocks it has.
        """

        if self.pnl_history[-1] >= -self.initial_balance:
            if iteration_number % 2:
                number_of_stocks = self.buying_process(buying_price=buying_price)
            else:
                number_of_stocks = self.selling_process(selling_price=selling_price)
        else:
            number_of_stocks = self.holding_process()

        return -number_of_stocks

    def reset_special_attributes(self):
        return


class IntelligentTrader(Trader):
    def __init__(self, initial_balance=1000, name="Intelligent Trader"):
        super().__init__(initial_balance=initial_balance, name=name)

        self.buying_price_history = np.array([])
        self.selling_price_history = np.array([])
        self.threshold_for_buying = np.array([])
        self.threshold_for_selling = np.array([])

    def should_buy(self, buying_price):
        """
        Calculates the number of stocks the Trader should buy given the
            buying_price and its finances.

        Args:
            buying_price: Price at which the Trader can buy stocks

        Returns:
            int: Maximum number of stocks the Trader can buy given his
                current finances
        """
        return self.can_buy(buying_price)

    def trade(self, buying_price, selling_price, iteration_number):
        """
        Constitutes the core of the Trader's strategy. It decides what to do, given
            information on the system.

        Args:
            buying_price: Price at which the Trader can buy stocks
            selling_price: Price at which the Trader can sell stocks
            iteration_number: Iteration number

        Returns:
            None
        """

        number_of_stocks = 0

        assert buying_price >= selling_price, (
            "The buying price for a Trader should always"
            "be higher than the selling price"
        )

        self.buying_price_history = np.append(self.buying_price_history, buying_price)
        self.selling_price_history = np.append(
            self.selling_price_history, selling_price
        )

        if iteration_number < 2:
            number_of_stocks = 0

            self.pnl = 0
            self.pnl_history.append(self.pnl)
            self.running_pnl.append(0)

            self.transactions.append(0)

        else:
            if self.pnl_history[-1] >= -self.initial_balance:
                threshold_for_buying = np.mean(self.buying_price_history) - np.std(
                    self.buying_price_history
                ) * (1 / iteration_number)
                np.append(self.threshold_for_buying, threshold_for_buying)
                threshold_for_selling = np.mean(self.selling_price_history) - np.std(
                    self.selling_price_history
                ) * (1 / iteration_number)
                np.append(self.threshold_for_selling, threshold_for_selling)

                if (
                    threshold_for_buying > buying_price
                    and not threshold_for_selling < selling_price
                ):
                    number_of_stocks = self.buying_process(buying_price=buying_price)

                elif (
                    not threshold_for_buying > buying_price
                    and threshold_for_selling < selling_price
                ):
                    number_of_stocks = self.selling_process(selling_price=selling_price)

                else:
                    if abs(threshold_for_buying - buying_price) > abs(
                        threshold_for_selling < selling_price
                    ):
                        number_of_stocks = self.buying_process(
                            buying_price=buying_price
                        )
                    else:
                        number_of_stocks = self.selling_process(
                            selling_price=selling_price
                        )
            else:
                number_of_stocks = self.holding_process()

        return -number_of_stocks

    def reset_special_attributes(self):

        self.buying_price_history = np.array([])
        self.selling_price_history = np.array([])
        self.threshold_for_buying = np.array([])
        self.threshold_for_selling = np.array([])

        return


class RandomTrader(Trader):
    def __init__(self, initial_balance, name="Random Trader"):
        super().__init__(initial_balance=initial_balance, name=name)

    def should_buy(self, buying_price):
        """
        Calculates the number of stocks the Trader should buy given the
            buying_price and its finances.

        Args:
            buying_price: Price at which the Trader can buy stocks

        Returns:
            int: Maximum number of stocks the Trader can buy given his
                current finances
        """

        number_of_stocks = self.can_buy(buying_price=buying_price)

        if number_of_stocks > 1:
            number_of_stocks = np.random.randint(1, number_of_stocks)

        return number_of_stocks

    def should_sell(self, selling_price=None):
        """
        Returns a quantity of stocks to sell, randomly chosen
        """
        if self.stocks > 1:
            number_of_stocks = np.random.randint(1, self.stocks + 1)
        else:
            number_of_stocks = self.stocks

        return number_of_stocks

    def trade(self, buying_price, selling_price, iteration_number):
        """
        Constitutes the core of the Trader's strategy. It decides what to do, given
            information on the system.

        Args:
            buying_price: Price at which the Trader can buy stocks
            selling_price: Price at which the Trader can sell stocks
            iteration_number: Iteration number

        Returns:
            None
        """
        number_of_stocks = 0

        if self.pnl_history[-1] >= -self.initial_balance:
            random_decision = np.random.randint(0, 3)

            buys = random_decision == 0
            sells = random_decision == 1
            holds = random_decision == 2

            if buys:
                number_of_stocks = self.buying_process(buying_price=buying_price)
            elif sells:
                number_of_stocks = self.selling_process(selling_price=selling_price)
            elif holds:
                number_of_stocks = self.holding_process()
        else:
            number_of_stocks = self.holding_process()

        return -number_of_stocks

    def reset_special_attributes(self):
        return


class TemplateTrader(Trader):
    """
    Class to create new trader
    """

    def __init__(self, initial_balance, name="??? Trader"):
        super().__init__(initial_balance=initial_balance, name=name)

    def should_buy(self, buying_price):
        """
        Calculates the number of stocks the Trader should buy given the
            buying_price and its finances.

        Args:
            buying_price: Price at which the Trader can buy stocks

        Returns:
            int: Maximum number of stocks the Trader can buy given his
                current finances
        """

        number_of_stocks = None

        return number_of_stocks

    def trade(self, buying_price, selling_price, iteration_number):
        """
        Constitutes the core of the Trader's strategy. It decides what to do, given
            information on the system.

        Args:
            buying_price: Price at which the Trader can buy stocks
            selling_price: Price at which the Trader can sell stocks
            iteration_number: Iteration number

        Returns:
            None
        """

        return

    def reset_special_attributes(self):
        return
