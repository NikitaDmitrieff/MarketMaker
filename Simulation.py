from MarketMaker import GaussianMarketMaker
from Traders import EvenTrader, IntelligentTrader, RandomTrader
import matplotlib.pyplot as plt
import numpy as np


class Simulation:
    def __init__(self, traders, market_maker, number_of_iteration=300):

        self.number_of_iterations = number_of_iteration
        self.traders = traders
        self.market_maker = market_maker
        self.balance = 0

        return

    def reset(self):
        """
        Resets stats, useful in case multiple runs are done.
        """
        self.balance = 0

        for trader in self.traders:
            trader.reset()

        self.market_maker.reset()

        return

    def run(self):
        """
        Runs a full simulation

        Returns:
            None
        """

        self.reset()

        for iteration_number in range(1, self.number_of_iterations + 1):

            buying_price_of_MM, selling_price_of_MM = self.market_maker.set_prices()
            assert buying_price_of_MM <= selling_price_of_MM, breakpoint()

            list_of_stocks_exchanged = []
            for trader in self.traders:

                # Here, the buying price and selling price invert: it's a matter of perspective
                # In other words, the MM's buying price is the Trader's selling price.
                number_of_stocks = trader.trade(
                    buying_price=selling_price_of_MM,
                    selling_price=buying_price_of_MM,
                    iteration_number=iteration_number,
                )

                list_of_stocks_exchanged.append(number_of_stocks)

                if len(trader.pnl_history) > 1:

                    old_pnl, new_pnl = trader.pnl_history[-2], trader.pnl_history[-1]
                    difference = round(new_pnl - old_pnl)

                    # An un expected error rises when this particular price is set
                    # due to the fact that, multiplied by 7, equals exactly 180.0
                    # although the difference might be of 179.99. In other words,
                    # rounding errors mislead us.
                    if buying_price_of_MM != 25.714285714285715:
                        if number_of_stocks < 0:  # Sold some stocks
                            assert (
                                round(number_of_stocks * selling_price_of_MM)
                                == difference
                            ), breakpoint()
                        else:
                            assert (
                                round(number_of_stocks * buying_price_of_MM)
                                == difference
                            ), breakpoint()

            self.market_maker.update_stats(list_of_stocks_exchanged, self.traders)
            # self.display_current_status(iteration_number)

        self.sanity_check()

        return self.gather_results()

    def gather_results(self):
        """
        Gathers the results of the simulation
        """

        trader_assets_stats = []
        vol_of_mm = 0
        assets_history = []

        for trader in self.traders:
            assets_array = np.array(trader.assets_history)
            avg = np.mean(assets_array)
            std = np.std(assets_array)
            vol = trader.volumes
            trader_assets_stats.append(
                (trader.minimum_asset, trader.maximum_asset, avg, std, vol)
            )
            vol_of_mm += vol
            assets_history.append(assets_array)

        market_maker_assets_stats = []

        assets_array = np.array(self.market_maker.assets_history)
        avg = np.mean(assets_array)
        std = np.std(assets_array)
        market_maker_assets_stats.append(
            (
                self.market_maker.minimum_asset,
                self.market_maker.maximum_asset,
                avg,
                std,
                vol_of_mm,
            )
        )
        assets_history.append(assets_array)

        return trader_assets_stats, market_maker_assets_stats, assets_history

    def display_current_status(self, iteration_number):
        """
        Displays the status of the system
        """
        print("\nStarting iteration number", iteration_number)
        print(f"MARKET MAKER PNL ----> {self.market_maker.pnl}")

        for index, trader in enumerate(self.traders):
            print(f"TRADER {index} PNL --------> {trader.pnl}")

        return

    def display_results(
        self,
        detailed_PnL=True,
        detailed_running_PnL=False,
        price_history=True,
        transaction_history=False,
        assets_history=True,
    ):
        """
        Method to help visualize the results.

        Args:
            detailed_PnL: Display breakdown of the PnL in the system
            detailed_running_PnL: Display breakdown of the running PnL in the system
            price_history: Display the stock price history
            transaction_history: Display the transactions occurring in the system
            assets_history: Display assets (always positive)
        """

        plt.rcParams.update(plt.rcParamsDefault)

        if transaction_history:
            self.display_transactions()

        if price_history:
            plt.figure(figsize=(15, 8))

            ymin = 0
            ymax = self.market_maker.mid_price * 1.3
            plt.ylim(ymin, ymax)

            plt.plot(self.market_maker.price_history, label="Price History")
            plt.fill_between(
                range(len(self.market_maker.price_history)),
                self.market_maker.price_history,
                self.market_maker.buying_price_history,
                alpha=0.2,
                color="b",
            )
            plt.fill_between(
                range(len(self.market_maker.price_history)),
                self.market_maker.price_history,
                self.market_maker.selling_price_history,
                alpha=0.2,
                color="r",
            )

            title_name = "Stock Price during the simulation"
            plt.title(title_name, fontsize=20, fontweight="bold")

            plt.xlabel("Iteration number", fontsize=16, fontweight="bold")
            plt.ylabel("Price", fontsize=17, fontweight="bold")

            plt.grid()
            plt.legend()
            plt.show()

        if detailed_PnL:
            plt.figure(figsize=(15, 8))
            plt.plot(self.market_maker.pnl_history, label="Market Maker's PNL")
            plt.legend()

            for index, trader in enumerate(self.traders):
                plt.plot(trader.pnl_history, label=f"{trader.name}'s PNL")
                plt.legend()

            title_name = "Detailed Profit and Losses (PnL) of the agents in the system"
            plt.title(title_name, fontsize=20, fontweight="bold")

            plt.xlabel("Iteration number", fontsize=15, fontweight="bold")
            plt.ylabel("Profit and Losses", fontsize=15, fontweight="bold")

            plt.grid()

            print("\nDisclaimer!\n")
            print(
                "The following graph shows the PnL: it considers buying stock as decrease in profits and selling"
                "stocks as increase in profits, as required."
            )

            plt.show()

            print(
                "This plot is confusing and a better metric should be used to accurately display the system's status."
                " Therefore, assets were plotted. Assets can be either cash or stocks. When it takes the form of"
                " stocks, it updates its value at each selling point."
            )

        if detailed_running_PnL:
            plt.figure(figsize=(15, 8))
            plt.plot(self.market_maker.pnl_history, label="Market Maker's PNL")
            plt.legend()

            plt.plot(
                [0, 300],
                [1000, 1000],
                "r--",
                linewidth=2,
                label="Break even for Traders",
            )
            plt.plot(
                [0, 300],
                [0, 0],
                "b--",
                linewidth=2,
                label="Break even for Market Maker",
            )

            for index, trader in enumerate(self.traders):
                plt.plot(trader.running_pnl, label=f"{trader.name} running PnL")
                plt.legend()

            title_name = (
                "Detailed averaged Profit and Losses (PnL) of the agents in the system"
            )
            plt.title(title_name, fontsize=20, fontweight="bold")

            plt.xlabel("Iteration number", fontsize=15, fontweight="bold")
            plt.ylabel("Running Profit and Losses", fontsize=15, fontweight="bold")

            plt.grid()
            plt.show()

        if assets_history:

            plt.figure(figsize=(15, 8))
            plt.plot(
                self.market_maker.assets_history,
                linewidth=2.5,
                label="Market Maker's assets",
            )
            plt.legend()

            max_asset = self.market_maker.maximum_asset
            min_asset = self.market_maker.minimum_asset

            for index, trader in enumerate(self.traders):
                plt.plot(
                    trader.assets_history,
                    linewidth=2.5,
                    label=f"{trader.name}'s assets",
                )
                plt.legend()
                max_asset = max(max_asset, trader.maximum_asset)
                min_asset = min(min_asset, trader.minimum_asset)

            if min_asset < 0:
                ymin = min_asset * 1.1
            else:
                ymin = min_asset * 0.9

            ymax = max_asset * 1.1
            plt.ylim(ymin, ymax)

            xmin = -5
            xmax = len(trader.assets_history) + 10
            plt.xlim(xmin, xmax)

            plt.plot(
                [-1000, len(trader.assets_history) + 1000],
                [1000, 1000],
                "r--",
                linewidth=1.5,
                label="Break even for Traders",
            )
            # plt.plot([-10, len(trader.assets_history) + 10], [0, 0], "b--", linewidth=3, label="Break even for Market Maker")
            plt.legend()

            plt.fill_between(
                np.arange(-1000, len(self.market_maker.price_history) + 1000),
                1000,
                1000 + 10000,
                alpha=0.15,
                color="g",
            )
            plt.fill_between(
                np.arange(-1000, len(self.market_maker.price_history) + 1000),
                1000,
                1000 - 10000,
                alpha=0.15,
                color="r",
            )

            title_name = "Detailed Asset history of the agents in the system"
            plt.title(title_name, fontsize=20, fontweight="bold")

            plt.xlabel("Iteration number", fontsize=15, fontweight="bold")
            plt.ylabel("Assets", fontsize=15, fontweight="bold")

            plt.grid()
            plt.show()

        return

    def display_transactions(self):
        plt.figure(figsize=(14, 11))
        plt.plot(self.market_maker.transactions, label="Market Maker's transactions")
        plt.legend()

        for index, trader in enumerate(traders):
            plt.plot(trader.transactions, label=f"{trader.name}'s transactions")
            plt.legend()

        title_name = "Detailed transactions of the agents in the system"
        plt.title(title_name, fontsize=20, fontweight="bold")
        plt.grid()
        plt.show()
        return

    def sanity_check(self, display=False):
        """
        Money cannot be created. Balance between all traders and market maker should
            be constant. This is verified here.
        """

        summed_list = []

        assert len(self.traders[0].pnl_history) == len(self.market_maker.pnl_history)

        for index in range(len(self.traders[0].pnl_history)):
            total_traders = 0
            for trader in self.traders:
                total_traders += trader.pnl_history[index]

            assert int(total_traders + self.market_maker.pnl_history[index]) == 0, (
                "Balance Error: " "Money must come from somewhere."
            )

            summed_list.append(
                int(total_traders + self.market_maker.pnl_history[index])
            )

        if display:
            plt.figure(figsize=(15, 8))
            plt.plot(
                summed_list,
                linewidth=3,
                label="Sum of the PNL of all agents in the system",
            )
            plt.legend()
            plt.grid()
            title_name = "Sum of the Profit and Losses of all agents in the system should be 0 at all times"
            plt.title(title_name, fontsize=20, fontweight="bold")
            plt.show()

        return


if __name__ == "__main__":
    # Create Traders
    my_even_trader = EvenTrader(1000)
    my_intelligent_trader = IntelligentTrader(1000)
    my_random_trader = RandomTrader(1000)
    traders = [my_even_trader, my_intelligent_trader, my_random_trader]

    # Create Market Maker
    my_market_maker = GaussianMarketMaker()

    # Create Simulation
    my_simulation = Simulation(traders, my_market_maker, number_of_iteration=500)

    # Run and Display results
    my_simulation.run()
    my_simulation.display_results()
