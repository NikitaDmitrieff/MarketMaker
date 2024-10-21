from MarketMaker import GaussianMarketMaker
from Traders import EvenTrader, IntelligentTrader, RandomTrader
from Simulation import Simulation
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd
import numpy as np
from tqdm import tqdm


class RunMultipleSimulations(Simulation):
    def __init__(
        self, traders, market_maker, number_of_iteration=300, number_of_runs=100
    ):

        super().__init__(traders, market_maker, number_of_iteration=number_of_iteration)
        self.number_of_runs = number_of_runs
        self.names = []

        for trader in self.traders:
            self.names.append(trader.name)

        self.concatenated_results = pd.DataFrame(columns=self.names)
        self.concatenated_std = pd.DataFrame(columns=self.names)

        self.results_min_assets = {}
        self.results_max_assets = {}
        self.results_avg_assets = {}
        self.results_std_assets = {}
        self.volume_traded = {}
        self.results_assets_history = {}

        self.results = {}

        for name in self.names:
            try:
                self.results[name] = pd.DataFrame(
                    columns=[
                        "Minimum Asset",
                        "Maximum Asset",
                        "Average Asset",
                        "Standard Deviation Asset",
                    ]
                )
            except ValueError:
                breakpoint()
            self.results_min_assets[name] = []
            self.results_max_assets[name] = []
            self.results_avg_assets[name] = []
            self.results_std_assets[name] = []
            self.volume_traded[name] = []
            self.results_assets_history[name] = []

        self.results["Market Maker"] = pd.DataFrame(
            columns=[
                "Minimum Asset",
                "Maximum Asset",
                "Average Asset",
                "Standard Deviation Asset",
            ]
        )

        self.results_min_assets["Market Maker"] = []
        self.results_max_assets["Market Maker"] = []
        self.results_avg_assets["Market Maker"] = []
        self.results_std_assets["Market Maker"] = []

    def process_simulation_results(self):
        """
        Process the results of the simulation.
        """
        self.display_concatenated_history_of_assets(display=False)

        # Create a Pandas Excel writer using XlsxWriter as the engine.
        writer = pd.ExcelWriter("Outputs.xlsx", engine="xlsxwriter")

        # Write each dataframe to a different worksheet.
        for name in self.names:
            self.results[name].to_excel(writer, sheet_name=f"Stats {name}")

        self.results["Market Maker"].to_excel(
            writer, sheet_name="Statistiques du Market Maker"
        )
        self.concatenated_results.to_excel(
            writer, sheet_name="Historique actifs agreges (moy)"
        )
        self.concatenated_std.to_excel(
            writer, sheet_name="Historique actifs agreges (e-t)"
        )

        # Close the Pandas Excel writer and output the Excel file.
        writer.close()

        return

    def run_multiple_simulations(self):
        """
        Runs the simulations and gathers all results.
        """

        for run_number in range(self.number_of_runs):
            results = self.run()
            for index, name in enumerate(self.names):
                self.results_min_assets[name].append(results[0][index][0])
                self.results_max_assets[name].append(results[0][index][1])
                self.results_avg_assets[name].append(results[0][index][2])
                self.results_std_assets[name].append(results[0][index][3])
                self.volume_traded[name].append(results[0][index][4])
                self.results_assets_history[name].append(results[2][index])

            self.results_min_assets["Market Maker"].append(results[1][0][0])
            self.results_max_assets["Market Maker"].append(results[1][0][1])
            self.results_avg_assets["Market Maker"].append(results[1][0][2])
            self.results_std_assets["Market Maker"].append(results[1][0][3])

        self.results["Market Maker"]["Minimum Asset"] = self.results_min_assets[
            "Market Maker"
        ]
        self.results["Market Maker"]["Maximum Asset"] = self.results_max_assets[
            "Market Maker"
        ]
        self.results["Market Maker"]["Average Asset"] = self.results_avg_assets[
            "Market Maker"
        ]
        self.results["Market Maker"][
            "Standard Deviation Asset"
        ] = self.results_std_assets["Market Maker"]

        for name in self.names:
            try:
                self.results[name]["Minimum Asset"] = self.results_min_assets[name]
                self.results[name]["Maximum Asset"] = self.results_max_assets[name]
                self.results[name]["Average Asset"] = self.results_avg_assets[name]
                self.results[name][
                    "Standard Deviation Asset"
                ] = self.results_std_assets[name]
            except ValueError:
                breakpoint()

        return

    def display_concatenated_history_of_assets(self, display=True):

        complete_matrix = {}

        for index, name in enumerate(self.names):
            matrix_history = np.array([])

            for index2, array_to_append in enumerate(self.results_assets_history[name]):
                if matrix_history.shape[0] == 0:
                    matrix_history = array_to_append
                else:
                    try:
                        matrix_history = np.vstack((matrix_history, array_to_append))
                    except ValueError:
                        matrix_history = np.vstack(
                            (matrix_history, array_to_append[:-1])
                        )

            complete_matrix[name] = matrix_history

        if display:
            plt.figure(figsize=(15, 8))

            ymin = float("inf")
            ymax = -float("inf")

            colours = ["b", "g", "r", "c", "m"]

            for index, name in enumerate(self.names):

                concat_asset_history_mean = np.mean(complete_matrix[name], axis=1)
                concat_asset_history_std = np.std(complete_matrix[name], axis=1)

                argmax = np.argmax(concat_asset_history_mean)
                argmin = np.argmin(concat_asset_history_mean)

                ymin = min(
                    concat_asset_history_mean[argmin]
                    - concat_asset_history_std[argmin]
                    - 20,
                    0,
                    ymin,
                )
                ymax = max(
                    (
                        concat_asset_history_mean[argmax]
                        + concat_asset_history_std[argmax]
                    )
                    * 1.1,
                    ymax,
                )
                plt.ylim(ymin, ymax)

                plt.plot(
                    concat_asset_history_mean,
                    label=f"Asset History of the {name}",
                    color=colours[index],
                    linewidth=1.8,
                )
                plt.fill_between(
                    range(len(concat_asset_history_mean)),
                    concat_asset_history_mean,
                    concat_asset_history_mean + concat_asset_history_std,
                    alpha=0.25,
                    color=colours[index],
                )
                plt.fill_between(
                    range(len(concat_asset_history_mean)),
                    concat_asset_history_mean,
                    concat_asset_history_mean - concat_asset_history_std,
                    alpha=0.25,
                    color=colours[index],
                )
                plt.legend()

            plt.plot(
                [0, len(concat_asset_history_mean)],
                [1000, 1000],
                "r--",
                linewidth=2.5,
                label="Break even for Traders",
            )
            title_name = f"Mean Asset history over {self.number_of_runs} simulations"
            plt.title(title_name, fontsize=20, fontweight="bold")

            plt.xlabel("Iteration number", fontsize=16, fontweight="bold")
            plt.ylabel("Assets", fontsize=17, fontweight="bold")

            plt.grid()
            plt.legend()
            plt.show()
            return
        else:
            for index, name in enumerate(self.names):
                self.concatenated_results[name] = np.mean(complete_matrix[name], axis=1)
                self.concatenated_std[name] = np.std(complete_matrix[name], axis=1)

            print(self.concatenated_results)
            print(self.concatenated_std)
            return

    def display_multiple_complete_results(
        self,
        views=(
            ((13, 100), (13, 125), (13, 150), (13, 170)),
            ((31, 53), (1, 90), (-91, 180), (0, 179)),
        ),
    ):
        """
        Displays multiple 3D views of the results
        """
        for view in views:
            self.display_multiple_complete_results_inner_function(views=view)

        return

    def display_multiple_complete_results_inner_function(
        self, views=((13, 100), (13, 125), (13, 150), (13, 170))
    ):
        """
        Displays multiple views of the 3D plot
        """
        fig = plt.figure(figsize=(16, 15))
        ax1 = fig.add_subplot(2, 2, 1, projection="3d")
        ax2 = fig.add_subplot(2, 2, 2, projection="3d")
        ax3 = fig.add_subplot(2, 2, 3, projection="3d")
        ax4 = fig.add_subplot(2, 2, 4, projection="3d")

        axs = [ax1, ax2, ax3, ax4]

        for index, ax in enumerate(axs):

            ax.set_xlabel("Volume Traded", fontweight="bold", fontsize=10)
            ax.set_ylabel("Standard Deviation", fontweight="bold", fontsize=10)
            ax.set_zlabel("Average Assets", fontweight="bold", fontsize=10)

            markers = ["o", "s", "x", ">"]

            for index2, name in enumerate(self.names):
                ax.scatter3D(
                    self.volume_traded[name],
                    self.results_std_assets[name],
                    self.results_avg_assets[name],
                    marker=markers[index2],
                    s=15,
                )

            ax.legend(
                self.names,
                title=f"Different Trading Strategies",
                loc="best",
                prop={"size": 8},
                fontsize=9,
            )

            elevation = views[index][0]
            azim = views[index][1]

            ax.grid(True)
            ax.view_init(elev=elevation, azim=azim)

            # plt.title(f'Results over {self.number_of_runs} Runs', fontweight='bold', fontsize=11)

        plt.show()

    def display_complete_results(self, elevation=13, azim=136, figsize=(5, 4)):
        """
        Displays the results of multiple simulations.
        """

        plt.figure(figsize=figsize)
        ax = plt.axes(projection="3d")
        ax.set_xlabel("Volume Traded", fontweight="bold", fontsize=15)
        ax.set_ylabel("Standard Deviation", fontweight="bold", fontsize=15)
        ax.set_zlabel("Average Assets", fontweight="bold", fontsize=15)

        markers = ["o", "s", "x", ">"]

        for index, name in enumerate(self.names):
            ax.scatter3D(
                self.volume_traded[name],
                self.results_std_assets[name],
                self.results_avg_assets[name],
                marker=markers[index],
                s=40,
            )

        ax.legend(
            self.names,
            title=f"Different Trading Strategies",
            loc="best",
            prop={"size": 12},
            fontsize=23,
        )

        ax.grid(True)
        ax.view_init(elev=elevation, azim=azim)

        plt.title(
            f"Results over {self.number_of_runs} Runs", fontweight="bold", fontsize=23
        )

        plt.show()

        return

    def display_price_slice(
        self,
        max_result=False,
        min_result=False,
        average_result=True,
        std_result=False,
        average_result_log=False,
        nb_of_replicates=20,
        avg_samples=100,
        std_samples=100,
        cap=5000,
    ):
        """
        Understand what characteristics are best for the Market Maker
        """

        plt.rcParams.update(plt.rcParamsDefault)

        avg_min = 10
        avg_max = 60
        std_min = 0
        std_max = 5

        avgs = np.linspace(avg_min, avg_max, avg_samples)
        stds = np.linspace(std_min, std_max, std_samples)

        results_array = np.zeros((avg_samples, std_samples))

        for i, avg in tqdm(enumerate(avgs)):
            for j, std in enumerate(stds):

                self.market_maker = GaussianMarketMaker(std=std, mid_price=avg)

                res = []
                for rep in range(nb_of_replicates):
                    res.append(self.run())

                results_mm_min = 0
                results_mm_max = 0
                results_mm_avg = 0
                results_mm_std = 0

                # res = [ [[ (min, max, avg, std), ..., ], [(min, max, avg, std)]],
                #         [[ (min, max, avg, std), ..., ], [(min, max, avg, std)]] ]
                for item in res:
                    results_mm_min += item[1][0][0] / len(res)
                    results_mm_max += item[1][0][1] / len(res)
                    results_mm_avg += item[1][0][2] / len(res)
                    results_mm_std += item[1][0][3] / len(res)

                if average_result:
                    if results_mm_avg < -cap:
                        results_mm_avg = (
                            results_array[i - 1, j] / 2 + results_array[i, j - 1] / 2
                        )
                    if results_mm_avg > cap:
                        results_mm_avg = (
                            results_array[i - 1, j] / 2 + results_array[i, j - 1] / 2
                        )
                    results_array[i, j] = results_mm_avg
                    title_name = f"Average Asset for the Gaussian Market Maker\n"
                elif max_result:
                    if results_mm_max > cap:
                        results_mm_max = (
                            results_array[i - 1, j] / 2 + results_array[i, j - 1] / 2
                        )
                    results_array[i, j] = results_mm_max
                    title_name = f"Maximum Asset for the Gaussian Market Maker\n"
                elif min_result:
                    if results_mm_min < -cap:
                        results_mm_min = -cap
                    results_array[i, j] = results_mm_min
                    title_name = f"Minimum Asset for the Gaussian Market Maker\n"
                elif std_result:
                    if results_mm_std < -cap:
                        results_mm_std = (
                            results_array[i - 1, j] / 2 + results_array[i, j - 1] / 2
                        )
                    if results_mm_std > cap:
                        results_mm_std = (
                            results_array[i - 1, j] / 2 + results_array[i, j - 1] / 2
                        )
                    results_array[i, j] = results_mm_std
                    title_name = f"Standard Deviation for the Gaussian Market Maker\n"

        plt.contourf(avgs, stds, results_array.T, cmap="cividis", levels=1000)
        plt.colorbar()

        hfont = {"fontname": "Helvetica"}

        plt.title(title_name, fontsize=13, fontweight="bold", **hfont)
        plt.xlabel("Average Price", fontsize=10, fontweight="bold", **hfont)
        plt.ylabel("Standard Deviation", fontsize=10, fontweight="bold", **hfont)
        plt.show()

        if average_result_log:
            title_name = (
                f"Best hyper-parameters for the Gaussian Market Maker (log scale)\n"
            )
            plt.contourf(
                avgs,
                stds,
                results_array.T,
                locator=ticker.LogLocator(),
                cmap="cividis",
                levels=200,
            )
            plt.colorbar()

            plt.title(title_name, fontsize=13, fontweight="bold", **hfont)
            plt.xlabel("Average Price", fontsize=10, fontweight="bold", **hfont)
            plt.ylabel("Standard Deviation", fontsize=10, fontweight="bold", **hfont)
            plt.show()

        return


if __name__ == "__main__":
    # Create Traders
    my_even_trader = EvenTrader(1000)
    my_intelligent_trader = IntelligentTrader(1000)
    my_random_trader = RandomTrader(1000)
    traders = [my_intelligent_trader, my_random_trader, my_even_trader]

    # Create Market Maker
    my_market_maker = GaussianMarketMaker()

    # Create Simulation
    my_multiple_simulations = RunMultipleSimulations(
        traders, my_market_maker, number_of_iteration=100
    )

    if False:
        my_multiple_simulations.display_price_slice()

    # Run and Display results
    if False:
        my_multiple_simulations.run_multiple_simulations()
        my_multiple_simulations.display_multiple_complete_results()

    if False:
        my_multiple_simulations.run_multiple_simulations()
        my_multiple_simulations.display_concatenated_history_of_assets()

    if True:
        my_multiple_simulations.run_multiple_simulations()
        my_multiple_simulations.process_simulation_results()
