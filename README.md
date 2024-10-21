# README

## Project Overview

This project is a simplified stock market simulator built using multi-agent systems (SMA). The goal is to replicate the behavior of market participants (traders and market makers) in a stock exchange, allowing us to analyze different trading strategies and market dynamics.

## Main Concepts:
- **Multi-Agent Systems (SMA):** The simulator uses multiple agents (market maker and traders) to model real-world trading behaviors, including buying, selling, and price adjustments.
- **Market Maker:** Always ready to buy and sell. Adjusts prices at each iteration while ensuring buy prices are lower than sell prices.
- **Traders:** 
   - One trader alternates between buying and selling.
   - The other trader can follow a strategy of your choice.
   
## Simulation Flow:
- The simulation runs in discrete time steps (iterations).
- During each iteration:
   - The market maker updates prices.
   - Traders decide to either buy or sell.
   - The profit and loss (PnL) for each trader is updated accordingly.

## Project Goals:
- Simulate trading in a simplified stock market to observe market trends and dynamics.
- Analyze the performance of different trading strategies and their impact on profits.
- Experiment with adding more complexity, like multiple traders or smarter trading strategies.

## Features:
- **PnL Tracking:** Traders’ profits and losses are tracked and updated after every trade.
- **Market Maker Pricing:** Dynamic pricing by the market maker based on trading conditions.
- **Data Export:** Output key simulation variables into CSV for further analysis.
  
## Advanced Options:
- You can expand the simulation with more participants, smarter agents, and even create a dashboard to visualize market behavior.

Feel free to explore, modify, and test different scenarios to see how market dynamics evolve!
