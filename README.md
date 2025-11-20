# energynow-vpp-optimization

> This repo is linked to the Energy Now competition where we aim to do a profitability assessment of battery energy storage systems integrated into a virtual power plant (VPP)

## 📝 Overview
This repository contains all the optimization code for running the various Virtual-power-plant scenaios.


## 🔗 Documentation & Resources
All detailed notes, progress updates, and documentation can be found here:  
👉 [View the Notion Page](https://www.notion.so/Energy-Now-BESS-Profitability-in-VPPs-28aacaa51f9980f5b706ec4fb4e03fcb)

## Runing the optimization
To run the optimization a full license of the Gurobi solver is needed: https://www.gurobi.com/academia/academic-program-and-licenses/
After loging in with the ETH account you need to past the provided command into your CMD
Also you need to install the following packages: gurobipy, pandas, plotly

IMPORTANT: To run the gurobi optimization a python installation of version 3.13 or LESS is needed. 3.14 does not work!

Use a python virtual environment to run it, package requirements in requirements.txt.
Steps to run:
1. Activate the virtual environment
2. Ensure gurobi is installed ad enabled
3. python3 model.py
