# Rain or Shine? Optimal Utility Pricing under Different Weather Patterns

**Author**: Guozhen (Gordon) Ji  
**Current Affiliation**: Associate, Econic Partners
**Degree**: Ph.D. in Economics (2026) 
**Advisor**: Dr. Eugenio Miravete  
**Institution**: University of Texas at Austin  


---

## Abstract

> As climate change amplifies more volatile weather patterns, water utilities face increasing difficulty in simultaneously ensuring revenue feasibility, promoting water conservation, and protecting low-income consumers.
> This paper tests and concludes that price alone cannot achieve these competing policy goals under different weather patterns.
> Using granular household data from Austin, TX, and a structural demand model enhanced with satellite imagery-derived vegetation index, I find that because high-water users exist across all income levels, traditional tiered pricing doesn’t work as intended.
> Furthermore, higher-income households—who are both weather-sensitive and surprisingly price-elastic—complicate the utility's ability to achieve its distributional objectives while meeting the conservation target.
> When high-demand conditions (e.g., drought) make conservation measures necessary, low-income families experience an average welfare loss of \$74 per month.
> This highlights the necessity of complementary policies to achieve distributional goals when demand increases.
> For example, a program encouraging households to convert 30\% of their lawns to water-saving landscapes (zeroscaping/xeriscaping) could generate approximately \$70 per month in welfare for the lowest-income families, nearly offsetting the financial burden imposed by conservation policies during droughts.

---
## Repository Structure

```text
├── pre_analysis/               # Data cleaning, GIS data, and NDVI
├── demand/                     # Demand Estimation
├── price_elasticity/           # Code to generate price elasticity
├── NDVI_categorization/        # Code to generate 3DPD results related to NDVI based segmentation
├── preliminary_intuition/      # Code related to constructing preliminary intuition of the optimal Ramsey price
├── counterfactual_temp/        # Counterfactual analysis that's not been used
├── counterfactual_ramsey/      # Counterfactual analysis of Ramsey Pricing Model
├── other_app_info/             # Folders contain other application info
├── Austin_Water.pdf            # Formatted Paper (Last edited Oct 2026)
├── Austin_Water_Online_Appendix.pdf            # Online Appendix (Last edited Oct 2026)
└── README.md                   # Project overview (this file)

```
---

## Overview

This repository accompanies my dissertation research on the intersection of utility pricing, climate variability, and economic welfare. It includes:

- A structural DCC model of household water demand
- Counterfactual simulations of Ramsey pricing under different weather conditions
- Estimation of the shadow cost of policy constraint under counterfactual weather conditions, especially for the lowest income stratum
- The welfare benefit from the additional policies like Zeroscaping towards the lowest income stratum

---

## Data

The data comes from Austin Water's monthly transaction records.  
Due to privacy constraints, raw data are not publicly included.  

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.


---

## Contact

If you have questions or want to collaborate, feel free to reach out: [gordonjgz@gmail.com]

---
