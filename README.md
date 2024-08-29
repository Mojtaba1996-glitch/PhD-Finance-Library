# PhD Finance Library

## Overview
This library was developed as part of my PhD studies in statistics at Chiang Mai university (CMU). It provides a set of tools for portfolio optimization, risk analysis, and financial modeling.
Special thanks to my advisor Dr. Nawapon for her support.

## Features
- Portfolio optimization using various methods (e.g., Minimum Variance, Maximum Sharpe Ratio)
- Risk metrics calculation (e.g., VaR, CVaR, Maximum Drawdown)
- Efficient frontier plotting
- Various utility functions for financial calculations

## Installation
```bash
pip install cmu
```

## Usage
Here's a simple example of how to use the library:

```python
import cmu as cmu

# Example code here
```
returns = pd.Series([...])  # Replace with your data
dd = cmu.drawdown(returns)
sd = cmu.semideviation(returns)

print(dd)
print(sd)


## Dependencies
- numpy
- pandas
- scipy
- matplotlib
- yfinance


## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Licance
This project is licensed under the MIT License

## Contact
Mojtaba Safari - growingscience1996@gmail.com 

Project Link: [https://github.com/Mojtaba1996-glitch/](https://github.com/Mojtaba1996-glitch/PhD Finance Library)
