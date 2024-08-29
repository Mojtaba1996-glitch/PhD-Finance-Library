import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Global variable to log the optimization process
log = []

# -------------------------
# Function: Drawdown
# -------------------------
def drawdown(return_series: pd.Series) -> pd.DataFrame:
    """
    Computes the drawdown of a return series.
    Drawdown is defined as the peak-to-trough decline during a specific period.

    Parameters:
    return_series (pd.Series): Time series of asset returns

    Returns:
    pd.DataFrame: DataFrame containing Wealth, Previous Peaks, and Drawdown
    """
    wealth_index = 1000 * (1 + return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    return pd.DataFrame({"Wealth": wealth_index, "Peaks": previous_peaks, "Drawdown": drawdowns})


# -------------------------
# Function: Semideviation
# -------------------------
def semideviation(returns: pd.Series) -> float:
    """
    Computes the semideviation (downside risk) of a return series.
    Semideviation measures the risk of returns falling below the mean.

    Parameters:
    returns (pd.Series): Time series of asset returns

    Returns:
    float: Semideviation of the return series
    """
    is_negative = returns < 0
    return returns[is_negative].std(ddof=0)


# -------------------------
# Function: Portfolio Volatility
# -------------------------
def portfolio_vol(weights: np.ndarray, covmatrix: np.ndarray) -> float:
    """
    Computes the volatility (standard deviation) of a portfolio.

    Parameters:
    weights (np.ndarray): Weights of the assets in the portfolio
    covmatrix (np.ndarray): Covariance matrix of asset returns

    Returns:
    float: Portfolio volatility
    """
    return np.sqrt(np.dot(weights.T, np.dot(covmatrix, weights)))


# -------------------------
# Function: Minimize Volatility
# -------------------------
def minimize_volatility(rets: np.ndarray, covmatrix: np.ndarray, target_return: float = None) -> np.ndarray:
    """
    Finds the portfolio weights that minimize volatility given a target return.

    Parameters:
    rets (np.ndarray): Expected returns of the assets
    covmatrix (np.ndarray): Covariance matrix of asset returns
    target_return (float, optional): Target return for the portfolio

    Returns:
    np.ndarray: Optimal portfolio weights
    """
    n_assets = len(rets)
    init_guess = np.repeat(1/n_assets, n_assets)

    # Constraint: Weights sum to 1
    weights_constraint = {"type": "eq", "args": (rets,), "fun": lambda w, r: np.sum(w) - 1}

    # If a target return is specified, add the constraint
    if target_return is not None:
        return_constraint = {
            "type": "eq",
            "args": (rets,),
            "fun": lambda w, r: target_return - np.dot(w, r)
        }
        constr = (return_constraint, weights_constraint)
    else:
        constr = weights_constraint

    result = minimize(portfolio_vol,
                      init_guess,
                      args=(covmatrix,),
                      method="SLSQP",
                      options={"disp": False},
                      constraints=constr,
                      bounds=((0.0, 1.0),) * n_assets)
    return result.x


# -------------------------
# Function: Optimal Weights
# -------------------------
def optimal_weights(n_points: int, rets: np.ndarray, cov: np.ndarray) -> list:
    """
    Generates a list of optimal portfolio weights for different target returns.

    Parameters:
    n_points (int): Number of points to generate between min and max returns
    rets (np.ndarray): Expected returns of the assets
    cov (np.ndarray): Covariance matrix of asset returns

    Returns:
    list: List of optimal portfolio weights
    """
    target_rets = np.linspace(rets.min(), rets.max(), n_points)
    weights = [minimize_volatility(rets, cov, target) for target in target_rets]
    return weights


# -------------------------
# Function: Plot Efficient Frontier
# -------------------------
def plot_ef(n_points: int, er: np.ndarray, cov: np.ndarray, periods_per_year: int):
    """
    Plots the efficient frontier for a set of assets.

    Parameters:
    n_points (int): Number of points to generate on the efficient frontier
    er (np.ndarray): Expected returns of the assets
    cov (np.ndarray): Covariance matrix of asset returns
    periods_per_year (int): Number of periods per year (e.g., 252 for daily returns)

    Returns:
    Plot: Scatter plot of the efficient frontier
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [np.dot(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    vols = np.array(vols) * np.sqrt(periods_per_year)

    ef = pd.DataFrame({
        "Returns": rets,
        "Volatility": vols,
        "Sharp ratio": np.array(rets) / np.array(vols)
    })
    return ef.plot.scatter(x="Volatility", y="Returns", c='Sharp ratio', colormap='viridis')


# -------------------------
# Function: Prepare Weights
# -------------------------
def Preparew(W: np.ndarray) -> np.ndarray:
    """
    Normalizes a weight vector so that the sum is 1.

    Parameters:
    W (np.ndarray): Weight vector

    Returns:
    np.ndarray: Normalized weight vector
    """
    W = W / np.sum(W)
    W = W.reshape(-1)
    return W


# -------------------------
# Function: Annualize Volatility
# -------------------------
def annualize_vol(s: pd.Series, periods_per_year: int, ddof: int = 1) -> float:
    """
    Annualizes the volatility of a return series.

    Parameters:
    s (pd.Series): Time series of asset returns
    periods_per_year (int): Number of periods per year (e.g., 252 for daily returns)
    ddof (int): Delta degrees of freedom for standard deviation calculation

    Returns:
    float: Annualized volatility
    """
    return s.std(ddof=ddof) * np.sqrt(periods_per_year)


# -------------------------
# Function: Loss Function - Minimize Semi-Volatility
# -------------------------
def LossFunctionMinSemiVol(W: np.ndarray, er: np.ndarray, sCo: np.ndarray, M0: float, Penalty: float = 100000) -> float:
    """
    Loss function to minimize portfolio semi-volatility.

    Parameters:
    W (np.ndarray): Weight vector
    er (np.ndarray): Expected returns of the assets
    sCo (np.ndarray): Covariance matrix of asset returns
    M0 (float): Minimum acceptable return
    Penalty (float): Penalty for not achieving the minimum return

    Returns:
    float: Loss value (penalized semi-volatility)
    """
    W = Preparew(W)
    m = np.dot(W, er)
    s = portfolio_vol(W, sCo)

    # Penalize if the return is below M0
    loss = s
    if m < M0:
        loss += Penalty * (M0 - m)

    log.append(loss)
    return loss


# -------------------------
# Function: Loss Function - Minimize Max Drawdown
# -------------------------
def LossFunctionMinMaxDrawdown(W: np.ndarray, Mu: np.ndarray, mdd: np.ndarray, M0: float, Penalty: float = 100000, epsilon: float = 1e-10) -> float:
    """
    Loss function to minimize portfolio maximum drawdown.

    Parameters:
    W (np.ndarray): Weight vector
    Mu (np.ndarray): Expected returns of the assets
    mdd (np.ndarray): Maximum drawdown of the assets
    M0 (float): Minimum acceptable return
    Penalty (float): Penalty for not achieving the minimum return
    epsilon (float): Small constant to avoid division by zero

    Returns:
    float: Loss value (penalized maximum drawdown)
    """
    W = Preparew(W)
    m = np.dot(W, Mu)
    s = np.dot(W, mdd)

    # Avoid division by zero
    if s <= epsilon:
        s = epsilon

    # Penalize if the return is below M0
    loss = s
    if m < M0:
        loss += Penalty * (M0 - m)

    log.append(loss)
    return loss


# -------------------------
# Function: Loss Function - Minimize CVaR
# -------------------------
def LossFunctionMinCVaR(W: np.ndarray, returns: pd.DataFrame, M0: float, alpha: float = 0.95, Penalty: float = 100000) -> float:
    """
    Loss function to minimize portfolio CVaR (Conditional Value at Risk).

    Parameters:
    W (np.ndarray): Weight vector
    returns (pd.DataFrame): Historical returns of the assets
    M0 (float): Minimum acceptable return
    alpha (float): Significance level for CVaR calculation
    Penalty (float): Penalty for not achieving the minimum return

    Returns:
    float: Loss value (penalized CVaR)
    """
    W = Preparew(W)
    portfolio_returns = np.dot(returns, W)
    m = portfolio_returns.mean()

    # Calculate VaR at the alpha level
    VaR = np.percentile(portfolio_returns, (1 - alpha) * 100)

    # Calculate CVaR as the mean of returns that are worse than VaR
    cvar = portfolio_returns[portfolio_returns <= VaR].mean()

    # Penalize if the return is below M0
    loss = -cvar  # CVaR is negative, so negate it to minimize positive values
    if m < M0:
        loss += Penalty * (M0 - m)

    log.append(loss)
    return loss
