# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np


def calculate_mse(e):
    """Calculate the Mean Squared Error (MSE) for vector e."""
    return 1/2 * np.mean(e**2)

def calculate_mae(e):
    """Calculate the Mean Absolute Error (MAE) for vector e."""
    return np.mean(np.abs(e))

def compute_loss(y, tx, w, loss_type='mse'):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2,). The vector of model parameters.
        loss_type: 'mse' or 'mae' to specify which loss to calculate.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # Calculate the error
    e = y - tx.dot(w)

    # Return MSE or MAE based on the selected loss type
    if loss_type == 'mse':
        return calculate_mse(e)
    elif loss_type == 'mae':
        return calculate_mae(e)
    else:
        raise ValueError("Invalid loss_type. Choose 'mse' or 'mae'.")

