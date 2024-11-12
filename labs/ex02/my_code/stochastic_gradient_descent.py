# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Stochastic Gradient Descent
"""
from helpers import batch_iter
from costs import compute_loss


def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from a mini-batch of size B.

    Args:
        y: numpy array of shape=(B, )
        tx: numpy array of shape=(B,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ), containing the stochastic gradient of the loss at w.
    """
    # Compute the error for the mini-batch
    err = y - tx.dot(w)
    
    # Compute the stochastic gradient
    stoch_grad = -tx.T.dot(err) / len(err)
    
    return stoch_grad


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """The Stochastic Gradient Descent algorithm (SGD).

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        batch_size: a scalar denoting the number of data points in a mini-batch used for computing the stochastic gradient
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of SGD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of SGD
    """
    # Initialize parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        # Create a random mini-batch of data
        for mini_y, mini_tx in batch_iter(y, tx, batch_size):
            # Compute the stochastic gradient for the mini-batch
            stoch_grad = compute_stoch_gradient(mini_y, mini_tx, w)
            
            # Update w using the stochastic gradient
            w = w - gamma * stoch_grad
            
            # Compute the loss for this mini-batch
            loss = compute_loss(y, tx, w)
            
            # Store updated weights and loss
            ws.append(w)
            losses.append(loss)
            
            # Print progress
            print(
                "SGD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                    bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
                )
            )

    return losses, ws
