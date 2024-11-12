# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Gradient Descent
"""

from costs import compute_loss

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        w: numpy array of shape=(2, ). The vector of model parameters.

    Returns:
        A numpy array of shape (2, ), containing the gradient of the loss at w.
    """
    # Compute the error (residuals)
    err = y - tx.dot(w)
    
    # Compute the gradient
    grad = -tx.T.dot(err) / len(err)
    
    return grad



def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: numpy array of shape=(N, )
        tx: numpy array of shape=(N,2)
        initial_w: numpy array of shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        losses: a list of length max_iters containing the loss value (scalar) for each iteration of GD
        ws: a list of length max_iters containing the model parameters as numpy arrays of shape (2, ), for each iteration of GD
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    for n_iter in range(max_iters):
        # Compute the gradient and the loss
        grad = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)

        # Update w by taking a step in the direction of the negative gradient
        w = w - gamma * grad

        # Store the updated weights and the loss
        ws.append(w)
        losses.append(loss)
        
        # Print progress
        print(
            "GD iter. {bi}/{ti}: loss={l}, w0={w0}, w1={w1}".format(
                bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]
            )
        )

    return losses, ws