import torch

def matrix_factorization(R, K, steps=100, lr=0.002):
    """
    Input
    -----
    R - Tensor: Ratings matrix_factorization:
        Dimensions: N-users by M-items
    K - Int: Number of latent features

    Output
    ------
    P: User-feature matrix
        Dimensions: N-users by K-features
    Qt: Transpose of Item-feature matrix
        Dimensions: K-features by M-items

    Reference
    ---------
    https://towardsdatascience.com/recommendation-system-matrix-factorization-d61978660b4b
    """
    # Initialize matrices P and Qt with random values between 1 and 0
    N_users, M_items = R.size()
    P = torch.randint(1, 5, (N_users, K))
    Qt = torch.randint(1, 5, (K, M_items))
    beta = 0.02
    prev_e = float("inf")
    # Setup training loop
    for step in range(steps):
        # Calculate and apply gradients
        for user_i in range(N_users):
            for item_j in range(M_items):
                if R[user_i][item_j] > 0:
                    pred = torch.dot(P[user_i,:], Qt[:,item_j])
                    err_ij = R[user_i][item_j] - pred

                    for k in range(K):
                        # Calculate and shift the gradient
                        P[user_i][k] = P[user_i][k] + lr * (2*err_ij*Qt[k][item_j] - beta*P[user_i][k])
                        Qt[k][item_j] = Qt[k][item_j] + lr * (2*err_ij*P[user_i][k] - beta*Qt[k][item_j])

        # Calculate difference in loss
        e = 0.0
        for user_i in range(N_users):
            for item_j in range(M_items):
                if R[user_i][item_j] > 0:
                    e = e + pow(R[user_i][item_j] - torch.dot(P[user_i,:],Qt[:,item_j]), 2)
                    for k in range(K):
                        e = e + (beta/2) * (pow(P[user_i][k],2) + pow(Qt[k][item_j],2))

        if 0 < (prev_e - e) < 50:
            break
        prev_e = e

        if step % 1 == 0:
            print("step: %s, loss: %s" % (step+1, e))
    return P, Qt




    # Calculate Root mean squared error for all values of R and P*Qt
    # Calculate gradients for all values of R and P*Qt

    # Apply gradients to P and Q

    # Return P and Qt