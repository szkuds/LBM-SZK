def bc(f_i, boundary_condition):
    if boundary_condition == "bounce_back":
        return bounce_back(f_i)


def bounce_back(f_i):
    # Bounce-back top wall
    f_i = f_i.at[:, -1, 7].set(f_i[:, -1, 5])
    f_i = f_i.at[:, -1, 4].set(f_i[:, -1, 2])
    f_i = f_i.at[:, -1, 8].set(f_i[:, -1, 6])
    # Bounce-back bottom wall
    f_i = f_i.at[:, 0, 6].set(f_i[:, 0, 8])
    f_i = f_i.at[:, 0, 2].set(f_i[:, 0, 4])
    f_i = f_i.at[:, 0, 5].set(f_i[:, 0, 7])
    return f_i
