import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.integrate import solve_ivp
from torch.optim import AdamW
from torchdyn.core import ODEProblem
from tqdm import tqdm


def seed_everything(seed=1234):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


seed_everything()

device = tordevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###################################
# Argments
IS_RESTART = False
P_CUTOFF = 0.0
N_EPOCH = 1000000
N_PLOT = 100
DATASIZE = 100
TSTEP = 0.4
N_EXP_TRAIN = 20
N_EXP_TEST = 10
N_EXP = N_EXP_TRAIN + N_EXP_TEST
NOISE = 5.0e-2
NS = 5
NR = 4
K = torch.tensor([0.1, 0.2, 0.13, 0.3])
ATOL = 1e-5
RTOL = 1e-2

MAXITERS = 10000

LB = 1.0e-5
UB = 1.0e1

B0 = -10

#######################################################

# Make sure to not put a '/' after the folder name
BASE_DIR = "./"
SAVE_EXP_DIR = "figs_py"
CHECKPOINT_DIR = "checkpoint"
CHECKPOINT_SAVE_PATH = f"{BASE_DIR}/{CHECKPOINT_DIR}/mymodel.pt"
LOSS_SAVE_PATH = f"{BASE_DIR}/{SAVE_EXP_DIR}/loss.png"


if os.path.exists(BASE_DIR) == False:
    os.mkdir(BASE_DIR)
if os.path.exists(f"{BASE_DIR}/{SAVE_EXP_DIR}") == False:
    os.mkdir(f"{BASE_DIR}/{SAVE_EXP_DIR}")
if os.path.exists(f"{BASE_DIR}/{CHECKPOINT_DIR}") == False:
    os.mkdir(f"{BASE_DIR}/{CHECKPOINT_DIR}")


def trueODEfunc(t, y, k):
    dydt_0 = -2 * k[0] * y[0] ** 2 - k[1] * y[0]
    dydt_1 = k[0] * y[0] ** 2 - k[3] * y[1] * y[3]
    dydt_2 = k[1] * y[0] - k[2] * y[2]
    dydt_3 = k[2] * y[2] - k[3] * y[1] * y[3]
    dydt_4 = k[3] * y[1] * y[3]
    return [dydt_0, dydt_1, dydt_2, dydt_3, dydt_4]


def max_min(ode_data):
    return torch.amax(ode_data, dim=1) - torch.amin(ode_data, dim=1) + LB


def p2vec(p):
    w_b = p[:NR] + B0
    w_out = torch.reshape(p[NR:], (NR, NS)).transpose(0, 1)
    w_in = torch.clamp(-w_out, 0, 2.5)
    return w_in, w_b, w_out


def display_p(p):
    w_in, w_b, w_out = p2vec(p)
    print("species (column) reaction (row)")
    print("w_in")
    print(w_in.transpose(0, 1))
    print("\nw_b")
    print(w_b)
    print("\nw_out")
    print(w_out.transpose(0, 1))
    print("\n\n")


class CRNN(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p = torch.nn.Parameter(torch.randn(NR * (NS + 1)) * 1.0e-1)

    def forward(self, x):
        w_in, w_b, w_out = self._p2vec()
        w_in_x = w_in.transpose(0, 1) @ torch.log(torch.clamp(x, min=LB, max=UB))
        return w_out @ torch.exp(w_in_x + w_b)

    def _p2vec(self):
        w_b = self.p[:NR] + B0
        w_out = torch.reshape(self.p[NR:], (NR, NS)).transpose(0, 1)
        w_in = torch.clamp(-w_out, 0, 2.5)
        return w_in, w_b, w_out


def plot_losses(list_loss_train, list_loss_test):
    fig, ax = plt.subplots(figsize=(5, 2.7), layout="constrained")
    ax.plot(list_loss_train, label="train")
    ax.plot(list_loss_test, label="val")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    ax.grid(True)
    ax.set_title("Loss Plots")  # Add a title to the axes.
    ax.legend()
    # Add a legend.
    fig.savefig(LOSS_SAVE_PATH)


def plot_exps(tsteps, ode_data_list, pred, i_exp):
    species = ["A", "B", "C", "D", "E"]

    ncols = NS - int(NS / 2)
    nrows = int(NS / 2)
    fig, axs = plt.subplots(
        ncols=ncols, nrows=nrows, figsize=(20.5, 12.5), layout="constrained"
    )

    ode_index = 0
    for row in range(nrows):
        for col in range(ncols):
            if ode_index == NS:
                break
            ode_data = ode_data_list[i_exp, ode_index, :]
            axs[row, col].scatter(tsteps, ode_data, label="Exp", c="orange", marker="x")
            axs[row, col].plot(tsteps, pred[ode_index], label="CRNN-ODE")
            axs[row, col].set_xlabel("Time")
            axs[row, col].set_ylabel(f"Concentration of {species[ode_index]}")

            if ode_index == 0:
                axs[row, col].legend()

            if row == 1 and col == 2:
                axs[row, col].get_xaxis().set_visible(False)
                axs[row, col].get_yaxis().set_visible(False)

            ode_index += 1

    fig.savefig(f"{BASE_DIR}/{SAVE_EXP_DIR}/i_exp_{i_exp}.png")


def main():
    seed_everything()

    # Generate Datasets
    u0_list = torch.rand((N_EXP, NS)).reshape((NS, N_EXP)).transpose(0, 1)
    u0_list[:, 0:2] += 2.0e-1
    u0_list[:, 2:] = 0.0
    tspan = torch.tensor([0.0, DATASIZE * TSTEP])
    tsteps = torch.linspace(tspan[0], tspan[1], DATASIZE)
    ode_data_list = torch.zeros((N_EXP, NS, DATASIZE))
    std_list = torch.tensor([])

    # push ode data to std list
    print("Calculating y_std...")
    for i in range(N_EXP):
        u0 = u0_list[i]
        ode_data = solve_ivp(
            trueODEfunc,
            tspan.numpy(),
            u0.numpy(),
            t_eval=tsteps.numpy(),
            args=(K.numpy(),),
        )
        ode_data = ode_data.y
        ode_data = torch.tensor(ode_data)
        # The reshape and the transpose makes the alignment same with julia
        ode_data += (
            torch.rand((ode_data.shape))
            .reshape((ode_data.shape[1], -1))
            .transpose(0, 1)
            * ode_data
            * NOISE
        )
        ode_data_list[i] = ode_data
        std_list = torch.cat((std_list, ode_data.unsqueeze(dim=0)))

    y_std = torch.max(std_list)
    print(f"y_std: \t{y_std}")

    u0 = u0_list[0]
    p = torch.randn(NR * (NS + 1)) * 1.0e-1

    model = CRNN()
    model_prob = ODEProblem(model, solver="rk4", atol=ATOL, rtol=RTOL).to(device)

    criterion = torch.nn.L1Loss()
    optimizer = AdamW(model_prob.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1.0e-8)

    epoch_start = 0
    if IS_RESTART:
        checkpoint = torch.load(CHECKPOINT_SAVE_PATH)
        model_prob.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch_start = checkpoint["epoch"]
        loss_train = checkpoint["loss_train"]
        loss_test = checkpoint["loss_test"]

    list_loss_train = []
    list_loss_test = []

    tsteps = tsteps.to(device)
    ode_data_list = ode_data_list.to(device)
    y_std = y_std.to(device)

    for num_iter, epoch in enumerate(tqdm(range(epoch_start, N_EPOCH))):
        loss_epoch = torch.zeros(N_EXP).to(device)

        model_prob.train()
        for i_exp in torch.randperm(N_EXP_TRAIN):
            u_train = u0_list[i_exp].to(device)
            t_train, y_hat = model_prob(u_train, tsteps)
            y_hat = y_hat.transpose(0, 1)

            loss = criterion(ode_data_list[i_exp] / y_std, y_hat / y_std)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch[i_exp] = loss

        model_prob.eval()
        for i_exp_eval in range(N_EXP_TRAIN, N_EXP):
            u_eval = u0_list[i_exp_eval].to(device)
            t_eval, y_hat_eval = model_prob(u_eval, tsteps)
            y_hat_eval = y_hat_eval.transpose(0, 1)

            loss_eval = criterion(ode_data_list[i_exp_eval] / y_std, y_hat_eval / y_std)
            loss_epoch[i_exp_eval] = loss_eval

        loss_train = torch.mean(loss_epoch[:N_EXP_TRAIN])
        loss_test = torch.mean(loss_epoch[N_EXP_TRAIN:])

        print(f"\nLoss Train: \t{loss_train.item()}")
        print(f"Loss Test: \t{loss_test.item()}")

        list_loss_train.append(loss_train.item())
        list_loss_test.append(loss_test.item())

        if num_iter % N_PLOT == 0:
            display_p(next(model_prob.parameters()).cpu().detach())

            print(f"Minimum Loss Train: \t{min(list_loss_train)}")
            print(f"Minimum Loss Test: \t{min(list_loss_test)}")

            # Plot exp figures
            i_exp = np.random.permutation(N_EXP)[0]
            model_prob.eval()
            i_exp_data = u0_list[i_exp].to(device)
            _, y_hat_exp = model_prob(i_exp_data, tsteps)
            y_hat_exp = y_hat_exp.transpose(0, 1)
            # plot_exps(tsteps, ode_data_list, y_hat_exp.detach().numpy(), i_exp)
            plot_exps(
                tsteps.cpu().numpy(),
                ode_data_list.cpu().numpy(),
                y_hat_exp.detach().cpu().numpy(),
                i_exp,
            )

            # Save loss
            plot_losses(list_loss_train, list_loss_test)

            # Save checkpoint
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_prob.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss_train": loss_train.item(),
                    "loss_test": loss_test.item(),
                },
                CHECKPOINT_SAVE_PATH,
            )


if __name__ == "__main__":
    main()
