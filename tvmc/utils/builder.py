from tvmc.models.LPTF import LPTF
from tvmc.models.PTF import PTF
from tvmc.models.RNN import PRNN


def build_model(config):
    # === TRAIN CONFIG ===
    train_cfg = config["TRAIN"]
    L = train_cfg["L"]
    # seed = train_cfg.get("seed") or np.random.randint(65536)

    # # Set seeds
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # train_cfg["seed"] = seed

    # Ensure batch size
    train_cfg["B"] = train_cfg["K"] * train_cfg["Q"]

    # === HAMILTONIAN CONFIG ===
    if config["HAMILTONIAN"]["name"] == "Rydberg" or config["HAMILTONIAN"]["name"] == "Ising":
        config["HAMILTONIAN"]["Lx"] = config["HAMILTONIAN"]["Ly"] = int(L**0.5)
        config["HAMILTONIAN"]["L"] = L

    # === MODEL SELECTION ===
    model_type = train_cfg.get("model", "PTF")

    if model_type == "LPTF":
        # Subsampler model name (e.g., "rnn" → "RNN")
        sub_name = config["LPTF"].get("subsampler", "rnn").upper()
        sub_cfg = config.get(sub_name, {})
        lptf_cfg = config["LPTF"]

        # Set hidden size(s)
        if sub_name == "PTF":
            Nh = [sub_cfg["Nh"], lptf_cfg["Nh"]]
        else:
            Nh = lptf_cfg["Nh"]

        # Build subsampler
        if sub_name == "RNN":
            submodel = PRNN(sub_cfg["L"], sub_cfg["patch"], sub_cfg["rnntype"], Nh)
        elif sub_name == "PTF":
            submodel = PTF(
                sub_cfg["L"],
                sub_cfg["patch"],
                Nh,
                sub_cfg["dropout"],
                sub_cfg["num_layers"],
                sub_cfg["nhead"],
                sub_cfg["repeat_pre"],
            )
        else:
            raise ValueError(f"Unknown subsampler: {sub_name}")

        # Build LPTF
        model = LPTF(
            submodel,
            lptf_cfg["L"],
            lptf_cfg["patch"],
            lptf_cfg["Nh"],
            lptf_cfg["dropout"],
            lptf_cfg["num_layers"],
            lptf_cfg["nhead"],
            lptf_cfg["full_seq"],
        )

    else:
        model_cfg = config[model_type]
        if model_type == "PTF":
            model = PTF(
                model_cfg["L"],
                model_cfg["patch"],
                model_cfg["Nh"],
                model_cfg["dropout"],
                model_cfg["num_layers"],
                model_cfg["nhead"],
                model_cfg["repeat_pre"],
            )
        elif model_type == "RNN":
            model = PRNN(model_cfg["L"], model_cfg["patch"], model_cfg["rnntype"], model_cfg["Nh"])
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    config["TRAIN"] = train_cfg

    return model, config
