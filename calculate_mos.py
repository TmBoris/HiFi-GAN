import hydra
from wvmos import get_wvmos


@hydra.main(version_base=None, config_path="src/configs", config_name="calculate_mos")
def main(config):
    model = get_wvmos(cuda=True)

    if config.path_to_file is not None:
        mos = model.calculate_one(config.path_to_file)
    elif config.path_to_dir is not None:
        mos = model.calculate_dir(config.path_to_dir, mean=True)
    else:
        raise ValueError("You should provide path to dir of to the file")

    print("MOS =", mos)


if __name__ == "__main__":
    main()
