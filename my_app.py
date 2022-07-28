from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(config_path="conf", config_name="config", version_base=None)
def my_app(cfg: DictConfig) -> None:
    print("Nope, this is not the config you are looking for")
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    my_app()
