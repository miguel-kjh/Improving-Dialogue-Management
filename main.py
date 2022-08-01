from omegaconf import DictConfig, OmegaConf
import hydra


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    print(cfg)
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
