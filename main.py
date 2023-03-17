'''
Main entrypoint of experiments: train, test, and eval models
'''

from src.utils.args import parse_args
from src.utils.data.getters import get_train_val_test_dataloader
from src.models.recon_classifier import ReconClassifier
from src.scripts.train import train_model
import src.config as cfg


def main():
    args = parse_args()

    train_loader, val_loader, test_loader = get_train_val_test_dataloader(
        cfg.DATASET
    )

    if args.mode == 'train':
        model = ReconClassifier(dim_c=512, dim_s=512).to(cfg.DEVICE)
        train_model(model=model, train_loader=train_loader,
                    val_loader=val_loader)


if __name__ == '__main__':
    main()
