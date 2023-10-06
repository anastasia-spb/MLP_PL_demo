import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from mpl_pl_model import MPLNetWrapper
from config import CONFIG

from mnist_eval_dataset import MNISTEvalDataset



def main():

    model_kwargs = {"in_dim": CONFIG["in_dim"], "out_dim": CONFIG["out_dim"],
                    "lr": CONFIG["lr"], "weight_decay": CONFIG["weight_decay"], "batch_size": CONFIG["batch_size"]}
    model = MPLNetWrapper.load_from_checkpoint(checkpoint_path=CONFIG["checkpoint_path"], **model_kwargs)

    trainer = pl.Trainer(accelerator='cpu')

    eval_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])
    ])

    eval_dataset = MNISTEvalDataset(img_folder='./mnist_eval_images', transform=eval_transforms)
    eval_loader = DataLoader(eval_dataset, batch_size=CONFIG["batch_size"], shuffle=False)
    
    output_predictions = trainer.predict(model, eval_loader)
    results = torch.argmax(output_predictions[0], dim=1)
    print(results)


if __name__ == "__main__":
    main()