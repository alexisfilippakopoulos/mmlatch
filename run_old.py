import argparse
import os
import sys
from pprint import pprint

import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.metrics import Accuracy, Fbeta, Loss
from torch.utils.data import DataLoader
import numpy as np

from mmlatch.cmusdk import mosei, mosi
from mmlatch.config import load_config
from mmlatch.data import MOSEI, MOSEICollator, ToTensor
from mmlatch.mm import AudioVisualTextClassifier, AVTClassifier
from mmlatch.trainer import MOSEITrainer
from mmlatch.util import safe_mkdirs
from mmlatch.noise import add_noise

# Fix generator seed for consistent DataLoader shuffling
generator = torch.Generator()
generator.manual_seed(42)  # Fixed seed
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
    
class BCE(nn.Module):
    """Custom binary cross-entropy loss function"""
    def __init__(self):
        super(BCE, self).__init__()

    def forward(self, out, tgt):
        tgt = tgt.view(-1, 1).float()

        return F.binary_cross_entropy_with_logits(out, tgt)


def get_parser():
    """Defines and returns an ArgumentParser for parsing command-line inputs"""
    parser = argparse.ArgumentParser(description="CLI parser for experiment")

    parser.add_argument( #pecifies the dropout probability to be used in the model.
        "--dropout",
        dest="common.dropout",
        default=None,
        type=float,
        help="Dropout probabiity",
    )

    parser.add_argument( #Defines the projection size for modality features.
        "--proj-size",
        dest="fuse.projection_size",
        default=None,
        type=int,
        help="Modality projection size",
    )

    parser.add_argument( #Defines if the RNN is bidirectional
        "--bidirectional",
        dest="common.bidirectional",
        action="store_true",
        help="Use BiRNNs",
    )

    parser.add_argument( #Defines if LSTM or GRU will be used
        "--rnn-type", dest="common.rnn_type", default=None, type=str, help="lstm or gru"
    )

    parser.add_argument( #Defines if feedback will be used
        "--feedback",
        dest="feedback",
        action="store_true",
        help="Use feedback fusion",
    )

    parser.add_argument(
        "--mask-index-train",
        dest="model.mask_index_train",
        default=1,
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Masking strategy index (1: average, 2: max, 3: min, 4: residual, 5: max deviation from 0.5)",
    )

    parser.add_argument(
        "--mask-index-test",
        dest="model.mask_index_test",
        default=1,
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        help="Masking strategy index (1: average, 2: max, 3: min, 4: residual, 5: max deviation from 0.5, 6: no mmlatch)",
    )

    parser.add_argument( #Defines the number of classes
        "--mask-dropout-train",
        dest="model.mask_dropout_train",
        default=0.0,
        type=float,
        help="Masking dropout probability during training",
    )

    parser.add_argument( #Defines the number of classes
        "--mask-dropout-test",
        dest="model.mask_dropout_test",
        default=0.0,
        type=float,
        help="Masking dropout probability during testing",
    )


    parser.add_argument( #Defines the directory for the results
        "--result-dir",
        dest="results_dir",
        help="Results directory",
    )

    parser.add_argument( 
        "--noise-type",
        dest="model.noise_type",
        default="none",
        type=str,
        help="Type of noise to be used on train/test data (none, gaussian, dropout, shuffle)",
    )

    parser.add_argument( 
        "--noise-percentage-train",
        dest="model.noise_percentage_train",
        default=0.0,
        type=float,
        help="Percentage of noise on train set",
    )

    parser.add_argument( 
        "--noise-percentage-test",
        dest="model.noise_percentage_test",
        default=0.0,
        type=float,
        help="Percentage of noise on test set",
    )

    parser.add_argument(
        "--augment_train_data",
        dest="model.augment_train_data",
        action="store_true",
        help="Determines whether to augment train data with noise",
    )

 
    parser.add_argument( 
        "--noise-modality",
        dest="model.noise_modality",
        default="all",
        type=str,
        help="Modality to be affected by the noise (all, text, audio, visual)",
    )
    
    
    return parser


C = load_config(parser=get_parser()) #Loads the configuration file

collate_fn = MOSEICollator( #Defines the collator for the DataLoader
    device="cpu", modalities=["text", "audio", "visual"], max_length=-1
)


if __name__ == "__main__":
    
    print("Running with configuration")
    pprint(C)
    SUBSET_FRACTION = 1 #Defines the fraction of the dataset to be used
    train, dev, test, vocab = mosi( #loads the mosi dataset
        C["data_dir"],
        modalities=["text", "glove", "audio", "visual"],
        remove_pauses=False,
        max_length=-1,
        pad_front=True,
        pad_back=False,
        aligned=False,
        cache=os.path.join(C["cache_dir"], "mosei_avt.p"),
        fraction=SUBSET_FRACTION,  # Pass the fraction here

    )

    # Use GloVe features for text inputs
    for d in train:
        d["text"] = d["glove"]

    for d in dev:
        d["text"] = d["glove"]

    for d in test:
        d["text"] = d["glove"]

    # Add noise
    train = add_noise(train,
                        noise_type=C['model']['noise_type'], 
                        noise_modality=C['model']['noise_modality'], 
                        noise_level=C['model']['noise_percentage_train'],
                        augment = C['model']['augment_train_data'],
                        )
    #train = append_noise(train,
    #                  noise_type=C['model']['noise_type'], 
    #                  noise_modality=C['model']['noise_modality'], 
    #                  noise_level=C['model']['noise_percentage_train']
    #                  )

    #converts data to tensors
    to_tensor = ToTensor(device="cpu")
    to_tensor_float = ToTensor(device="cpu", dtype=torch.float)

    def create_dataloader(data, shuffle=True):
        """Creates a DataLoader for the given data"""
        d = MOSEI(data, modalities=["text", "glove", "audio", "visual"], select_label=0)
        d.map(to_tensor_float, "visual", lazy=True) #maps the visual data to tensor
        d.map(to_tensor_float, "text", lazy=True) #maps the text data to tensor
        d = d.map(to_tensor_float, "audio", lazy=True) #maps the audio data to tensor
        d.apply_transforms() #applies the transforms to the data
        dataloader = DataLoader(
            d,
            batch_size=C["dataloaders"]["batch_size"],
            num_workers=C["dataloaders"]["num_workers"],
            pin_memory=C["dataloaders"]["pin_memory"],
            shuffle=shuffle,
            collate_fn=collate_fn,
            generator=generator,
        )

        return dataloader

    train_loader = create_dataloader(train)
    dev_loader = create_dataloader(dev)
    test_loader = create_dataloader(test)
    print("Running with feedback = {}".format(C["model"]["feedback"]))

    model = AVTClassifier(
        C["model"]["text_input_size"],
        C["model"]["audio_input_size"],
        C["model"]["visual_input_size"],
        C["model"]["projection_size"],
        text_layers=C["model"]["text_layers"],
        audio_layers=C["model"]["audio_layers"],
        visual_layers=C["model"]["visual_layers"],
        bidirectional=C["model"]["bidirectional"],
        dropout=C["model"]["dropout"],
        encoder_type=C["model"]["encoder_type"],
        attention=C["model"]["attention"],
        feedback=C["model"]["feedback"],
        feedback_type=C["model"]["feedback_type"],
        device=C["device"],
        num_classes=C["num_classes"],
        mask_index=C["model"]["mask_index_train"],  # Pass mask_index
        mask_dropout=C["model"]["mask_dropout_train"],  # Pass mask_dropout
    )

    # if you want to run it with the AudioVisualTextClassifier class instead
    """
    text_cfg = {
            "input_size": C["model"]["text_input_size"], "hidden_size": C["model"]["projection_size"], "layers": C["model"]["text_layers"], "bidirectional": C["model"]["bidirectional"],
            "dropout": C["model"]["dropout"], "rnn_type": "lstm", "attention": C["model"]["attention"]
        }
    audio_cfg = {
            "input_size": C["model"]["audio_input_size"], "hidden_size": C["model"]["projection_size"], "layers": C["model"]["audio_layers"], "bidirectional": C["model"]["bidirectional"],
            "dropout": C["model"]["dropout"], "rnn_type": "lstm", "attention": C["model"]["attention"]
        }
    visual_cfg = {
            "input_size": C["model"]["visual_input_size"], "hidden_size": C["model"]["projection_size"], "layers": C["model"]["visual_layers"], "bidirectional": C["model"]["bidirectional"],
            "dropout": C["model"]["dropout"], "rnn_type": "lstm", "attention": C["model"]["attention"]
        }
    fuse_cfg = {
        "projection_size": C["model"]["projection_size"], "feedback_type": C["model"]["feedback_type"]
    }
    model = AudioVisualTextClassifier(
        audio_cfg=audio_cfg,
        text_cfg=text_cfg,
        visual_cfg=visual_cfg,
        fuse_cfg=fuse_cfg,
        modalities=["text", "audio", "visual"],
        num_classes=C["num_classes"],
        feedback=C["model"]["feedback"],
        device=C["device"],
    
    )
    """

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("NUMBER OF PARAMETERS: {}".format(count_parameters(model)))

    model = model.to(C["device"])
    optimizer = getattr(torch.optim, C["optimizer"]["name"])(
        [p for p in model.parameters() if p.requires_grad],
        lr=C["optimizer"]["learning_rate"],
    )

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        factor=0.5,
        patience=2,
        cooldown=2,
        min_lr=C["optimizer"]["learning_rate"] / 20.0,
    )

    criterion = nn.L1Loss()

    def bin_acc_transform(output):
        y_pred, y = output
        nz = torch.nonzero(y).squeeze()
        yp, yt = (y_pred[nz] >= 0).long(), (y[nz] >= 0).long()

        return yp, yt

    def acc_transform(output):
        y_pred, y = output
        yp, yt = (y_pred >= 0).long(), (y >= 0).long()

        return yp, yt

    def acc7_transform(output):
        y_pred, y = output
        yp = torch.clamp(torch.round(y_pred) + 3, 0, 6).view(-1).long()
        yt = torch.round(y).view(-1).long() + 3
        yp = F.one_hot(yp, 7)

        return yp, yt

    def acc5_transform(output):
        y_pred, y = output
        yp = torch.clamp(torch.round(y_pred) + 2, 0, 4).view(-1).long()
        yt = torch.round(y).view(-1).long() + 2
        yp = F.one_hot(yp, 5)

        return yp, yt

    metrics = {
        "acc5": Accuracy(output_transform=acc5_transform),
        "acc7": Accuracy(output_transform=acc7_transform),
        "bin_accuracy": Accuracy(output_transform=bin_acc_transform),
        "f1": Fbeta(1, output_transform=bin_acc_transform),
        "accuracy_zeros": Accuracy(output_transform=acc_transform),
        "loss": Loss(criterion),
    }

    if C["overfit_batch"] or C["overfit_batch"] or C["train"]:
        import shutil

        try:
            shutil.rmtree(C["trainer"]["checkpoint_dir"])
        except:
            pass
        if C["trainer"]["accumulation_steps"] is not None:
            acc_steps = C["trainer"]["accumulation_steps"]
        else:
            acc_steps = 1
        trainer = MOSEITrainer(
            model,
            optimizer,
            # score_fn=score_fn,
            experiment_name=C["experiment"]["name"],
            checkpoint_dir=C["trainer"]["checkpoint_dir"],
            metrics=metrics,
            non_blocking=C["trainer"]["non_blocking"],
            patience=C["trainer"]["patience"],
            validate_every=C["trainer"]["validate_every"],
            retain_graph=C["trainer"]["retain_graph"],
            loss_fn=criterion,
            accumulation_steps=acc_steps,
            lr_scheduler=lr_scheduler,
            device=C["device"],
            enable_plot_embeddings=False # we enable this flag only at test trainer (if it is true  in the config file)
        )

    if C["debug"]:
        if C["overfit_batch"]:
            trainer.overfit_single_batch(train_loader)
        trainer.fit_debug(train_loader, dev_loader)
        sys.exit(0)

    if C["train"]:
        trainer.fit(train_loader, dev_loader, epochs=C["trainer"]["max_epochs"])

    if C["test"]:
        try:
            del trainer
        except:
            pass

        trainer = MOSEITrainer(
            model,
            optimizer,
            experiment_name=C["experiment"]["name"],
            checkpoint_dir=C["trainer"]["checkpoint_dir"],
            metrics=metrics,
            model_checkpoint=C["trainer"]["load_model"],
            non_blocking=C["trainer"]["non_blocking"],
            patience=C["trainer"]["patience"],
            validate_every=C["trainer"]["validate_every"],
            retain_graph=C["trainer"]["retain_graph"],
            loss_fn=criterion,
            device=C["device"],
            enable_plot_embeddings=C["model"]["enable_plot_embeddings"]
            
        )
        #UPDATE MASK INDEX HERE IF NEEDED
        trainer.set_mask_index(C["model"]["mask_index_test"])
        trainer.set_mask_dropout(C["model"]["mask_dropout_test"])
        predictions, targets,masks_txt,masks_au,masks_vi = trainer.predict(test_loader)
        
        experiment_name = C["experiment"]["name"]
        results_dir = C["results_dir"] + f"/{experiment_name}"
        
        if C["model"]["enable_plot_embeddings"]:
            model.plot_embeddings(torch.cat(targets), results_dir)
           





        # insert code to load
        pred = torch.cat(predictions)
        y_test = torch.cat(targets)
 
        import uuid

        from mmlatch.mosei_metrics import (
        eval_mosei_senti,
        print_metrics,
        save_metrics,
        save_comparison_data_pickle,  # Newly added
        compare_masks,         # Newly added if needed
        prediction_count,   # Newly added
        plot_masks,         # Newly added if needed
        save_histogram_data
        )

        metrics = eval_mosei_senti(pred, y_test, True)
        print_metrics(metrics)
        safe_mkdirs(results_dir+"/numeric_results") #creates the directory for the results if it does not exist
        safe_mkdirs(results_dir+"/plot_images") #creates the directory for the plots if it does not exist
        safe_mkdirs(results_dir+"/plot_numbers")
        fname = f'results'
        results_file = os.path.join(results_dir + f"/numeric_results", fname)
        fname2 = fname + "_masks"
        results_file2 = os.path.join(results_dir + f"/numeric_results", fname2)
        save_metrics(metrics, results_file)

        comparison_filename = f"comparison_mask.pkl"
        comparison_filepath = os.path.join(C["results_dir"], comparison_filename)

        data = {
    'predictions': pred.cpu().numpy(),      # Removed torch.cat
    'targets': y_test.cpu().numpy(),        # Removed torch.cat
    'masks_txt': [mask.cpu().numpy() for mask in masks_txt],
    'masks_au': [mask.cpu().numpy() for mask in masks_au],
    'masks_vi': [mask.cpu().numpy() for mask in masks_vi],
}

        avg_metrics, mean_mask_mod_target, diff_mask_mod_target,mean_mask_new_pred = compare_masks(data, comparison_filepath)

        # Print average metrics
        print_metrics(avg_metrics)

        # Save the metrics
        save_metrics(avg_metrics, results_file2)

        # Retrieve experiment name for labeling
        

        # Plot and save masks for each modality
        for modality in ["txt", "au", "vi"]:
            # Plot and save mean masks
            plot_masks(
                mean_mask_mod_target[modality],
                f'Mean_{modality}_Target',
                save_directory=results_dir,
                title=f"Averaged {modality} Masks per Target for {experiment_name}",
                ylabel= 'Target'
            )
            plot_masks(
                diff_mask_mod_target[modality],
                f'Difference_{modality}_Target',
                save_directory=results_dir,
                title=f"Difference of {modality} Averaged Masks with Default per Target for {experiment_name}",
            )
            plot_masks(
                mean_mask_new_pred[modality],
                f'Mean_{modality}_Prediction',
                save_directory=results_dir,
                title=f"Averaged{modality}  Masks per Prediction for {experiment_name}",
                ylabel= 'Prediction'
            )
        predictions_distr_new,predictions_distr_comparison,targets_distr = prediction_count(data, comparison_filepath)

        save_histogram_data(predictions_distr_new, predictions_distr_comparison, targets_distr, results_dir, experiment_name)
        

        # Add noise to the test set and evaluate the model
        noise_types = [ "gaussian", "dropout", "shuffle",'all'] #all noise types
        noise_modalities = ["text", "audio", "visual","all"] #all modalities
        noise_levels = {}
        noise_levels['high'] = [0.01, 0.15, 0.15, [0.01, 0.15, 0.15]]  # Fourth element is a list
        noise_levels['low'] = [0.1, 0.4, 0.4, [0.1, 0.4, 0.4]]

        for modality in noise_modalities:
            for level in ['high', 'low']:
                for i,noise in enumerate(noise_types):
                    # if we train with noise, we only want to test with the same noise and all types of noises
                    if(C['model']['noise_type'] != 'none' and C['model']['noise_type'] != noise and noise != 'all'):
                        continue
  
                    test_noise = add_noise(test,
                        noise_type=noise, 
                        noise_modality=modality, 
                        noise_level=noise_levels[level][i]
                    )
                    
                    test_loader = create_dataloader(test_noise)

                    predictions, targets, masks_txt, masks_au, masks_vi = trainer.predict(test_loader)

                    pred = torch.cat(predictions)
                    y_test = torch.cat(targets)
                    data = {
                        'predictions': pred.cpu().numpy(),      # Removed torch.cat
                        'targets': y_test.cpu().numpy(),        # Removed torch.cat
                        'masks_txt': [mask.cpu().numpy() for mask in masks_txt],
                        'masks_au': [mask.cpu().numpy() for mask in masks_au],
                        'masks_vi': [mask.cpu().numpy() for mask in masks_vi],
                    }

                    metrics = eval_mosei_senti(pred, y_test, True)
                    avg_metrics, _, _,_ = compare_masks(data, comparison_filepath)

                    print(f"Metrics on test set with {level} of {noise} noise at the {modality} modality")
                    print_metrics(metrics)
                    print_metrics(avg_metrics)

                    fname = f'results_{modality}_{noise}_{level}'
                    results_file = os.path.join(results_dir + f"/numeric_results", fname)
                    save_metrics(metrics, results_file)

                    fname2 = fname + "_masks"
                    results_file = os.path.join(results_dir + f"/numeric_results", fname2)
                    save_metrics(avg_metrics, results_file)

       



                

