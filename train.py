import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

from dataloader import DatasetLoader
from model_arch import Model
from config import *



def train():
    '''
        This is wrapper method for training the model.
    '''

    # Dataset Loaders
    train_loader = DatasetLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    print("### Training Dataset loaded from ", train_dataset)
    test_loader = DatasetLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    print("### Testing Dataset loaded from ", test_dataset)

    # Model Initialization
    model = Model()
    print("### Model Initialized")
    if model_last_state_epoch != 0:
        assert model_last_state != '', "Model last state must be given"
        model.load_state_dict(torch.load(model_last_state))
        print("### Model Loaded from epoch ", model_last_state_epoch)
    model.cuda()

    # Loss Function Initialization
    loss_func = nn.BCELoss()
    print("### Loss Function Initialized")

    # Optimizer Initialization
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    print("### Optimizer initialized")

    # Seed Selection
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    print("### Seed selection done")

    # Training Initialization
    print("### Starting Training ...")
    for epoch in tqdm(range(model_last_state_epoch, epoch_num)):
        model.train()
        for i, (image, target) in enumerate(tqdm(train_loader.load_dataset())):
            image = image.cuda()
            output = model(image)
            optimizer.zero_grad()
            loss = loss_func(output.squeeze(), target.float().cuda())
            loss.backward()
            optimizer.step()

            if i % 300 == 0:
                print("### Training Loss: ", loss.item())
                with open(os.path.join(logs_save_loc, 'train_loss.txt'), 'a') as f:
                    f.write('Training Loss at Epoch {} Iteration {}: {} \n'.format(epoch, i, loss.item()))
                f.close()
        
        torch.save(model.state_dict(), os.path.join(model_save_loc, 'model_{}.pth'.format(epoch)))
        
        print("############################ Evaluation #############################")
        model.eval()
        actual_target = []
        predicted_target = []
        with open(os.path.join(logs_save_loc, 'eval_accuracy.txt'), 'a') as f:
            for i, (image, target) in enumerate(tqdm(test_loader.load_dataset())):
                image = image.cuda()
                output = model(image)
                output = torch.where(output < 0.15, torch.zeros_like(output), torch.ones_like(output))
                actual_target.extend(target.float().cpu().tolist())
                predicted_target.extend(output.squeeze().cpu().tolist())
            acc = accuracy_score(actual_target, predicted_target, normalize=True)
            tn, fp, fn, tp = confusion_matrix(actual_target, predicted_target, labels=[0, 1]).ravel()
            apcer = fp/(tn + fp)
            bpcer = fn/(fn + tp)
            acer = (apcer + bpcer)/2
            
            print("Accuracy: %.4f, TN: %i, FP: %i, FN: %i, TP: %i, APCER: %.4f, BPCER: %.4f, ACER: %.4f" % (acc, tn, fp, fn, tp, apcer, bpcer, acer)) 
            f.write("Epoch : %i \n Accuracy: %.4f, TN: %i, FP: %i, FN: %i, TP: %i, APCER: %.4f, BPCER: %.4f, ACER: %.4f \n" % (epoch, acc, tn, fp, fn, tp, apcer, bpcer, acer))
        f.close()


