# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
import torchviz
from torchsummary import summary

def normalize(x):
	return x/np.linalg.norm(x)

winequality = pd.read_csv('winequality-red.csv', ';').values
winequality.astype(np.float32)
winequality = np.array(list(map(normalize, winequality.transpose()))).transpose()

train, validate, test = np.split(winequality, [int(.6*len(winequality)), int(.8*len(winequality))])

train_wine = torch.from_numpy(train[:,:-1]).type('torch.FloatTensor')
train_quality = torch.from_numpy(train[:,-1:]).type('torch.FloatTensor')

validate_wine = torch.from_numpy(validate[:,:-1]).type('torch.FloatTensor')
validate_quality = torch.from_numpy(validate[:,-1:]).type('torch.FloatTensor')

test_wine = torch.from_numpy(test[:,:-1]).type('torch.FloatTensor')
test_quality = torch.from_numpy(test[:,-1:]).type('torch.FloatTensor')

best_loss = 1
best_model = None
best_hidden = None

loss_fn = torch.nn.MSELoss(reduction='sum')

for hidden in range(3, 12, 1):
	N, D_in, H, D_out = int(.6*len(winequality)), 11, hidden, 1

	model = torch.nn.Sequential(
	    torch.nn.Linear(D_in, H),
	    torch.nn.ReLU(),
	    torch.nn.Linear(H, D_out),
	)
	
	prev_loss = 0
	learning_rate = 1e-4
	for t in range(500):
		pred_quality = model(train_wine)

		loss = loss_fn(pred_quality, train_quality)
		#print(t, loss.item())

		if abs(prev_loss - loss.item()) < 1e-8:
			break

		prev_loss = loss.item()

		model.zero_grad()

		loss.backward()

		with torch.no_grad():
			for param in model.parameters():
				param -= learning_rate * param.grad


	pred_quality = model(validate_wine)
	loss = loss_fn(pred_quality, validate_quality)
	print(hidden, loss.item())

	if loss < best_loss:
	    best_loss = loss
	    best_model = model
	    best_hidden = hidden
	    
pred_quality = best_model(test_wine)
loss = loss_fn(pred_quality, test_quality)
print(loss.item())

dot = torchviz.make_dot(pred_quality)
dot.format = 'jpeg'
dot.render('winequality-model')
summary(best_model, input_size=(1, best_hidden, 11))

