import os
import math
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import argparse
from dataloader import data_loader
from evaluation import evaluation_metrics
from model import Net

'''
data
 \_ train
   \_ 0.tif, ..., train.csv
 \_ test
   \_ 0.tif, ..., test.csv
 \_ prediction.csv (test시 test 예측값 저장)

model
 \_ 1.pth, ...

 evaluation.py를 통해 예측값의 f1_socre 확인 가능

# 데이터, 모델 경로 수정 바랍니다.
'''


DATASET_PATH = os.path.join("/home/workspace/user-workspace/leeejihyun/data")


def _infer(model, cuda, data_loader):
    res_fc = None
    
    for index, (image, label) in enumerate(data_loader):
        if cuda:
            image = image.cuda()
        fc = model(image)
        fc = fc.detach().cpu().numpy()

        if index == 0:
            res_fc = fc
        else:
            res_fc = np.concatenate((res_fc, fc), axis=0)

    res_cls = np.argmax(res_fc, axis=1)
    #print('res_cls{}\n{}'.format(res_cls.shape, res_cls))

    return res_cls


def feed_infer(output_file, infer_func):
    prediction_class = infer_func()

    print('write output')
    predictions_str = []

    with open(output_file, 'w') as file_writer:
        for pred in prediction_class:
            file_writer.write("{}\n".format(pred))

    if os.stat(output_file).st_size == 0:
        raise AssertionError('output result of inference is nothing')


def test(prediction_file, model, test_dataloader, cuda):
    feed_infer(prediction_file, lambda : _infer(model, cuda, data_loader=test_dataloader))


def save_model(model_name, model, optimizer, scheduler):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(state, os.path.join(model_name + '.pth'))
    print('model saved')


def load_model(model_name, model, optimizer=None, scheduler=None):
    state = torch.load(os.path.join(model_name))
    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])
    print('model loaded')


if __name__ == '__main__':
    # mode argument
    args = argparse.ArgumentParser()
    args.add_argument("--num_classes", type=int, default=2)
    args.add_argument("--lr", type=float, default=0.001)
    args.add_argument("--cuda", type=bool, default=True)
    args.add_argument("--num_epochs", type=int, default=100)
    args.add_argument("--print_iter", type=int, default=100)
    args.add_argument("--model_name", type=str, default="/home/workspace/user-workspace/leeejihyun/model/model.pth")
    args.add_argument("--prediction_file", type=str, default="/home/workspace/user-workspace/leeejihyun/data/prediction.tsv")
    args.add_argument("--batch", type=int, default=32)
    args.add_argument("--mode", type=str, default="train")

    config = args.parse_args()

    num_classes = config.num_classes
    base_lr = config.lr
    cuda = config.cuda
    num_epochs = config.num_epochs
    print_iter = config.print_iter
    model_name = config.model_name
    prediction_file = config.prediction_file
    batch = config.batch
    mode = config.mode

    # create model
    model = Net(num_classes=num_classes)

    if mode == 'test':
        load_model(model_name, model)

    if cuda:
        model = model.cuda()

    if mode == 'train':
        # define loss function
        loss_fn = nn.CrossEntropyLoss()
        if cuda:
            loss_fn = loss_fn.cuda()

        # set optimizer
        optimizer = Adam(
            [param for param in model.parameters() if param.requires_grad],
            lr=base_lr, weight_decay=1e-4)
        scheduler = StepLR(optimizer, step_size=40, gamma=0.1)

        # get data loader
        train_dataloader, _ = data_loader(root=DATASET_PATH, phase='train', batch_size=batch)
        time_ = datetime.datetime.now()
        num_batches = len(train_dataloader)

        
        #check parameter of model
        print("------------------------------------------------------------")
        total_params = sum(p.numel() for p in model.parameters())
        print("num of parameter : ",total_params)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("num of trainable_ parameter :",trainable_params)
        print("------------------------------------------------------------")
        
        # train
        for epoch in range(num_epochs):
            model.train()
            for iter_, train_data in enumerate(train_dataloader):
                # fetch train data
                image, label = train_data
                
                
                if cuda:
                    image = image.cuda()
                    label = label.cuda()


                # update weight
                pred = model(image)
                loss = loss_fn(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if (iter_ + 1) % print_iter == 0:
                    elapsed = datetime.datetime.now() - time_
                    expected = elapsed * (num_batches / print_iter)
                    _epoch = epoch + ((iter_ + 1) / num_batches)
                    print('[{:.3f}/{:d}] loss({}) '
                          'elapsed {} expected per epoch {}'.format(
                              _epoch, num_epochs, loss.item(), elapsed, expected))
                    time_ = datetime.datetime.now()

            # scheduler update
            scheduler.step()

            # save model
            save_model('/home/workspace/user-workspace/leeejihyun/model/'+str(epoch + 1), model, optimizer, scheduler)

            elapsed = datetime.datetime.now() - time_
            print('[epoch {}] elapsed: {}'.format(epoch + 1, elapsed))

    elif mode == 'test':
        model.eval()
        # get data loader
        test_dataloader, test_label_file = data_loader(root=DATASET_PATH, phase='test', batch_size=batch)
        test(prediction_file, model, test_dataloader,cuda )
        # submit test result