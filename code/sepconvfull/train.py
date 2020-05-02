from TorchDB import DBreader_frame_interpolation
from torch.utils.data import DataLoader
from model import SepConvNet
import argparse
from torchvision import transforms
import torch
from torch.autograd import Variable
import os
from TestModule import Middlebury_other

parser = argparse.ArgumentParser(description='SepConv Pytorch')

# parameters
parser.add_argument('--train', type=str, default='./db')
parser.add_argument('--kernel', type=int, default=51)
parser.add_argument('--out_dir', type=str, default='./output_sepconv_pytorch')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--load_model', type=str, default=None)
parser.add_argument('--test_input', type=str, default='./Interpolation_testset/input')
parser.add_argument('--gt', type=str, default='./Interpolation_testset/gt')

transform = transforms.Compose([transforms.ToTensor()])


def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def main():
    args = parser.parse_args()
    db_dir = args.train

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    result_dir = args.out_dir + '/result'
    ckpt_dir = args.out_dir + '/checkpoint'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    logfile = open(args.out_dir + '/log.txt', 'w')
    logfile.write('batch_size: ' + str(args.batch_size) + '\n')

    total_epoch = args.epochs
    batch_size = args.batch_size

    dataset = DBreader_frame_interpolation(db_dir, resize=(128, 128))
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    TestDB = Middlebury_other(args.test_input, args.gt)
    test_output_dir = args.out_dir + '/result'

    if args.load_model is not None:
        checkpoint = torch.load(args.load_model)
        kernel_size = checkpoint['kernel_size']
        model = SepConvNet(kernel_size=kernel_size)
        state_dict = checkpoint['state_dict']
        model.load_state_dict(state_dict)
        model.epoch = checkpoint['epoch']
    else:
        kernel_size = args.kernel
        model = SepConvNet(kernel_size=kernel_size)

    logfile.write('kernel_size: ' + str(kernel_size) + '\n')

    if torch.cuda.is_available():
        model = model.cuda()

    max_step = train_loader.__len__()

    model.eval()
    TestDB.Test(model, test_output_dir, logfile, str(model.epoch.item()).zfill(3) + '.png')

    while True:
        if model.epoch.item() == total_epoch:
            break
        model.train()
        for batch_idx, (frame0, frame1, frame2) in enumerate(train_loader):
            frame0 = to_variable(frame0)
            frame1 = to_variable(frame1)
            frame2 = to_variable(frame2)
            loss = model.train_model(frame0, frame2, frame1)
            if batch_idx % 100 == 0:
                print('{:<13s}{:<14s}{:<6s}{:<16s}{:<12s}{:<20.16f}'.format('Train Epoch: ', '[' + str(model.epoch.item()) + '/' + str(total_epoch) + ']', 'Step: ', '[' + str(batch_idx) + '/' + str(max_step) + ']', 'train loss: ', loss.item()))
        model.increase_epoch()
        if model.epoch.item() % 1 == 0:
            torch.save({'epoch': model.epoch, 'state_dict': model.state_dict(), 'kernel_size': kernel_size}, ckpt_dir + '/model_epoch' + str(model.epoch.item()).zfill(3) + '.pth')
            model.eval()
            TestDB.Test(model, test_output_dir, logfile, str(model.epoch.item()).zfill(3) + '.png')
            logfile.write('\n')

    logfile.close()


if __name__ == "__main__":
    main()
