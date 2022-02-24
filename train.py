import os
import torch
from torch import nn
from torch import optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

from model.vgg import VGG
from custom_dataset import path_to_img

if __name__ == '__main__': # 인터프리터에서 직접 실행했을 경우에만 if문 내의 코드를 돌리라는 의미

    # ------------------------------------------
    # 파라미터 설정

    # 전체 데이터를 몇 번이나 볼 것인지
    start_epoch = 1
    epoch_num = 10

    # 학습 시 한번에 몇 개의 데이터를 볼 것인지
    batch_size = 256

    # 검증 데이터 비율
    val_percent = 0.05

    # 학습률
    lr = 0.001
    
    # 이미지 크기 조정
    img_size = 32
    
    # 체크포인트 저장 경로
    checkpoint_dir = '/home/danbibibi/jupyter/checkpoint_dir/' # gpu 서버
    # checkpoint_dir = '/Users/dan_bibibi/Downloads/Capstone/checkpoint_dir/' # local

    # 학습 재개 시 resume = True, resume_checkpoint='재개할 체크포인트 경로'
    resume = False
    resume_checkpoint = ''
    # ------------------------------------------
    
    # gpu 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    
    # model 생성
    model = VGG(input_channel=3, num_class=2350) # 한글 완성형 2350자
    model.to(device)

    # 최적화 기법 및 손실 함수
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss() # 다중 분류 (nn.LogSoftmax + nn.NLLLoss)

    train_loss_list = []
    val_loss_list = []
    
    # 디렉토리 생성 ( exist_ok=True, 해당 디렉토리가 없을 경우에만 생성)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ------------------------------------------
    # 이전 체크포인트로부터 모델 로드
    if resume:
        print("start model load...")

        # 체크포인트 로드
        checkpoint = torch.load(resume_checkpoint, map_location=device)

        # 각종 파라미터 로드
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        train_loss_list = checkpoint['train_loss_list']
        val_loss_list = checkpoint['val_loss_list']
        start_epoch = checkpoint['epoch'] + 1
        batch_size = checkpoint['batch_size']

        print("model load end. start epoch : ", start_epoch)
    # ------------------------------------------

    # ------------------------------------------
    # 이미지 변형
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
#         transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) # 정규화
    ])

    # 데이터셋 로드
    train_data = pd.read_csv('./data/hangeul_2350.csv')

    # 학습 테스트 데이터 분할
    train_X = train_data['img_path']  # img_path
    train_y = train_data['label']  # label
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=val_percent)

    # dataset
    train_datasets = path_to_img(img_path=train_X, labels=train_y, transform=transform)
    valid_datasets = path_to_img(img_path=val_X, labels=val_y, transform=transform)

    # DataLoader
    train_loader = DataLoader(dataset=train_datasets, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_datasets, batch_size=batch_size, shuffle=True)

    # ------------------------------------------
    
    for epoch in range(start_epoch, epoch_num+1): # 에폭 한 번마다 전체 데이터를 봄
        print('[epoch %d]' % epoch)

        train_loss = 0.0
        valid_loss = 0.0

        # 학습
        model.train()
        for img, label in train_loader:
            img = img.to(device)
            label = label.to(device)
            
#             print(img)
#             print(label)
#             print(img.shape)
#             print(label.shape)
#             break
                
            out = model(img)

            # loss 계산
            loss = criterion(out, label)
            train_loss = train_loss + loss.item()

            # 가중치 갱신
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = train_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)
        print('train loss : %f' % (avg_train_loss))

        # 검증
        model.eval()
        with torch.no_grad():
            for img, label in valid_loader:
                
                img = img.to(device)
                label = label.to(device)
                
                out = model(img)

                loss = criterion(out, label)
                valid_loss = valid_loss + loss.item()

            avg_val_loss = valid_loss / len(valid_loader)
            val_loss_list.append(avg_val_loss)
            print('validation loss : %f' % (avg_val_loss))
            
            
            # 최적의 모델 저장
            if epoch<2:
                print('first model save...')
                torch.save(model.state_dict(), '/home/danbibibi/jupyter/model/handwrite_recognition.pt')
            else:
                if val_loss_list[-1] < val_loss_list[-2]:
                    print('better model save...')
                    torch.save(model.state_dict(), '/home/danbibibi/jupyter/model/handwrite_recognition.pt')

        # 체크포인트 저장
        checkpoint_name = checkpoint_dir + '{:d}_checkpoint.pth'.format(epoch)
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'train_loss_list': train_loss_list,
            'val_loss_list': val_loss_list,
            'batch_size': batch_size
        }

        torch.save(checkpoint, checkpoint_name)
        print('checkpoint saved : ', checkpoint_name)

    # 모델 저장
#     print('model save...')
#     torch.save(model.state_dict(), '/home/danbibibi/jupyter/model/handwrite_recognition.pt')
    
    # 학습 그래프 그리기
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.title("train loss")
    x1 = np.arange(0, len(train_loss_list))
    plt.plot(x1, train_loss_list)

    plt.subplot(1, 2, 2)
    plt.title("validation loss")
    x2 = np.arange(0, len(val_loss_list))
    plt.plot(x2, val_loss_list)
    plt.show()