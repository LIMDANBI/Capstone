import os
import torch
import torchsummary
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
    epoch_num = 15

    # 학습 시 한번에 몇 개의 데이터를 볼 것인지
    batch_size = 256

    # 검증 데이터 비율
    val_percent = 0.05

    # 학습률
    lr = 0.001
    
    # 이미지 크기 조정
    img_size = 32
    
    # validation loss가 가장 좋은 model을 저장
    min_loss = 0
    
    # 학습 그래프 출력에 사용
    train_loss_list = []
    val_loss_list = []
    # ------------------------------------------
    
    # gpu 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print()
    
    # model 생성
    model = VGG(input_channel=3, num_class=2350) # 한글 완성형 2350자
    model.to(device)
    torchsummary.summary(model, (3, img_size, img_size)) # model 정보 
    print()

    # 최적화 기법 및 손실 함수
    optimizer = optim.Adam(params=model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss() # 다중 분류 (nn.LogSoftmax + nn.NLLLoss)

    # ------------------------------------------
    # 이미지 변형
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ])

    # 데이터셋 로드
    train_data = pd.read_csv('./data/hangeul_2350.csv')
    print('data size is', len(train_data))
    print()

    # 학습 테스트 데이터 분할
    train_X = train_data['img_path']  # img_path
    train_y = train_data['label']  # label
    
    train_X, val_X, train_y, val_y = train_test_split(train_X, train_y, test_size=val_percent)

    # dataset
    train_datasets = path_to_img(img_path=train_X, labels=train_y, transform=transform)
    valid_datasets = path_to_img(img_path=val_X, labels=val_y, transform=transform)

    # DataLoader
    train_loader = DataLoader(dataset=train_datasets, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset=valid_datasets, batch_size=batch_size)

    # ------------------------------------------
    
    for epoch in range(start_epoch, epoch_num+1): # 에폭 한 번마다 전체 데이터를 봄
        print('[epoch %d]' % epoch)

        train_loss = 0.0
        train_acc = 0.0
        train_total = 0
        
        valid_loss = 0.0
        valid_acc = 0.0
        valid_total = 0

        # 학습
        model.train()
        for img, label in train_loader:
            img = img.to(device)
            label = label.to(device)
            
            out = model(img)
            _, predicted = torch.max(out, 1) 

            # loss 계산
            loss = criterion(out, label)
            train_loss = train_loss + loss.item()
            train_total += out.size(0) 
            train_acc += (predicted == label).sum()

            # 가중치 갱신
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        train_acc = train_acc / train_total
        avg_train_loss = train_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)
        
        print('train loss : %.4f || train accuracy: %.4f' % (avg_train_loss, train_acc))

        # 검증
        model.eval()
        with torch.no_grad():
            for img, label in valid_loader:
                
                img = img.to(device)
                label = label.to(device)
                
                out = model(img)
                _, predicted = torch.max(out, 1) 

                loss = criterion(out, label)
                valid_loss = valid_loss + loss.item()
                valid_total += out.size(0) 
                valid_acc += (predicted == label).sum()

            valid_acc = valid_acc / valid_total
            avg_val_loss = valid_loss / len(valid_loader)
            val_loss_list.append(avg_val_loss)
    
            print('valid loss : %.4f || valid accuracy: %.4f' % (avg_val_loss , valid_acc))
            
            
        # 최적의 모델 저장
        if epoch<2:
            min_loss = val_loss_list[-1]
            print('first model save...')
            torch.save(model.state_dict(), '/home/danbibibi/jupyter/model/handwrite_recognition.pt')
        else:
            if val_loss_list[-1] < min_loss:
                min_loss = val_loss_list[-1]
                print('better model save...')
                torch.save(model.state_dict(), '/home/danbibibi/jupyter/model/handwrite_recognition.pt')
        print()

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