# AI를 활용한 똑똑한 노트 필기앱 "[APlus](https://github.com/APlus22)"

## ✅ APlus 배경
- 최근 아이패드, 갤럭시 탭 등의 태블릿을 학업 용도로 사용하는 학생들이 늘어나면서 다양한 노트 application이 인기를 끌고 있음.
- 수업 중에 이러한 노트 application을 사용할 때, 수업 내용을 따라가느라 빠르게 필기하기 때문에 나중에 글씨를 알아보기 힘들거나 다시 깔끔하게 정리해야하는 경우가 있음. <br>
**=> 사용자가 작성한 손 글씨를 자동으로 텍스트화해주는 노트 필기앱을 제작한다면, 사용자의 효율적인 노트 필기를 도울 수 있을 것 같다!**

<br>
 
## ✅ 주기능
### 1. 자동으로 사용자의 글씨를 텍스트화 (이용할 데이터: [한국어 글자체 이미지](https://aihub.or.kr/aidata/133), [textGenerator](https://github.com/Belval/TextRecognitionDataGenerator))
- 사용자의 손 글씨를 자동으로 텍스트화 (원하지 않을 경우 필기 데이터 그대로 보존)
- 텍스트화 시 사용자가 원하는 폰트 선택 가능 
- 텍스트화 진행 과정에서 맞춤법을 자동으로 수정
    
### 2. 노트의 템플릿과 디자인 추천
사용자가 작성한 노트 내용에 맞춰서 노트 속지, 글자/이미지 배치 등을 추천

 <br>
 
## ✅ project 진행과정에서 배우고자 하는 것
다음은 project 진행과정에서 배우고자 하는 내용이자 어떤 것을 공부해야할지 간단하게 정리해본 것이다.

### 1. 이미지 처리
- 광학 문자 인식(Optical character recognition; OCR) 기술
- 사용자의 필기를 Detection, Recognition [(관련영상)](https://tv.naver.com/v/11210453)
- [vgg model (2014)](https://arxiv.org/abs/1409.1556)
- json 파일 이해

### 2. 추천 시스템
- 콘텐츠 기반 필터링(content-based filtering)
- 협업 필터링(collaborative filtering)

### 3. 앱 개발 (React native)- Frontend/Backend
- react native와 db 공부 필요

<br>

## ✅ 내가 맡은 부분
- 손글씨 인식 모델 개발 부분을 맡았다. 
- 아래 사진은 streamlit을 통해 간단한 테스트 앱을 만들어 테스트를 해본 결과이다.
- test 결과 ㅇ과 ㅁ을 혼돈할 때도 있고, 예측에 실패할 때도 있지만, 꽤 높은 확률로 잘 인식을 하는 것을 확인할 수 있었다.

### Project run
```
npm start
```

<div>
<img src ="https://user-images.githubusercontent.com/55095806/167298672-73a5af6c-7a8e-489d-a727-e249e890d31d.png" width = "250">
<img src ="https://user-images.githubusercontent.com/55095806/167298675-e717c35b-f87f-461d-9ceb-209a2c17f015.png" width = "250">
</div>

<div>
<img src ="https://user-images.githubusercontent.com/55095806/167298678-2bd99a9f-d729-4a8e-b44a-87e4e84fd7f1.png" width = "250">
<img src ="https://user-images.githubusercontent.com/55095806/167298681-cc0295e8-c726-4aff-b319-cb5ac7f0fa21.png" width = "250">
<img src ="https://user-images.githubusercontent.com/55095806/167298685-8d000bcb-957d-4b79-be4a-818e67ed3804.png" width = "250">
 </div>

<br>
 
## ✅ 협업 과정에서 사용할 tool
### 1. github
### 2. Jira
### 3. slack
