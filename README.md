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
 
## ✅ 협업 과정에서 사용할 tool
### 1. github
### 2. Jira
### 3. slack
