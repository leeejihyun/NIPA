<제출 파일>
submission.zip

1. 모델 학습 코드 & 모델 예측 코드
main.py : baseline에서 제공하는 main.py에서 validate 없이 작동하도록 수정
dataloader.py : baseline과 동일
evaluation.py : baseline과 동일
model.py : baseline과 동일

2. 모델 파일
model.pth : data augmentation한 상태에서 main.py로 모델 학습시킨 후 저장한 epoch 100번째 모델 파일

3. Readme
제출 파일 및 학습, 예측을 위한 실행방법 설명

4. data augmentation 코드
data_augmentation.py : train 이미지 1장당 4장씩 증강

5. 모델 예측 파일
prediction.tsv

<학습 및 예측을 위한 실행방법>
1. 자신의 workspace에 data, baseline, model, prediction 폴더 생성
2. data 폴더 안에 train, test 폴더 생성
3. 제출 파일 1번과 4번에 대해 자신의 workspace에 맞춰 경로 수정
4. 제출 파일 4번 전체 실행하여 데이터 증강
5. !python /home/workspace/user-workspace/leeejihyun/baseline/main_no_val.py로 모델 학습
6. !python /home/workspace/user-workspace/leeejihyun/baseline/main_no_val.py --mode test --model_name /home/workspace/user-workspace/leeejihyun/model/100.pth --prediction_file /home/workspace/user-workspace/leeejihyun/prediction/prediction.tsv로 모델 예측
