# EEG DATA ANALYSIS

## Dependency
- TF ver 1.x
- scipy
- numpy
- matplotlib
- urllib

## File Description
- datasetn_parsed.mat
    - 분석할 eeg data의 예시 형식이다.
    - ep: 4 x 3000 x N (# of channel, ms, # of trials)
    - lb: 1 x N (1, # of trials)
- model
    - MBMF(or AE, ...)로 사전학습된 웨이트가 저장될 폴더이다.
- eeg_classify.py
    - 다양한 Network의 classification 학습에 사용할 코드이다.
- move.py
    - eeg_classify.py 의 결과로 생성된 logs 폴더를 정리하여 result 폴더를 생성해주는 코드이다.
- run_tensorboard.py
    - 새로운 쓰레드에서 tensorboard를 구동시키고 계속 현재 터미널을 사용할 수 있게 해주는 코드이다.
    - 터미널에 6006포트에서 서버가 돌아간다고 출력될때까지 기다려야 한다.
    - tensorboard를 모두 사용한 후에는 ps aux | grep tensorboard 로 pid 를 찾아 수동으로 tensorboard server가 돌아가는 process를 kill 해줘야한다.
- ext.py
    - tensorboard가 돌아갈때 이 코드를 실행시키면 모든 csv를 추출하여 알맞은 위치에 정리해준다.
    - 간혹 데이터가 너무 많으면 바로 안돌아가는 현상이 발생하지만 당황하지 말고 다음 해결방법을 따라주면 된다.
        1. 브라우저를 켜서 localhost:6006 에 접속하여 모든 데이터가 로드 되기를 기다린다.
        1. 어느정도 기다린 후 ext.py 를 계속 반복적으로 실행시킨다. ext.py 를 실행시키고 에러메시지가 뜨지 않고 Done! 이라는 출력을 보게 된다면 모든 추출 과정이 성공적으로 끝난 것이다.
        1. 여러번의 실험결과 아무리 데이터가 커도 인내심을 가지고 브라우저를 킨 후 계속 ext.py 를 실행시키다보면 언젠가는 확실하게 완료된다.
- processing.py
    - 결과로 얻은 csv 데이터를 처리하고 분석하는 핵심 코드이다.
- save_figs.py
    - processing 모듈을 이용해 모든 데이터에 대해 분석 결과를 저장하는 코드이다.

## Work Flow

1. dataset 준비
1. weight 준비
1. python3 eeg_classify.py
1. python3 move.py
1. python3 run_tensorboard.py
1. python3 ext.py
1. python3 save_figs.py