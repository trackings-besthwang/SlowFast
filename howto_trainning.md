## 준비사항
- docker 환경에서 실행되는 것을 전제로 함(5080).
- AVA 데이터셋을 다운로드 되어 있어야함.
- `SlowFast` 디렉토리에서 `sudo python3 setup.py develop` 실행  
   관련 코드 위치를 등록시킴.   

## tranning 방법? 
- `./run_trainning.sh` 실행시킴. <br>
  이 스크립트는 `tools/run_net.py`를 실행 시킴.

``` bash
python tools/run_net.py \
  --cfg ./demo/AVA/SLOWFAST_32x2_R101_50_50.yaml \
  --opts \
  TRAIN.ENABLE True \
  DATA.PATH_TO_DATA_DIR  "/home/developer/AVA" \
  NUM_GPUS 1 \
  DEMO.ENABLE False \
  TRAIN.BATCH_SIZE 4 
```
`run_trainning.sh`에서  
- `DATA.PATH_TO_DATA_DIR` 옵션은 dataset의 위치를 설명.
-  `NUM_GPUS`  사용될 GPU 개수를 의미함.
-  `TRAIN.BATCH_SIZE` 한번에 GPU 메모리에 들어가는 batch의 개수

**세부 옵션 변경은** 
 `--cfg` 옵션의  `./demo/AVA/SLOWFAST_32x2_R101_50_50.yaml` 파일과
 `slowfast/config/default.py` 에서 설정함. 

##  trainning 결과
- weight 파일은 checkpoints 폴더에 저장됨.
- 파일의 이름은  checkpoint_epoch_00021.pyth 형식임.
  숫자는  epoch 횟수를 가르킴.
- 데모로 제공 파일 확장자가 *.pkl 이나, 여기서는 pyth임
  pyth 확장 형식도 동작하는걸 확인함. 심지어 pyth 확장자를 pkl로 변경해도 동작함.
  demo/demo_ava.ipynb 파일 참조


## action이 변경되었을 때 수정할 포인트
- `slowfast/config/default.py` 수정이 필요할 수 도 있음.
- .`/demo/AVA/SLOWFAST_32x2_R101_50_50.yaml` 수정 필요.

 아래 부분은 SLOWFAST_32x2_R101_50_50.yaml에서 `MODEL` 부분에서
`NUM_CLASSES`를 수정해야 함. <br>
행동이 10개가 추가되면 `NUM_CLASSES: 90`이 됨.
``` yml
MODEL:
  NUM_CLASSES: 80
  ARCH: slowfast
  MODEL_NAME: SlowFast
  LOSS_FUNC: bce
  DROPOUT_RATE: 0.5
  HEAD_ACT: sigmoid
```

- `NUM_CLASSES` 개수에 대응하도록 dataloader 수정

`tools/train_net.py`의 train() 함수에서 아래처럼 호출됨.

``` python
    train_loader = loader.construct_loader(cfg, "train")
    loader.shuffle_dataset(train_loader, cur_epoch)
```

- `slowfast/dataset/ava_dataset.py` 에서  `__getitem__` 만 보면 될것 같음.

`__getitem__` 메서드에서 label 데이터를 설정하는 부분
```
        label_arrs = np.zeros((len(labels), self._num_classes), dtype=np.int32)
        for i, box_labels in enumerate(labels):
            # AVA label index starts from 1.
            for label in box_labels:
                if label == -1:
                    continue
                assert label >= 1 and label <= 80
                label_arrs[i][label - 1] = 1
```

## pycharm debugging

- 설정 파라미터  
--cfg ./demo/AVA/SLOWFAST_32x2_R101_50_50.yaml --opts TRAIN.ENABLE True DATA.PATH_TO_DATA_DIR "/home/developer/AVA" NUM_GPUS 1 DEMO.ENABLE False TRAIN.BATCH_SIZE 16

- working directory  
/home/developer/SlowFast/