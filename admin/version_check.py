import torch
import tensorflow as tf
import sklearn


if __name__ == '__main__':
    print(f"토치 버전 : {torch.__version__}")
    print(f"텐서플로 버전 : {tf.__version__}")
    print(f"사이킷런 버전 : {sklearn.__version__}")
    print(f"torch의 gpu 사용 가능 여부 확인 : {torch.cuda.is_available()}")
    print(f"사용가능한 gpu 보기 : {torch.cuda.get_device_name(0)}")