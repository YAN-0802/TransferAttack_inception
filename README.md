# TransferAttack_inception
Add the Inception series of models that we use frequently to TransferAttack(Maybe have mistakes just for saving here))

## About
This code is modeled after [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack) and [tf_to_pytorch_model](https://github.com/ylhz/tf_to_pytorch_model) based on the common model (Inception series) pytorch framework, but the recurrence result is relatively large float, need to be checked and modified, to be improved.

## Usage
The basic operation is the same as [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack).
Some of the differences are explained below:
```
python main_dev.py --input_dir ./path/to/data --output_dir adv_data/ifgsm/resnet18 --model_dir ./path/to/model_dir/ --attack mifgsm --model=resnet18
python main_dev.py --input_dir ./path/to/data --output_dir adv_data/ifgsm/resnet18 --model_dir ./path/to/model_dir/ --eval
```

The dataset is changed, which is 1000 299×299 ImageNet images that are commonly used.
Torch_nets and Torch_nets_weight can be found at [tf_to_pytorch_model](https://github.com/ylhz/tf_to_pytorch_model). 

## Note
The result of code reproduction is not ideal, there may be errors. The author will check again if there is time in the future. If you are interested in communicating with me, please contact me: 2310747@stu.neu.edu.cn， and I would be honored to hear from you. Thanks again to [TransferAttack](https://github.com/Trustworthy-AI-Group/TransferAttack) and [tf_to_pytorch_model](https://github.com/ylhz/tf_to_pytorch_model) for their contribution. In addition, this code is only for private research, if you want to use it, please contact me for permission.
