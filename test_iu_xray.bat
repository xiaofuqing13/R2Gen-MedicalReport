@echo off

:: 强制激活 Conda 环境
call conda activate r2gen

:: 检查 Python 路径
where python

:: 运行主程序
python main_test.py ^
 --image_dir data\iu_xray\images ^
 --ann_path data\iu_xray\annotation.json ^
 --dataset_name iu_xray ^
 --max_seq_length 60 ^
 --threshold 3 ^
 --batch_size 4^
 --epochs 100 ^
 --save_dir results\iu_xray ^
 --step_size 50 ^
 --gamma 0.1 ^
 --seed 9223 ^
 --load data\model_iu_xray.pth

pause