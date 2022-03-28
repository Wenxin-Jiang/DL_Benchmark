#!/bin/sh
python3 inference.py \
  --input /home/jiang784/Benchmark/dataset/coco/val2017/ \
  --model /home/jiang784/Benchmark/models/onnx/yolov4/yolov4.onnx \
  -a /home/jiang784/Benchmark/dataset/coco/annotations/instances_val2017.json \
  --stop 5 > output/output.txt

# python inference.py \
#   --input /home/lpmot/Dataset/COCO2017/val2017/ \
#   --model model/yolov4.onnx \
#   -a ../../Dataset/COCO2017/annotations/instances_val2017.json \
#   --stop 100 > output/output.txt

# python inference.py \
#   --input /home/lpmot/Dataset/COCO2017/val2017/ \
#   --model model/yolov4-416.onnx \
#   -a ../../Dataset/COCO2017/annotations/instances_val2017.json \
#   --stop 100\
#   --quantized >> output/output.txt

# python inference.py \
#   --input /home/lpmot/Dataset/COCO2017/val2017/ \
#   --model model/yolov4-416-fp16.onnx \
#   -a ../../Dataset/COCO2017/annotations/instances_val2017.json \
#   --stop 100\
#   --quantized >> output/output.txt

# python inference.py \
#   --input /home/lpmot/Dataset/COCO2017/val2017/ \
#   --model model/yolov4-416-int8.onnx \
#   -a ../../Dataset/COCO2017/annotations/instances_val2017.json \
#   --stop 100\
#   --quantized >> output/output.txt

# python inference.py \
#   --input /home/lpmot/Dataset/COCO2017/val2017/ \
#   --model model/yolov4-416-int16.onnx \
#   -a ../../Dataset/COCO2017/annotations/instances_val2017.json \
#   --stop 100\
#   --quantized >> output/output.txt
